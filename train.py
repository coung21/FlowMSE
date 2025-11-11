import torch
import torch.optim as optim
from tqdm import tqdm
import yaml
import wandb
from datetime import datetime
import argparse
import os

from src.data import get_dataloader
from src.model import loss_function, ConvVAE

def _complex_to_2channels(spec: torch.Tensor) -> torch.Tensor:
    """Convert complex STFT [B, F, T] to 2-channel real/imag tensor [B, 2, F, T]."""
    if torch.is_complex(spec):
        real = spec.real
        imag = spec.imag
    else:
        # If already real (e.g., 2-channel provided upstream), assume last dim=2 and convert
        raise ValueError("Expected complex STFT tensor. Got non-complex tensor.")
    return torch.stack([real, imag], dim=1)


def _center_crop_to_multiple(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Center-crop spatial dims [B, C, F, T] so F and T are divisible by `multiple`."""
    assert x.dim() == 4, f"Expected 4D tensor [B, C, F, T], got {x.shape}"
    B, C, F, T = x.shape
    new_F = F - (F % multiple)
    new_T = T - (T % multiple)
    # Ensure minimum > 0
    if new_F == 0 or new_T == 0:
        raise ValueError(f"Frequency/Time dims too small to crop to multiple {multiple}: {F}x{T}")
    f_start = (F - new_F) // 2
    t_start = (T - new_T) // 2
    return x[:, :, f_start:f_start + new_F, t_start:t_start + new_T]


def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)

    wandb.init(project=config['wandb']['project_name'],
               name=config['wandb']['run_name'],
                config=config)

    print("Loading data...")
    dataloader = get_dataloader(config=config, mode='train')

    # Peek one batch to infer input sizes for ConvVAE
    peek_batch = next(iter(dataloader))
    source_spec, target_spec = peek_batch  # [B, F, T] complex

    # Convert to 2-channel [B, 2, F, T] and crop for downsampling-by-16
    with torch.no_grad():
        src_2ch = _complex_to_2channels(source_spec)
        tgt_2ch = _complex_to_2channels(target_spec)
        src_2ch = _center_crop_to_multiple(src_2ch, multiple=16)
        tgt_2ch = _center_crop_to_multiple(tgt_2ch, multiple=16)

    in_channels = src_2ch.shape[1]
    out_channels = tgt_2ch.shape[1]
    input_f, input_t = src_2ch.shape[2], src_2ch.shape[3]

    # Create model
    model = ConvVAE(
        in_channels=in_channels,
        out_channels=out_channels,
        n_features=64,
        z_dim=128,
        input_f=input_f,
        input_t=input_t,
    ).to(device)

    # Optimizer and hyperparams
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    beta = config['train'].get('vae_beta', 1e-3)

    n_epochs = config['train']['n_epochs']
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()

        train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")

        for batch in train_pbar:
            source_spec, target_spec = batch  # [B, F, T] complex

            # Convert to model input format on CPU first, then move to device
            noisy = _center_crop_to_multiple(_complex_to_2channels(source_spec), multiple=16).to(device)
            clean = _center_crop_to_multiple(_complex_to_2channels(target_spec), multiple=16).to(device)

            optimizer.zero_grad()

            recon, mu, log_var = model(noisy)
            loss = loss_function(recon, clean, mu, log_var, beta)

            # Normalize loss by batch size to stabilize when using sum in loss_function
            loss = loss / noisy.shape[0]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            wandb.log({'train/loss': loss.item(), 'train/epoch': epoch})
    # Ensure save directory exists
    save_dir = config['train']['save_path']
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'flowmse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    torch.save(model.state_dict(), save_file)
    print("Model saved to", save_file)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()

    train(config_path=args.config)