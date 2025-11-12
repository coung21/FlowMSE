import torch
import torch.optim as optim
from tqdm import tqdm
import yaml
import wandb
from datetime import datetime
import argparse
import os

from src.data import get_dataloader
from src.model import (
    SpeechEnhancementVAE,
    VAEComplexLoss,
    complex_mse,
    kl_loss,
)


def _center_crop_complex(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """
    Center-crop complex STFT [B, F, T] để F và T chia hết cho `multiple`.
    """
    assert x.dim() == 3, f"Expected 3D complex tensor [B, F, T], got {x.shape}"
    B, F, T = x.shape
    new_F = F - (F % multiple)
    new_T = T - (T % multiple)
    if new_F == 0 or new_T == 0:
        raise ValueError(
            f"Frequency/Time dims too small to crop to multiple {multiple}: {F}x{T}"
        )
    f_start = (F - new_F) // 2
    t_start = (T - new_T) // 2
    return x[:, f_start:f_start + new_F, t_start:t_start + new_T]


def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    wandb.init(
        project=config['wandb']['project_name'],
        name=config['wandb']['run_name'],
        config=config
    )

    print("Loading data...")
    dataloader = get_dataloader(config=config, mode='train')

    # Tạo model & loss
    model = SpeechEnhancementVAE(z_dim=config['train'].get('z_dim', 128)).to(device)
    criterion = VAEComplexLoss(beta=config['train'].get('vae_beta', 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])

    n_epochs = config['train']['n_epochs']
    loss_reduction = config['train'].get('loss_reduction', 'mean')  # để tương thích log, không ảnh hưởng criterion

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")

        for batch in train_pbar:
            source_spec, target_spec = batch  # [B, F, T], complex tensors

            if not torch.is_complex(source_spec) or not torch.is_complex(target_spec):
                raise ValueError("Expected complex STFT tensors from dataloader.")

            # Crop để khớp kiến trúc down/up (stride 2 bốn lần -> bội số 16)
            noisy = _center_crop_complex(source_spec, multiple=16).to(device)
            clean = _center_crop_complex(target_spec, multiple=16).to(device)

            optimizer.zero_grad()

            # Forward: model nhận complex [B, F, T], trả về complex + (mu, logvar)
            recon, mu, log_var = model(noisy)

            loss, rec, kld = criterion(recon, clean, mu, log_var)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            # Logging
            with torch.no_grad():
                # rec đã là complex MSE; kld đã là KL
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mse': rec.item(),
                    'train/kld': kld.item(),
                    'train/epoch': epoch
                })

    # Lưu model
    save_dir = config['train']['save_path']
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'se_vae_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    torch.save(model.state_dict(), save_file)
    print("Model saved to", save_file)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Enhancement ConvVAE (complex STFT)")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()

    train(config_path=args.config)
