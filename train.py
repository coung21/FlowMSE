import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml
import wandb

from src.data import get_dataloader
# đổi import sang ConvAE; nếu bạn để class này trong src.model
from src.model import SpeechEnhancementConvAE


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


def _complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """
    [B, F, T] (complex) -> [B, 2, F, T] (real, imag)
    """
    assert torch.is_complex(x), "Expected complex tensor."
    return torch.stack([x.real, x.imag], dim=1)


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

    # ===== Model & Optimizer =====
    # Không đổi file config: vẫn dùng các trường cũ; z_dim/vae_beta sẽ bị bỏ qua.
    model = SpeechEnhancementConvAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])

    n_epochs = int(config['train']['n_epochs'])
    grad_clip = 1.0  # giống script cũ

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")

        for batch in train_pbar:
            source_spec, target_spec = batch  # [B, F, T], complex tensors

            if not torch.is_complex(source_spec) or not torch.is_complex(target_spec):
                raise ValueError("Expected complex STFT tensors from dataloader.")

            # Crop để khớp kiến trúc (4 lần /2 -> bội số 16)
            noisy_c = _center_crop_complex(source_spec, multiple=16).to(device)
            clean_c = _center_crop_complex(target_spec, multiple=16).to(device)

            # Đưa về 2 kênh (real, imag) cho ConvAE
            noisy = _complex_to_ri(noisy_c)   # [B, 2, F, T]
            clean = _complex_to_ri(clean_c)   # [B, 2, F, T]

            optimizer.zero_grad(set_to_none=True)

            pred = model(noisy)               # [B, 2, F, T]
            loss = F.mse_loss(pred, clean, reduction='mean')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Log: giữ key cũ để bảng W&B không gãy (kld=0.0)
            wandb.log({
                'train/loss': loss.item(),
                'train/epoch': epoch
            })

    # ===== Save =====
    save_dir = config['train']['save_path']
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'se_convae_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    torch.save(model.state_dict(), save_file)
    print("Model saved to", save_file)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Enhancement Conv Autoencoder (complex STFT)")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()
    train(config_path=args.config)
