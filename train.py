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
from src.model import GeneratorConv, DiscriminatorConvPatch  # ← GAN models


# =========================
# Utils
# =========================
def _center_crop_complex(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """
    Center-crop complex STFT [B, F, T] để F và T chia hết cho `multiple`.
    """
    assert x.dim() == 3 and torch.is_complex(x), f"Expected complex [B,F,T], got {x.shape}, complex={torch.is_complex(x)}"
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


# =========================
# Hinge GAN losses
# =========================
def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake

def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


# =========================
# Train
# =========================
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

    # ===== Models =====
    # Generator sinh residual theo mặc định (residual_out=True)
    G = GeneratorConv(residual_out=True).to(device)
    D = DiscriminatorConvPatch().to(device)

    # ===== Optims =====
    lr = float(config['train']['learning_rate'])
    opt_G = optim.AdamW(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.AdamW(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # ===== Schedules / misc (optional) =====
    n_epochs = int(config['train']['n_epochs'])
    grad_clip = float(config['train'].get('grad_clip', 1.0))
    lambda_l1 = float(config['train'].get('gan_l1_lambda', 100.0))  # mặc định như pix2pix
    d_updates = int(config['train'].get('gan_d_updates', 1))        # số lần update D / step
    g_updates = int(config['train'].get('gan_g_updates', 1))        # số lần update G / step
    amp_enabled = bool(config['train'].get('amp', True)) and device.type == 'cuda'
    scaler_G = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scaler_D = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")

        for batch in train_pbar:
            # Dataloader trả về complex STFT: [B, F, T]
            source_spec, target_spec = batch
            if not torch.is_complex(source_spec) or not torch.is_complex(target_spec):
                raise ValueError("Expected complex STFT tensors from dataloader.")

            # Crop để bội số 16 (4 lần /2)
            noisy_c = _center_crop_complex(source_spec, multiple=16).to(device)
            clean_c = _center_crop_complex(target_spec, multiple=16).to(device)

            # Chuyển sang 2 kênh real/imag cho Conv models
            noisy_ri = _complex_to_ri(noisy_c)   # [B,2,F,T]
            clean_ri = _complex_to_ri(clean_c)   # [B,2,F,T]

            # ---------------------
            # 1) Update D
            # ---------------------
            for _ in range(max(1, d_updates)):
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    with torch.no_grad():
                        fake_ri = G(noisy_ri)
                    real_logits = D(noisy_ri, clean_ri)
                    fake_logits = D(noisy_ri, fake_ri)
                    loss_D = d_hinge_loss(real_logits, fake_logits)

                opt_D.zero_grad(set_to_none=True)
                scaler_D.scale(loss_D).backward()
                if grad_clip and grad_clip > 0:
                    scaler_D.unscale_(opt_D)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=grad_clip)
                scaler_D.step(opt_D)
                scaler_D.update()

            # ---------------------
            # 2) Update G
            # ---------------------
            for _ in range(max(1, g_updates)):
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    fake_ri = G(noisy_ri)
                    fake_logits = D(noisy_ri, fake_ri)
                    loss_G_gan = g_hinge_loss(fake_logits)
                    loss_G_l1 = F.l1_loss(fake_ri, clean_ri, reduction='mean')
                    loss_G = loss_G_gan + lambda_l1 * loss_G_l1

                opt_G.zero_grad(set_to_none=True)
                scaler_G.scale(loss_G).backward()
                if grad_clip and grad_clip > 0:
                    scaler_G.unscale_(opt_G)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=grad_clip)
                scaler_G.step(opt_G)
                scaler_G.update()

            # ---------------------
            # Logging
            # ---------------------
            train_pbar.set_postfix({
                'D': f"{loss_D.item():.4f}",
                'G': f"{loss_G.item():.4f}",
                'L1': f"{loss_G_l1.item():.4f}"
            })
            wandb.log({
                'train/loss_D': loss_D.item(),
                'train/loss_G': loss_G.item(),
                'train/loss_G_gan': loss_G_gan.item(),
                'train/loss_G_l1': loss_G_l1.item(),
                'train/epoch': epoch
            })

    # ===== Save (chỉ cần G cho inference) =====
    save_dir = config['train']['save_path']
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    gen_path = os.path.join(save_dir, f'gan_G_{ts}.pth')
    disc_path = os.path.join(save_dir, f'gan_D_{ts}.pth')

    torch.save(G.state_dict(), gen_path)
    torch.save(D.state_dict(), disc_path)
    print("Generator saved to", gen_path)
    print("Discriminator saved to", disc_path)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN (GeneratorConv + Patch Discriminator) for Speech Enhancement (complex STFT)")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()
    train(config_path=args.config)
