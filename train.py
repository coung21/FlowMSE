import torch
import torch.optim as optim
from tqdm import tqdm
import yaml
import wandb
from datetime import datetime
import argparse
import os

from src.data import get_dataloader
from src.model import UNet
from src.flow import FlowMatchingLoss

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

    model = UNet(in_channels=2, out_channels=2).to(device)


    loss_fn = FlowMatchingLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    n_epochs = config['train']['n_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()

        train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}")

        for batch in train_pbar:
            source_spec, target_spec = batch
            source_spec = source_spec.to(device)  # [B, F, T]
            target_spec = target_spec.to(device)  # [B, F, T]

            optimizer.zero_grad()

            
            loss = loss_fn(model, source_spec, target_spec)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()

            train_pbar.set_postfix({'loss': loss.item()})

            wandb.log({'train/loss': loss.item()})
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