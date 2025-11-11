import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv_block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.down = ConvBlock(in_channels, out_channels )

    def forward(self, x):
        x = self.pool(x)
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels) 

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class ConvVAE(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=64, z_dim=128, input_f=256, input_t=128): # type: ignore
        super(ConvVAE, self).__init__()

        # Input: [B, 2, 128, 64]
        self.inc = ConvBlock(in_channels, n_features) # [B, 64, 128, 64]
        self.down1 = DownBlock(n_features, n_features * 2)  # [B, 128, 64, 32]
        self.down2 = DownBlock(n_features * 2, n_features * 4)  # [B, 256, 32, 16]
        self.down3 = DownBlock(n_features * 4, n_features * 8)  # [B, 512, 16, 8]
        self.down4 = DownBlock(n_features * 8, n_features * 8)  # [B, 512, 8, 4]
        
        # --- VAE Bottleneck ---
        self.bottleneck_f = input_f // (2**4) # 128 / 16 = 8
        self.bottleneck_t = input_t // (2**4) # 64 / 16 = 4
        self.bottleneck_channels = n_features * 8 # 512
        self.bottleneck_dim = self.bottleneck_channels * self.bottleneck_f * self.bottleneck_t # 512 * 8 * 4 = 16384
        
        self.fc_mu = nn.Linear(self.bottleneck_dim, z_dim)
        self.fc_log_var = nn.Linear(self.bottleneck_dim, z_dim)
        self.fc_z = nn.Linear(z_dim, self.bottleneck_dim)

        # Bắt đầu từ bottleneck [B, 512, 8, 4]
        self.dec_up1 = UpBlock(self.bottleneck_channels, n_features * 8) # [B, 512, 16, 8]
        self.dec_up2 = UpBlock(n_features * 8, n_features * 4)   # [B, 256, 32, 16]
        self.dec_up3 = UpBlock(n_features * 4, n_features * 2)   # [B, 128, 64, 32]
        self.dec_up4 = UpBlock(n_features * 2, n_features)       # [B, 64, 128, 64]
        
        self.outc = nn.Conv2d(n_features, out_channels, kernel_size=1) # [B, 2, 128, 64]

    def encode(self, x_noisy):
        x = self.inc(x_noisy)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x5 = self.down4(x)
        
        x_flat = torch.flatten(x5, start_dim=1)
        mu = self.fc_mu(x_flat)
        log_var = self.fc_log_var(x_flat)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_z(z)
        x = x.view(-1, self.bottleneck_channels, self.bottleneck_f, self.bottleneck_t) # Un-flatten
        
        x = self.dec_up1(x)
        x = self.dec_up2(x)
        x = self.dec_up3(x)
        x = self.dec_up4(x)
        
        recon_clean_spec = self.outc(x) 
        return recon_clean_spec

    def forward(self, x_noisy):
        mu, log_var = self.encode(x_noisy)
        z = self.reparameterize(mu, log_var)
        recon_clean = self.decode(z)
        return recon_clean, mu, log_var
   

def loss_function(recon_clean_spec, clean_spec_target, mu, log_var, beta):

    MSE = F.mse_loss(recon_clean_spec, clean_spec_target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + beta * KLD
    
