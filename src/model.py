import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t): #t: [B] 
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1) # ln(10000) / (d/2 - 1) - scaling factor scalar
        #freq_i = exp(-ln(10000 * i / (d/2 - 1))) = 10000^(-i/(d/2-1))
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings) # [d/2]

        # broadcasting
        embeddings = t[:, None] * embeddings[None, :] # [B, d/2]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) # [B, d]

        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1)) # [B, d+1]

        return embeddings  # [B, d]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ConvBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, time_emb):  # x: [B, in_channels, F, T], time_emb: [B, time_emb_dim]
        x = self.conv_block1(x)  # [B, out_channels, F, T]
        time_emb = self.time_proj(time_emb)  # [B, out_channels]

        x = x + time_emb[:, :, None, None]  # broadcast to [B, out_channels, F, T]
        x = self.conv_block2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.down = ConvBlock(in_channels, out_channels, time_emb_dim)

    def forward(self, x, time_emb):
        x = self.pool(x)
        return self.down(x, time_emb)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super(UpBlock, self).__init__()
        # Upsample from in_channels -> out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat with skip: channels = out_channels + skip_channels
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, time_emb_dim)

    def forward(self, x1, x2, time_emb):
        # x1: current features, x2: skip connection features
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, time_emb)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=64, time_emb_dim=128):
        super(UNet, self).__init__()

        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)

        # Encoder
        self.inc = ConvBlock(in_channels, n_features, time_emb_dim)  # [B, 2, F, T] -> [B, 64, F, T]
        self.down1 = DownBlock(n_features, n_features * 2, time_emb_dim)  # -> [B, 128, F/2, T/2]
        self.down2 = DownBlock(n_features * 2, n_features * 4, time_emb_dim)  # -> [B, 256, F/4, T/4]
        self.down3 = DownBlock(n_features * 4, n_features * 8, time_emb_dim)  # -> [B, 512, F/8, T/8]
        self.down4 = DownBlock(n_features * 8, n_features * 8, time_emb_dim)  # -> [B, 512, F/16, T/16]
        self.bottleneck = ConvBlock(n_features * 8, n_features * 16, time_emb_dim)  # -> [B, 1024, F/16, T/16]

        # Decoder (in_channels, skip_channels, out_channels)
        self.up1 = UpBlock(n_features * 16, n_features * 8, n_features * 4, time_emb_dim)  # 1024 -> 256, skip 512
        self.up2 = UpBlock(n_features * 4, n_features * 4, n_features * 2, time_emb_dim)   # 256 -> 128, skip 256
        self.up3 = UpBlock(n_features * 2, n_features * 2, n_features, time_emb_dim)       # 128 -> 64,  skip 128
        self.up4 = UpBlock(n_features, n_features, n_features, time_emb_dim)               # 64  -> 64,  skip 64
        self.outc = nn.Conv2d(n_features, out_channels, kernel_size=1)                     # [B, 64, F, T] -> [B, 2, F, T]

    def forward(self, x, t):
        time_emb = self.time_mlp(t)  # [B, time_emb_dim]

        x1 = self.inc(x, time_emb)
        x2 = self.down1(x1, time_emb)
        x3 = self.down2(x2, time_emb)
        x4 = self.down3(x3, time_emb)
        x5 = self.down4(x4, time_emb)

        x_bottleneck = self.bottleneck(x5, time_emb)

        x = self.up1(x_bottleneck, x4, time_emb)
        x = self.up2(x, x3, time_emb)
        x = self.up3(x, x2, time_emb)
        x = self.up4(x, x1, time_emb)
        output = self.outc(x)
        return output
    

