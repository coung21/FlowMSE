import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # down x2
        )
    def forward(self, x): return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class SpeechEnhancementConvAE(nn.Module):
    """
    Autoencoder tích chập cho phổ STFT 2 kênh (real/imag).
    Input:  [B, 2, F, T]
    Output: [B, 2, F, T]
    Lưu ý: F,T nên chia hết cho 16 (4 lần down/up x2).
    """
    def __init__(self, bottleneck_channels=256, use_groupnorm=False):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(8, c)) if use_groupnorm else nn.BatchNorm2d

        # Encoder: 2 -> 32 -> 64 -> 128 -> 256, mỗi lần /2 kích thước
        self.enc1 = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1),  Norm(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), Norm(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),Norm(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.enc4 = nn.Sequential(nn.Conv2d(128, bottleneck_channels, 3, padding=1), Norm(bottleneck_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        # (tuỳ chọn) thêm conv ở đáy làm "bottleneck" mỏng hơn
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            Norm(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder: upsample đối xứng
        self.dec1 = DecoderBlock(bottleneck_channels, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec3 = DecoderBlock(64, 32)
        # lớp cuối: ra 2 kênh, để linear nếu dùng MSE
        self.dec4 = nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2)

    def forward(self, x):
        # đảm bảo kích thước phù hợp
        assert x.shape[-2] % 16 == 0 and x.shape[-1] % 16 == 0, "F,T phải chia hết cho 16."
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        h  = self.bottleneck(h4)
        y  = self.dec1(h)
        y  = self.dec2(y)
        y  = self.dec3(y)
        y  = self.dec4(y)   # [B,2,F,T]
        return y
