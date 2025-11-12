# src/model.py (thêm vào file hiện tại)
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Blocks ----------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, norm='bn'):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm == 'bn' else (lambda c: nn.Identity())
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, norm='bn'):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm == 'bn' else (lambda c: nn.Identity())
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch, norm='bn'):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm == 'bn' else (lambda c: nn.Identity())
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = norm_layer(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = norm_layer(ch)
    def forward(self, x):
        h = F.relu(self.n1(self.conv1(x)), inplace=True)
        h = self.n2(self.conv2(h))
        return F.relu(x + h, inplace=True)

# --------- Generator (conditional) ----------
class GeneratorConv(nn.Module):
    """
    Conditional Generator cho khử nhiễu phổ STFT 2 kênh (real/imag).
    Input:  [B, 2, F, T]
    Output: [B, 2, F, T]
    - Sinh residual r, rồi y = x + r (có thể tắt nếu muốn).
    - Yêu cầu F,T chia hết cho 16 (4 lần down/up x2).
    """
    def __init__(self, base=32, residual_out=True, norm='bn'):
        super().__init__()
        self.residual_out = residual_out

        # Encoder
        self.e1 = Down(2,    base,   norm=norm)   # F,T -> /2
        self.e2 = Down(base, base*2, norm=norm)   # -> /4
        self.e3 = Down(base*2, base*4, norm=norm) # -> /8
        self.e4 = Down(base*4, base*8, norm=norm) # -> /16

        # Bottleneck
        self.b1 = ResBlock(base*8, norm=norm)
        self.b2 = ResBlock(base*8, norm=norm)

        # Decoder (skip concat)
        self.u1 = Up(base*8, base*4, norm=norm)
        self.u2 = Up(base*8, base*2, norm=norm)  # concat với e3 => ch*2
        self.u3 = Up(base*4, base,   norm=norm)  # concat với e2
        self.u4 = Up(base*2, base,   norm=norm)  # concat với e1

        # Head
        self.head = nn.Conv2d(base, 2, kernel_size=3, padding=1)  # linear

    def forward(self, x):
        assert x.shape[-2] % 16 == 0 and x.shape[-1] % 16 == 0, "F,T phải chia hết cho 16."
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        h = self.b1(e4)
        h = self.b2(h)

        h = self.u1(h)
        h = torch.cat([h, e3], dim=1)
        h = self.u2(h)
        h = torch.cat([h, e2], dim=1)
        h = self.u3(h)
        h = torch.cat([h, e1], dim=1)
        h = self.u4(h)

        r = self.head(h)
        return x + r if self.residual_out else r

# --------- Patch Discriminator (conditional) ----------
class DiscriminatorConvPatch(nn.Module):
    """
    PatchGAN: nhập [B, 4, F, T] (noisy concat target/enhanced theo kênh).
    Trả bản đồ logits [B, 1, F/16, T/16] (tuỳ strides).
    """
    def __init__(self, base=32, norm='bn'):
        super().__init__()
        norm_layer = nn.BatchNorm2d if norm == 'bn' else (lambda c: nn.Identity())
        ch = base
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),      # /2
            nn.Conv2d(ch, ch*2, 4, 2, 1), norm_layer(ch*2), nn.LeakyReLU(0.2, inplace=True),  # /4
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), norm_layer(ch*4), nn.LeakyReLU(0.2, inplace=True),# /8
            nn.Conv2d(ch*4, ch*8, 4, 2, 1), norm_layer(ch*8), nn.LeakyReLU(0.2, inplace=True),# /16
            nn.Conv2d(ch*8, 1, 3, 1, 1)  # logits per patch
        )

    def forward(self, noisy_ri, ref_ri):
        """
        noisy_ri: [B,2,F,T]
        ref_ri:   [B,2,F,T] (clean hoặc enhanced)
        """
        x = torch.cat([noisy_ri, ref_ri], dim=1)
        return self.net(x)  # [B,1,*,*]



def d_hinge_loss(real_logits, fake_logits):
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake

def g_hinge_loss(fake_logits):
    return -fake_logits.mean()

def l1_recon_loss(pred, target):
    return F.l1_loss(pred, target, reduction='mean')
