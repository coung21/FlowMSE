import torch
import torch.nn as nn
import torch.nn.functional as F


###########################
# Complex tensor utilities #
###########################
def complex_to_ri(X):
    """complex (B,F,T) -> (B,2,F,T)"""
    return torch.view_as_real(X).permute(0, 3, 1, 2).contiguous()


def ri_to_complex(X_ri):
    """(B,2,F,T) -> complex (B,F,T)"""
    return torch.view_as_complex(X_ri.permute(0, 2, 3, 1).contiguous())


###########################
# Convolutional VAE block #
###########################
def conv_block(in_ch, out_ch, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def deconv_block(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, k, s, p),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvVAE(nn.Module):
    def __init__(self, z_dim=128, in_ch=2):
        super().__init__()
        # --- Encoder ---
        self.enc = nn.Sequential(
            conv_block(in_ch, 32),      # /2
            conv_block(32, 64),         # /4
            conv_block(64, 128),        # /8
            conv_block(128, 256),       # /16
        )
        self.enc_out_shape = None
        self.z_dim = z_dim

        # Linear layers for latent space (set later after first forward)
        self.mu = None
        self.logvar = None
        self.fc_dec = None

        # --- Decoder (mirror) ---
        self.dec = nn.Sequential(
            deconv_block(256, 128),
            deconv_block(128, 64),
            deconv_block(64, 32),
            nn.ConvTranspose2d(32, 2, 4, 2, 1),  # output (B,2,F,T)
        )
        self.out_act = nn.Identity()  # could use nn.Tanh() for mask output

    def _build_latent_layers(self, shape, device):
        flat_dim = torch.prod(torch.tensor(shape)).item()
        self.mu = nn.Linear(flat_dim, self.z_dim).to(device)
        self.logvar = nn.Linear(flat_dim, self.z_dim).to(device)
        self.fc_dec = nn.Linear(self.z_dim, flat_dim).to(device)
        self.enc_out_shape = shape

    def encode(self, x):
        h = self.enc(x)
        if self.enc_out_shape is None:
            self._build_latent_layers(h.shape[1:], x.device)
        h_flat = h.flatten(1)
        mu = self.mu(h_flat)
        logvar = self.logvar(h_flat)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, *self.enc_out_shape)
        y = self.dec(h)
        return self.out_act(y)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar


#####################
# Top-level wrapper #
#####################
class SpeechEnhancementVAE(nn.Module):
    """
    Wrapper model: nhận STFT phức của noisy speech
    → xuất STFT phức của clean speech.
    """

    def __init__(self, z_dim=128):
        super().__init__()
        self.vae = ConvVAE(z_dim=z_dim, in_ch=2)

    def forward(self, noisy_complex):
        """
        noisy_complex: complex tensor (B, F, T)
        Trả về: (reconstructed complex STFT, mu, logvar)
        """
        X_ri = complex_to_ri(noisy_complex)  # (B,2,F,T)
        Y_ri, mu, logvar = self.vae(X_ri)
        Y_complex = ri_to_complex(Y_ri)
        return Y_complex, mu, logvar


#####################
# Loss definitions  #
#####################
def complex_mse(pred, target):
    """Complex MSE loss"""
    diff = pred - target
    return (diff.real.pow(2) + diff.imag.pow(2)).mean()


def kl_loss(mu, logvar):
    """KL divergence for VAE"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class VAEComplexLoss(nn.Module):
    """Kết hợp complex MSE và KL"""

    def __init__(self, beta=1e-4):
        super().__init__()
        self.beta = beta

    def forward(self, pred_complex, target_complex, mu, logvar):
        rec = complex_mse(pred_complex, target_complex)
        kl = kl_loss(mu, logvar)
        loss = rec + self.beta * kl
        return loss, rec, kl
