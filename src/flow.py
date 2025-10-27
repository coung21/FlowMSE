import torch
import torch.nn as nn
import torch.nn.functional as F


def broadcast_t(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:

    assert t.dim() == 1, "Input tensor must be 1-dimensional" # [B]

    new_shape = (t.shape[0],) + (1,) * (len(shape) - 1)  # [B, 1, 1]

    return t.view(new_shape)  # [B, 1, 1]


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Performs linear interpolation between x0 and x1 using t.

    Args:
        x0 (torch.Tensor): The starting tensor, shape [B, F, T].
        x1 (torch.Tensor): The ending tensor, shape [B, F, T].
        t (torch.Tensor): Time steps for interpolation, shape [B].

    Returns:
        torch.Tensor: The interpolated tensor.
    """
    t = broadcast_t(t, x0.shape)  # [B, 1, 1]
    # t real * x complex
    return (1 - t) * x0 + t * x1 # [B, F, T]

def target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Computes the target velocity u_t.

    Args:
        x0 (torch.Tensor): The starting tensor, shape [B, F, T].
        x1 (torch.Tensor): The ending tensor, shape [B, F, T].

    Returns:
        torch.Tensor: The target velocity tensor, shape [B, F, T].
    """

    return x1 - x0  # [B, F, T]


class FlowMatchingLoss(nn.Module):
    def __init__(self, sigma_min: float =1e-4):
        super(FlowMatchingLoss, self).__init__()
        self.sigma_min = sigma_min

    def forward(self, model: nn.Module, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Computes the flow matching loss.

        Args:
            model (nn.Module): The neural network model that predicts velocity.
            x0 (torch.Tensor): The starting tensor, shape [B, F, T].
            x1 (torch.Tensor): The ending tensor, shape [B, F, T].

        Returns:
            torch.Tensor: The computed flow matching loss.
        """

        B = x0.shape[0]
        device = x0.device

        # sample t uniformly from (0, 1)
        t = torch.rand(B, device=device) * (1.0 - self.sigma_min) + self.sigma_min  # [B]
        
        x_t = linear_interpolation(x0, x1, t)  # [B, F, T]

        u_t = target_velocity(x0, x1)  # [B, F, T]

        x_t = torch.stack([x_t.real, x_t.imag], dim=1)  # [B, 2, F, T]

        u_t = torch.stack([u_t.real, u_t.imag], dim=1)  # [B, 2, F, T]
        v_pred = model(x_t, t)  # [B, 2, F, T]

        loss = F.mse_loss(v_pred, u_t)

        return loss 