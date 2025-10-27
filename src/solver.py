import torch
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
def euler_solver(model: torch.nn.Module, x: torch.Tensor, num_steps: int, t_start: float = 0.0, t_end: float = 1.0) -> torch.Tensor:
    """
    Perform Euler method to solve ODE defined by the model.

    Args:
        model (torch.nn.Module): The neural network model defining the ODE.
        x (torch.Tensor): Initial state tensor of shape [B, 2, F, T].
        t_start (float): Starting time.
        t_end (float): Ending time.
        num_steps (int): Number of steps to take from t_start to t_end.

    Returns:
        torch.Tensor: The state tensor at time t_end.
    """

    time_steps = torch.linspace(t_start, t_end, num_steps + 1, device=x.device)
    dt = (t_end - t_start) / num_steps
    
    x_t = x.clone()

    for i in tqdm(range(num_steps), desc="Euler ODE Solver"):
        
        t_current = time_steps[i]

        t_tensor = torch.full((x.shape[0],), t_current, device=x.device)

        v_t = model(x_t, t_tensor)  # [B, 2, F, T]
        x_t = x_t + v_t * dt

    return x_t