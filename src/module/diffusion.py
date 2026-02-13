"""Flow Matching Model."""

from typing import Any

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.data.structures import Inputs, Outputs
from src.module.diffusion_networks import get_diffusion_network
from src.config import Experiment


def frozen_forward(network: nn.Module, x: torch.Tensor) -> Any:
    """Temporarily disable gradients for all parameters."""
    was_requires_grad: list[bool] = []
    for p in network.parameters():
        was_requires_grad.append(p.requires_grad)
        p.requires_grad_(False)

    # Run the network
    output = network(x)

    # Restore original requires_grad flags
    for p, req in zip(network.parameters(), was_requires_grad, strict=True):
        p.requires_grad_(req)

    return output


class DiffusionModel(nn.Module):
    """Flow Matching for Point Clouds (Rectified Flow)."""

    def __init__(self):
        super().__init__()
        cfg_diff = Experiment.get_config().diffusion
        self.network = get_diffusion_network()
        self.n_timesteps: int = cfg_diff.model.n_timesteps
        self.n_training_points_training: int = cfg_diff.n_training_output_points
        self.n_inference_output_points: int = cfg_diff.objective.n_inference_output_points
        return

    @property
    def n_output_points(self) -> int:
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass for training velocity prediction."""
        x_0 = inputs.cloud  # [B, N, 3] (Clean Data)
        with torch.no_grad():
            x_0.div_(x_0.std(dim=(1, 2), keepdim=True))

        noise = torch.randn_like(x_0)
        out = Outputs()

        device = x_0.device
        batch_size, n_points = x_0.shape[:2]
        indices = get_bijective_assignment(noise, x_0)
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(batch_size, n_points)
        x0_aligned = x_0[batch_idx, indices]

        # 1. Sample t in [0, 1] and reshape for broadcasting [B, 1, 1]
        t = torch.rand(batch_size, device=device)
        t_view = t.view(-1, 1, 1)

        # 3. Linear Interpolation (The Probability Path)
        # x_t = (1 - t)*x_0 + t*x_1
        x_t = (1.0 - t_view) * x0_aligned + t_view * noise

        # 4. Network Prediction
        # Passes t (the scalar per batch) to the network
        v_pred = self.network(x_t, t, self.n_output_points)

        # Where the points move according to the model
        out.v_pred = v_pred
        # Where the points move according to the ground truth flow
        out.v = noise - x0_aligned
        return out

    @torch.no_grad()
    def sample(self, n_samples: int, n_points: int, device: torch.device) -> list[torch.Tensor]:
        """Deterministic ODE Sampling (Euler Integration)."""
        # Start from pure noise (t=1.0)
        x_t = torch.randn(n_samples, n_points, 3, device=device)
        # x_t = self.x_T

        x_list = [x_t]

        # Flow from t=1.0 (noise) to t=0.0 (data)
        for i in reversed(range(1, self.n_timesteps + 1)):
            # Calculate tau (uniform steps)
            tau_curr = i / self.n_timesteps
            tau_prev = (i - 1) / self.n_timesteps

            # Warp tau into t-space
            t_curr = tau_curr
            t_prev = tau_prev

            # The specific dt for this step (it will get smaller as t -> 0)
            dt = t_curr - t_prev
            t = torch.full((n_samples,), t_curr, device=device)

            # Predict velocity at current position
            v_pred = self.network(x_t, t, n_points)

            # Euler step: x_{t-dt} = x_t - v * dt
            # Since we go from 1 to 0, we subtract the velocity
            x_t = x_t - v_pred * dt
            x_list.append(x_t)

        for x in x_list:
            x.subtract_(x.mean(dim=1, keepdim=True))
            norms = x.norm(dim=2, keepdim=True)
            max_norms = norms.max(dim=1, keepdim=True)[0]
            x.div_(max_norms + 1e-8)

        return x_list


@torch.no_grad()
def get_bijective_assignment(x, y, group_size=256):
    B, N, _ = x.shape
    device = x.device

    # 1. Spatial Sorting (e.g., sort by X-coordinate or sum of coordinates)
    # This ensures groups are spatially localized
    perm_x = x.sum(dim=-1).argsort(dim=1)
    perm_y = y.sum(dim=-1).argsort(dim=1)

    # Use batch_idx to handle the sorting across the batch dimension
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    x_shuffled = x[batch_idx, perm_x]
    y_shuffled = y[batch_idx, perm_y]

    num_groups = N // group_size
    actual_y_indices_aligned = torch.zeros((B, N), dtype=torch.long, device=device)

    for g in range(num_groups):
        start, end = g * group_size, (g + 1) * group_size

        x_g = x_shuffled[:, start:end, :].contiguous()
        y_g = y_shuffled[:, start:end, :].contiguous()

        # Use simple LSA for speed (or Sinkhorn if you prefer)
        dist = torch.cdist(x_g, y_g) ** 2
        cost_np = dist.detach().cpu().numpy()

        for b in range(B):
            _, col_ind = linear_sum_assignment(cost_np[b])
            # Map local to global using the specific batch's perm_y
            actual_y_indices_aligned[b, start:end] = perm_y[b, start:end][torch.from_numpy(col_ind).to(device)]

    # 3. Unshuffle
    inv_perm_x = torch.argsort(perm_x, dim=1)
    return actual_y_indices_aligned.gather(1, inv_perm_x)


def get_diffusion_module() -> DiffusionModel:
    return DiffusionModel()
