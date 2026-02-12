"""Flow Matching Model."""

from typing import Any

import geomloss
import torch
import torch.nn as nn
from pykeops.torch import LazyTensor

from src.data.structures import Inputs, Outputs
from src.module.diffusion_networks import get_diffusion_network, Encoder, MLPDDiscriminator
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
        self.encoder = Encoder()
        self.discriminator = MLPDDiscriminator()

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
        indices = get_hard_assignment(noise, x_0)
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


def get_hard_assignment(x, y):
    """
    Returns indices 'idx' such that y[idx] is the OT-match for x.
    """
    # 1. Use geomloss to get dual potentials (Sinkhorn)
    # p=2 is Wasserstein-2 (L2 distance)
    L = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.001, potentials=True)
    _, g = L(x, y)  # f and g are the dual potentials

    # 2. Reconstruct the 'cost-adjusted' distance
    # The optimal match for x_i is the y_j that minimizes C_ij - g_j
    # C_ij is |x_i - y_j|^2
    x_i = LazyTensor(x.unsqueeze(2))  # [B, N, 1, 3]
    y_j = LazyTensor(y.unsqueeze(1))  # [B, 1, M, 3]
    g_j = LazyTensor(g.unsqueeze(1).unsqueeze(-1))  # [B, 1, M, 1]

    # 3. Reconstruct the cost-adjusted distance: |x_i - y_j|^2 - g_j
    # Note: pykeops_square_distance is ((x_i - y_j)**2).sum(-1)
    C_minus_g = ((x_i - y_j) ** 2).sum(-1) - g_j

    # 4. Perform the reduction (argmin) on the LazyTensor
    # dim=2 corresponds to the M dimension (y points)
    idx = C_minus_g.argmin(dim=2).view(x.shape[0], x.shape[1])

    return idx


def get_diffusion_module() -> DiffusionModel:
    return DiffusionModel()
