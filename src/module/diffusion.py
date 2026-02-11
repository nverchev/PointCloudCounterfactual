"""Flow Matching Model."""

import torch
import torch.nn as nn
import torch.distributions.beta as beta

from src.data.structures import Inputs, Outputs
from src.module.diffusion_networks import get_diffusion_network
from src.config import Experiment


class DiffusionModel(nn.Module):
    """Flow Matching for Point Clouds (Rectified Flow)."""

    def __init__(self):
        super().__init__()
        cfg_diff = Experiment.get_config().diffusion
        self.network = get_diffusion_network()
        self.n_timesteps: int = cfg_diff.model.n_timesteps
        self.n_training_points_training: int = cfg_diff.n_training_output_points
        self.n_inference_output_points: int = cfg_diff.objective.n_inference_output_points
        self.sampling_dist = beta.Beta(1, 5)

    @property
    def n_output_points(self) -> int:
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass for training velocity prediction."""
        x_0 = inputs.cloud  # [B, N, 3] (Clean Data)
        x_0.div_(x_0.std(dim=(1, 2), keepdim=True))

        device = x_0.device
        batch_size = x_0.shape[0]

        # 1. Sample t in [0, 1] and reshape for broadcasting [B, 1, 1]
        dt = 1.0 / self.n_timesteps
        t = dt + self.sampling_dist.sample((batch_size,)).to(device)
        t_view = t.view(-1, 1, 1)

        # 2. Sample Gaussian Noise (Source distribution)
        x_1 = torch.randn_like(x_0)

        # 3. Linear Interpolation (The Probability Path)
        # x_t = (1 - t)*x_0 + t*x_1
        x_t = (1.0 - t_view) * x_0 + t_view * x_1

        # 4. Network Prediction
        # Passes t (the scalar per batch) to the network
        x_1 - x_0
        v_pred = self.network(x_t, t, self.n_output_points)

        out = Outputs()
        # Where the points move according to the model
        out.v_pred = v_pred
        # Where the points move according to the ground truth flow
        out.v = x_1 - x_0

        return out

    @torch.no_grad()
    def sample(self, n_samples: int, n_points: int, device: torch.device) -> list[torch.Tensor]:
        """Deterministic ODE Sampling (Euler Integration)."""
        # Start from pure noise (t=1.0)
        x_t = torch.randn(n_samples, n_points, 3, device=device)
        x_list = [x_t]

        # Integration step size
        dt = 1.0 / self.n_timesteps

        # Flow from t=1.0 (noise) to t=0.0 (data)
        for i in reversed(range(self.n_timesteps)):
            t_val = i / self.n_timesteps
            t = torch.full((n_samples,), t_val, device=device)

            # Predict velocity at current position
            v_pred = self.network(x_t, t, n_points)

            # Euler step: x_{t-dt} = x_t - v * dt
            # Since we go from 1 to 0, we subtract the velocity
            x_t = x_t - v_pred * dt
            x_list.append(x_t)

        for x in x_list:
            x.subtract_(x.mean(dim=1, keepdim=True))
            x.div_(torch.max(x.norm(dim=2, keepdim=True)))

        return x_list


def get_diffusion_module() -> DiffusionModel:
    return DiffusionModel()
