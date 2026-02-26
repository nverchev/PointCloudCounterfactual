"""Flow Matching Model."""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from src.config import Experiment
from src.data.structures import Inputs, Outputs
from src.module.decoders import get_decoder
from src.module.time import TimeEmbedding
from src.config.options import FlowModels
from src.utils.neighbour_ops import torch_square_distance


class FlowMatchingModel(nn.Module):
    """Flow Matching for Point Clouds."""

    def __init__(self):
        super().__init__()
        cfg_flow = Experiment.get_config().flow
        self.n_timesteps: int = cfg_flow.model.n_timesteps
        self.n_training_points_training: int = cfg_flow.objective.n_training_output_points
        self.n_inference_output_points: int = cfg_flow.objective.n_inference_output_points
        self.assign_noise: bool = cfg_flow.model.assign_noise
        self.decoder = get_decoder()
        self.time_embedding = TimeEmbedding(feature_dim=self.decoder.feature_dim)
        return

    @property
    def n_output_points(self) -> int:
        """Get the number of output points."""
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass for training velocity prediction."""
        x_1 = inputs.cloud
        noise = torch.randn_like(x_1)
        out = Outputs()
        device = x_1.device
        batch_size = x_1.shape[0]
        x_0 = self._assign_noise(x_1, noise) if self.assign_noise else noise
        t = self._sample_time(batch_size, device=device)
        x_t = (1.0 - t) * x_0 + t * x_1
        features = self.time_embedding(t)
        v_pred = self.decoder(x_t.transpose(1, 2), features, self.n_output_points)
        out.v_pred = v_pred
        out.v_target = x_1 - x_0
        return out

    @torch.no_grad()
    def sample(self, n_samples: int, n_points: int, device: torch.device) -> list[torch.Tensor]:
        """Deterministic ODE Sampling (Euler Integration)."""
        x_t = torch.randn(n_samples, n_points, 3, device=device)
        x_list = [x_t]

        # Flow from t=1.0 (noise) to t=0.0 (data)
        for i in reversed(range(1, self.n_timesteps + 1)):
            # Calculate tau (uniform steps)
            tau_curr = i / self.n_timesteps
            tau_prev = (i - 1) / self.n_timesteps

            # Warp tau into t-space
            t_curr = tau_curr
            t_prev = tau_prev

            dt = t_curr - t_prev
            t = torch.full((n_samples,), t_curr, device=device).view(-1, 1, 1)
            features = self.time_embedding(t)
            v_pred = self.decoder(x_t.transpose(1, 2), features, n_points)
            x_t = x_t - v_pred * dt
            x_list.append(x_t)

        return x_list

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time uniformly from [0, 1]."""
        return torch.rand(batch_size, device=device).view(-1, 1, 1)

    def _assign_noise(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Assign noise to x_0."""
        batch_size, n_points = x_0.shape[:2]
        device = x_0.device
        indices = get_bijective_assignment(x_0, x_1)
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(batch_size, n_points)
        x0_aligned = x_0[batch_idx, indices]
        return x0_aligned


@torch.no_grad()
def get_bijective_assignment(x, y):
    """Get bijective assignment between two point clouds."""
    B, _, _ = x.shape
    cost = torch_square_distance(x, y).cpu().numpy()
    batch_idx = torch.zeros((B, x.shape[1]), dtype=torch.long, device=x.device)
    for b in range(B):
        _, idx_b = linear_sum_assignment(cost[b])
        batch_idx[b] = torch.from_numpy(idx_b).to(x.device)

    return batch_idx


def get_flow_module() -> FlowMatchingModel:
    """Get the correct flow matching module according to the configuration."""
    model_registry = {
        FlowModels.FlowMatchingModel: FlowMatchingModel,
    }
    return model_registry[Experiment.get_config().flow.model.class_name]()
