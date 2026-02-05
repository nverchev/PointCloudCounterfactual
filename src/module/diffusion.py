"""Diffusion Model."""

import math

import torch
import torch.nn as nn

from src.config.options import DiffusionSchedulers
from src.config import Experiment
from src.data.structures import Inputs, Outputs
from src.module.diffusion_networks import get_diffusion_network


class DiffusionModel(nn.Module):
    """Diffusion Probabilistic Model for Point Clouds."""

    alphas: list[torch.Tensor]
    betas: list[torch.Tensor]
    alphas_cum_prod: list[torch.Tensor]
    sqrt_alphas_cum_prod: list[torch.Tensor]
    sqrt_one_minus_alphas_cum_prod: list[torch.Tensor]

    def __init__(self):
        super().__init__()
        cfg_diff = Experiment.get_config().diffusion
        self.network = get_diffusion_network()
        self.n_timesteps: int = cfg_diff.model.n_timesteps
        self.beta_start: float = cfg_diff.model.beta_start
        self.beta_end: float = cfg_diff.model.beta_end
        self.schedule_type: str = cfg_diff.model.schedule_type
        betas = self._get_beta_schedule()
        alphas = 1.0 - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cum_prod = torch.sqrt(alphas_cum_prod)
        sqrt_one_minus_alphas_cum_prod = torch.sqrt(1.0 - alphas_cum_prod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cum_prod', alphas_cum_prod)
        self.register_buffer('sqrt_alphas_cum_prod', sqrt_alphas_cum_prod)
        self.register_buffer('sqrt_one_minus_alphas_cum_prod', sqrt_one_minus_alphas_cum_prod)
        self.n_training_points_training: int = cfg_diff.n_training_output_points
        self.n_inference_output_points: int = cfg_diff.objective.n_inference_output_points
        return

    @property
    def n_output_points(self) -> int:
        """Number of generated points."""
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    def _get_beta_schedule(self):
        if self.schedule_type == DiffusionSchedulers.Linear:
            return torch.linspace(self.beta_start, self.beta_end, self.n_timesteps)

        elif self.schedule_type == DiffusionSchedulers.Cosine:
            s = 0.008
            steps = self.n_timesteps + 1
            x = torch.linspace(0, self.n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        raise ValueError(f'Unknown schedule type: {self.schedule_type}')

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass for training (epsilon prediction)."""
        x_0 = inputs.cloud  # [B, N, 3]
        device = x_0.device
        batch_size = x_0.shape[0]

        # Sample random timesteps (or fix for debugging)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device).long()

        # Sample noise
        epsilon = torch.randn_like(x_0)

        # Noise schedule terms
        sqrt_alpha_bar_t = self.sqrt_alphas_cum_prod[t].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cum_prod[t].view(batch_size, 1, 1)

        # Forward diffusion
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon

        # Normalized timestep
        t_normalized = t.float() / self.n_timesteps

        # Network prediction
        pred_epsilon = self.network(x_t, t_normalized, self.n_output_points)

        out = Outputs()
        out.epsilon = epsilon
        out.pred_epsilon = pred_epsilon
        return out

    @torch.no_grad()
    def sample(self, n_samples: int, n_points: int, device: torch.device) -> torch.Tensor:
        """Sample using DDPM (epsilon prediction)."""
        x = torch.randn(n_samples, n_points, 3, device=device)

        for i in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            t_normalized = t.float() / self.n_timesteps

            pred_epsilon = self.network(x, t_normalized, n_points)

            alpha = self.alphas[i]
            alpha_cum_prod = self.alphas_cum_prod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cum_prod)) * pred_epsilon
            ) + torch.sqrt(beta) * noise

        return x


def get_diffusion_module() -> DiffusionModel:
    """Get the diffusion model according to the configuration."""
    return DiffusionModel()
