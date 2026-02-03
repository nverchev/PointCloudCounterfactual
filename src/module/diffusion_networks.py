"""Diffusion networks."""

import math

import torch
import torch.nn as nn

from src.config import Experiment, ActClass
from src.module.decoders import PCGen
from src.module.layers import LinearLayer


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t: Timestep tensor of shape (B,) or (B, 1).
        """
        if t.dim() == 2:
            t = t.squeeze(1)

        device = t.device
        half_dim = self.dim // 2

        # Standard implementation: 10000 ^ (2 * i / dim)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Handle odd dimensions (though typically we use power of 2)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)

        return embeddings


class PCGenDiffusion(nn.Module):
    """PointNet-like diffusion network."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder.model
        self.w_dim: int = cfg_ae.w_dim
        self.embedding_dim: int = cfg_ae.embedding_dim
        self.pcgen = PCGen()
        self.act_cls: ActClass = self.pcgen.act_cls

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(self.embedding_dim),
            LinearLayer(self.embedding_dim, 1024, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(1024, self.w_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, n_output_points: int) -> torch.Tensor:
        """Forward pass."""
        w = self.time_embed(t)
        out = self.pcgen(w, n_output_points, x.transpose(2, 1)).transpose(2, 1)
        return out.contiguous()


def get_diffusion_network() -> nn.Module:
    """Get diffusion network according to the configuration."""
    return PCGenDiffusion()
