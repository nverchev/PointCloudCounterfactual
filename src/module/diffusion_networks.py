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


class PointNetDiffusionWithGlobal(nn.Module):
    """PointNet-style diffusion with global context (more powerful baseline)."""

    def __init__(self):
        super().__init__()
        Experiment.get_config()

        self.time_dim = 128
        self.hidden_dim = 256

        self.act_cls = nn.ReLU

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(self.time_dim),
            LinearLayer(self.time_dim, self.hidden_dim, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(self.hidden_dim, self.hidden_dim),
        )

        # Point-wise encoder
        self.point_encoder = nn.Sequential(
            LinearLayer(3, 64, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(64, 128, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(128, self.hidden_dim),
        )

        # Global feature aggregation
        self.global_pool = nn.Sequential(
            LinearLayer(self.hidden_dim, self.hidden_dim, act_cls=self.act_cls),
            self.act_cls(),
        )

        # Combine local, global, and time features
        self.feature_combine = nn.Sequential(
            LinearLayer(self.hidden_dim * 3, self.hidden_dim, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(self.hidden_dim, self.hidden_dim, act_cls=self.act_cls),
            self.act_cls(),
        )

        # Output head
        self.output_head = nn.Sequential(
            LinearLayer(self.hidden_dim, 128, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(128, 64, act_cls=self.act_cls),
            self.act_cls(),
            LinearLayer(64, 3),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, n_output_points: int) -> torch.Tensor:
        """Forward pass with global context.

        Args:
            x: Noisy point cloud [B, N, 3]
            t: Timesteps [B]
            n_output_points: Number of output points

        Returns:
            Predicted noise/velocity [B, N, 3]
        """
        _batch_size, n_points, _ = x.shape

        # Time embedding [B, hidden_dim]
        t_emb = self.time_embed(t)

        # Local point features [B, N, hidden_dim]
        point_features = self.point_encoder(x)

        # Global context via max pooling [B, hidden_dim]
        global_features = self.global_pool(point_features)
        global_features = torch.max(global_features, dim=1)[0]

        # Broadcast time and global features [B, N, hidden_dim]
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, n_points, -1)
        global_expanded = global_features.unsqueeze(1).expand(-1, n_points, -1)

        # Concatenate [B, N, hidden_dim * 3]
        combined = torch.cat([point_features, global_expanded, t_emb_expanded], dim=-1)

        # Process [B, N, hidden_dim]
        features = self.feature_combine(combined)

        # Output [B, N, 3]
        output = self.output_head(features)

        return output


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
    return PointNetDiffusionWithGlobal()
