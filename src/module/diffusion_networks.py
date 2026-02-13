"""Diffusion networks."""

import math

import torch
import torch.nn as nn

from src.config import Experiment, ActClass
from src.module.decoders import PCGen


class SinusoidalPositionalEmbedding(nn.Module):
    freqs: torch.Tensor

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim: int = dim

        half_dim = dim // 2
        # Precompute and register as buffer
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim).float() / half_dim)
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        if t.dim() > 1:
            t = t.view(-1)

        # [B, 1] * [1, half_dim] -> [B, half_dim]
        args = t[:, None] * self.freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, act_cls):
        super().__init__()
        # Time projection to get scale and shift parameters
        self.time_mlp = nn.Linear(time_emb_dim, dim * 2)

        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.act = act_cls()

    def forward(self, x, t_emb):
        # x: [B, C, N], t_emb: [B, time_emb_dim]
        residual = x

        # 1. Project time to scale and shift
        time_params = self.time_mlp(t_emb).unsqueeze(-1)  # [B, 2*C, 1]
        scale, shift = torch.chunk(time_params, 2, dim=1)

        # 2. Modulate features
        x = self.conv1(x)
        x = x * (1 + scale) + shift  # The modulation step
        x = self.act(x)

        x = self.conv2(x)
        return x + residual  # Skip connection


class MLPDiffusionWithGlobal(nn.Module):
    def __init__(self, hidden_dim=512, time_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Improved Sinusoidal Time Embedding
        self.time_embed_raw = SinusoidalPositionalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global Feature Extractor (PointNet)
        self.input_proj = nn.Conv1d(3, hidden_dim, 1)
        self.global_extractor = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, nn.SiLU), ResidualBlock(hidden_dim, hidden_dim, nn.SiLU)
        )

        # Refinement blocks
        self.refine = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, nn.SiLU), ResidualBlock(hidden_dim, hidden_dim, nn.SiLU)
        )

        self.final_proj = nn.Conv1d(hidden_dim, 3, 1)

    def forward(self, x, t, n_output_points=None):
        # x: [B, N, 3] -> [B, 3, N]
        x = x.transpose(1, 2)

        # 1. Time Embedding
        t_raw = self.time_embed_raw(t)
        t_emb = self.time_mlp(t_raw)  # [B, hidden_dim]

        # 2. Process features with time modulation
        feat = self.input_proj(x)

        # Extract global context
        feat = self.global_extractor[0](feat, t_emb)
        global_pool = feat.max(dim=-1, keepdim=True)[0]  # [B, hidden_dim, 1]

        # Add global context back to local features
        feat = feat + global_pool

        # Refine with more modulated blocks
        for block in self.refine:
            feat = block(feat, t_emb)

        out = self.final_proj(feat)
        return out.transpose(1, 2)  # [B, N, 3]


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
        self.time_embed = SinusoidalPositionalEmbedding(self.w_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, n_output_points: int) -> torch.Tensor:
        """Forward pass."""
        w = self.time_embed(t)
        out = self.pcgen(w, n_output_points, x.transpose(2, 1)).transpose(2, 1)
        return out.contiguous()


def get_diffusion_network() -> nn.Module:
    """Get diffusion network according to the configuration."""
    return PCGenDiffusion()
