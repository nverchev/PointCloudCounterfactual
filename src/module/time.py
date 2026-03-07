"""Module for time embeddings and vector modulation."""

import torch
import torch.nn as nn
from src.module.layers import LinearLayer, SinusoidalPositionalEmbedding
from src.config.specs import TimeEmbeddingConfig


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding module for Flow Matching."""

    def __init__(self, cfg: TimeEmbeddingConfig, feature_dim: int):
        super().__init__()
        embedding_dim = cfg.embedding_dim
        mlp_dims = cfg.mlp_dims

        self.time_embedding = SinusoidalPositionalEmbedding(dim=embedding_dim)

        layers = []
        curr_dim = embedding_dim
        for next_dim in mlp_dims:
            layers.append(LinearLayer(curr_dim, next_dim, act_cls=nn.GELU))
            curr_dim = next_dim

        layers.append(LinearLayer(curr_dim, 2 * feature_dim, use_trunc_init=True))
        self.time_mlp = nn.Sequential(*layers)

        self.feature_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)

    def forward(self, t: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Embed time and modulate features."""
        t_embed = self.time_embedding(t)
        modulation = self.time_mlp(t_embed)
        mean, scale = modulation.chunk(2, dim=1)
        features = self.feature_norm(features)
        features = features * (1 + scale) + mean
        return features


def get_time_embedding(cfg: TimeEmbeddingConfig, feature_dim: int) -> TimeEmbedding:
    """Get time embedding according to the configuration."""
    return TimeEmbedding(cfg, feature_dim)
