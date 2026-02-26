"""Module for time embeddings and vector modulation."""

import torch
import torch.nn as nn
from src.config import Experiment
from src.module.layers import LinearLayer, SinusoidalPositionalEmbedding


class TimeEmbedding(nn.Module):
    """Time embedding module for Flow Matching."""

    def __init__(self, feature_dim: int):
        super().__init__()
        cfg_model = Experiment.get_config().flow.model
        embedding_dim = cfg_model.time_embedding_dim
        mlp_dims = cfg_model.mlp_dims

        self.time_embedding = SinusoidalPositionalEmbedding(dim=embedding_dim)

        layers = []
        curr_dim = embedding_dim
        for next_dim in mlp_dims:
            layers.append(LinearLayer(curr_dim, next_dim, act_cls=nn.GELU))
            curr_dim = next_dim

        layers.append(LinearLayer(curr_dim, 2 * feature_dim, use_trunc_init=True))
        self.time_mlp = nn.Sequential(*layers)

        self.feature_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)

    def forward(self, t: torch.Tensor, opt_features: torch.Tensor | None = None) -> torch.Tensor:
        """Embed time and optionally modulate features."""
        if opt_features is None:
            feature_dim = self.feature_norm.normalized_shape[0]
            features = torch.randn(t.shape[0], feature_dim, device=t.device)
        else:
            features = opt_features

        t_embed = self.time_embedding(t)
        modulation = self.time_mlp(t_embed)
        mean, scale = modulation.chunk(2, dim=1)
        features = self.feature_norm(features)
        features = features * (1 + scale) + mean
        return features
