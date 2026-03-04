"""Latent decoder architecture."""

import abc

import torch
import torch.nn as nn

from src.config import ActClass, NormClass
from src.config.options import LatentDecoders
from src.config.specs import LatentDecoderConfig
from src.module.layers import LinearLayer


class BaseLatentDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for latent decoder."""

    def __init__(self, cfg: LatentDecoderConfig, z1_dim: int, z2_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.z1_dim: int = z1_dim
        self.z2_dim: int = z2_dim
        self.feature_dim: int = feature_dim
        self.act_cls: ActClass = cfg.act_cls
        self.norm_cls: NormClass = cfg.norm_cls
        return

    @abc.abstractmethod
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearLatentDecoder(BaseLatentDecoder):
    """Latent decoder using linear architecture."""

    def __init__(self, cfg: LatentDecoderConfig, z1_dim: int, z2_dim: int, feature_dim: int) -> None:
        super().__init__(cfg, z1_dim, z2_dim, feature_dim)
        dropout_rate = cfg.dropout_rate
        mlp_dims = cfg.mlp_dims
        layers = []
        input_dim = self.z1_dim + self.z2_dim
        for dim in mlp_dims:
            layers.append(LinearLayer(input_dim, dim, act_cls=self.act_cls))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_dim = dim

        layers.append(LinearLayer(input_dim, self.feature_dim))
        self.mlp = nn.Sequential(*layers)
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(torch.cat([z1, z2], dim=1))


def get_latent_decoder(cfg: LatentDecoderConfig, z1_dim: int, z2_dim: int, feature_dim: int) -> BaseLatentDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[LatentDecoders, type[BaseLatentDecoder]] = {
        LatentDecoders.Linear: LinearLatentDecoder,
    }
    return decoder_dict[cfg.class_name](cfg, z1_dim, z2_dim, feature_dim)
