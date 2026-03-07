"""Latent encoder architecture."""

import abc

import torch
import torch.nn as nn

from src.config import ActClass, NormClass
from src.config.options import LatentEncoders, ConditionalLatentEncoders
from src.config.specs import LatentEncoderConfig, ConditionalLatentEncoderConfig
from src.module.layers import LinearLayer


class BaseLatentEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for Latent space encoders in the autoencoder architecture."""

    def __init__(self, cfg: LatentEncoderConfig, feature_dim: int, z1_dim: int) -> None:
        super().__init__()
        self.feature_dim: int = feature_dim
        self.z1_dim: int = z1_dim
        self.act_cls: ActClass = cfg.act_cls
        self.norm_cls: NormClass = cfg.norm_cls
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearLatentEncoder(BaseLatentEncoder):
    """Latent space encoder using linear architecture."""

    def __init__(self, cfg: LatentEncoderConfig, feature_dim: int, z1_dim: int) -> None:
        super().__init__(cfg, feature_dim, z1_dim)
        dropout_rate = cfg.dropout_rate
        mlp_dims = cfg.mlp_dims
        layers = []
        input_dim = self.feature_dim
        for dim in mlp_dims:
            layers.append(LinearLayer(input_dim, dim, act_cls=self.act_cls))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = dim

        layers.append(LinearLayer(input_dim, 2 * self.z1_dim, use_trunc_init=True))
        self.mlp = nn.Sequential(*layers)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear encoder."""
        return self.mlp(x)


class ConditionalPrior(nn.Module):
    """Network for the conditional prior"""

    def __init__(self, n_classes: int, z2_dim: int) -> None:
        super().__init__()
        self.n_classes: int = n_classes
        self.z2_dim: int = z2_dim
        self.prior = LinearLayer(self.n_classes, 2 * self.z2_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prior(x)


class BaseLatentConditionalEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self, cfg: ConditionalLatentEncoderConfig, feature_dim: int, n_classes: int, z2_dim: int) -> None:
        super().__init__()
        self.feature_dim: int = feature_dim
        self.n_classes: int = n_classes
        self.z2_dim: int = z2_dim
        self.act_cls: ActClass = cfg.act_cls
        return

    @abc.abstractmethod
    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearLatentConditionalEncoder(BaseLatentConditionalEncoder):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self, cfg: ConditionalLatentEncoderConfig, feature_dim: int, n_classes: int, z2_dim: int) -> None:
        super().__init__(cfg, feature_dim, n_classes, z2_dim)
        dropout_rate = cfg.dropout_rate
        mlp_dims = cfg.mlp_dims
        layers = []
        input_dim = self.feature_dim + self.n_classes
        for dim in mlp_dims:
            layers.append(LinearLayer(input_dim, dim, act_cls=self.act_cls))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_dim = dim

        layers.append(LinearLayer(input_dim, 2 * self.z2_dim, use_trunc_init=True))
        self.mlp = nn.Sequential(*layers)
        return

    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(torch.cat([probs, x], dim=1))


def get_latent_encoder(cfg: LatentEncoderConfig, feature_dim: int, z1_dim: int) -> BaseLatentEncoder:
    """Get the latent encoder according to the configuration."""
    decoder_dict: dict[LatentEncoders, type[BaseLatentEncoder]] = {
        LatentEncoders.Linear: LinearLatentEncoder,
    }
    return decoder_dict[cfg.class_name](cfg, feature_dim, z1_dim)


def get_conditional_latent_encoder(
    cfg: ConditionalLatentEncoderConfig, feature_dim: int, n_classes: int, z2_dim: int
) -> BaseLatentConditionalEncoder:
    """Get the latent conditional encoder according to the configuration."""
    conditional_dict: dict[ConditionalLatentEncoders, type[BaseLatentConditionalEncoder]] = {
        ConditionalLatentEncoders.Linear: LinearLatentConditionalEncoder,
    }
    return conditional_dict[cfg.class_name](cfg, feature_dim, n_classes, z2_dim)
