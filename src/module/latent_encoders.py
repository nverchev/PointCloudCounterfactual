"""Latent encoder architecture."""

import abc

import torch
import torch.nn as nn

from src.config import ActClass, NormClass
from src.config.experiment import Experiment
from src.config.options import LatentEncoders, ConditionalLatentEncoders
from src.module.layers import LinearLayer


class BaseLatentEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for Latent space encoders in the autoencoder architecture."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        cfg_ae_model = cfg_ae.model
        cfg_latent_encoder = cfg_ae_model.latent_encoder
        self.feature_dim: int = cfg_ae_model.feature_dim
        self.z1_dim: int = cfg_ae_model.z1_dim
        self.act_cls: ActClass = cfg_latent_encoder.act_cls
        self.norm_cls: NormClass = cfg_latent_encoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearLatentEncoder(BaseLatentEncoder):
    """Latent space encoder using linear architecture."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_latent_encoder = cfg.autoencoder.model.latent_encoder
        dropout_rate = cfg_latent_encoder.dropout_rate
        mlp_dims = cfg_latent_encoder.mlp_dims
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

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_classes: int = cfg.data.dataset.n_classes
        self.z2_dim: int = cfg_ae_model.z2_dim
        self.prior = LinearLayer(self.n_classes, 2 * self.z2_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prior(x)


class BaseLatentConditionalEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_posterior = cfg_ae_model.conditional_latent_encoder
        self.feature_dim: int = cfg_ae_model.feature_dim
        self.n_classes: int = cfg.data.dataset.n_classes
        self.z2_dim: int = cfg_ae_model.z2_dim
        self.act_cls: ActClass = cfg_posterior.act_cls
        return

    @abc.abstractmethod
    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearLatentConditionalEncoder(BaseLatentConditionalEncoder):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_posterior = cfg.autoencoder.model.conditional_latent_encoder
        dropout_rate = cfg_posterior.dropout_rate
        mlp_dims = cfg_posterior.mlp_dims
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


def get_latent_encoder() -> BaseLatentEncoder:
    """Get the latent encoder according to the configuration."""
    decoder_dict: dict[LatentEncoders, type[BaseLatentEncoder]] = {
        LatentEncoders.Linear: LinearLatentEncoder,
    }
    return decoder_dict[Experiment.get_config().autoencoder.model.latent_encoder.class_name]()


def get_conditional_latent_encoder() -> BaseLatentConditionalEncoder:
    """Get the latent conditional encoder according to the configuration."""
    conditional_dict: dict[ConditionalLatentEncoders, type[BaseLatentConditionalEncoder]] = {
        ConditionalLatentEncoders.Linear: LinearLatentConditionalEncoder,
    }
    return conditional_dict[Experiment.get_config().autoencoder.model.conditional_latent_encoder.class_name]()
