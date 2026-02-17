"""Latent decoder architecture."""

import abc

import torch
import torch.nn as nn

from src.config import ActClass, NormClass
from src.config.experiment import Experiment
from src.config.options import LatentDecoders
from src.module.layers import LinearLayer


class BaseLatentDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for latent decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        cfg_ae_model = cfg_ae.model
        cfg_latent_decoder = cfg_ae_model.latent_decoder
        self.n_classes: int = cfg.data.dataset.n_classes
        self.z1_dim: int = cfg_ae_model.z1_dim
        self.z2_dim: int = cfg_ae_model.z2_dim
        self.feature_dim: int = cfg_ae_model.feature_dim
        self.act_cls: ActClass = cfg_latent_decoder.act_cls
        self.norm_cls: NormClass = cfg_latent_decoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearLatentDecoder(BaseLatentDecoder):
    """Latent decoder using linear architecture."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_latent_decoder = cfg.autoencoder.model.latent_decoder
        dropout_rate = cfg_latent_decoder.dropout_rate
        mlp_dims = cfg_latent_decoder.mlp_dims
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


def get_latent_decoder() -> BaseLatentDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[LatentDecoders, type[BaseLatentDecoder]] = {
        LatentDecoders.Linear: LinearLatentDecoder,
    }
    return decoder_dict[Experiment.get_config().autoencoder.model.latent_decoder.class_name]()
