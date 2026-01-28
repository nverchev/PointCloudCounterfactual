"""Encoder architecture for the W-Autoencoder."""

import abc

import torch
from torch import nn as nn

from src.config import Experiment, ActClass, NormClass
from src.config.options import WEncoders
from src.module.layers import PointsConvLayer, LinearLayer, PointsConvBlock, TransformerEncoder


class BaseWEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-space encoders in the autoencoder architecture."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder

        # Dataset parameters
        self.n_classes = cfg.data.dataset.n_classes

        # Model configuration
        cfg_ae_model = cfg_ae.model
        cfg_wae_model = cfg.w_autoencoder.model
        cfg_w_encoder = cfg_wae_model.w_encoder

        # Latent space dimensions
        self.w_dim: int = cfg_ae_model.w_dim
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.z1_dim: int = cfg_wae_model.z1_dim

        # Vector quantization parameters
        self.n_codes: int = cfg_ae_model.n_codes
        self.book_size: int = cfg_ae_model.book_size  # Size of each codebook

        # Network architecture parameters
        self.proj_dim: int = cfg_w_encoder.proj_dim
        self.n_heads: int = cfg_w_encoder.n_heads
        self.conv_dims: tuple[int, ...] = cfg_w_encoder.conv_dims
        self.n_transformer_layers: int = cfg_w_encoder.n_transformer_layers
        self.transformer_feedforward_dim: int = cfg_w_encoder.transformer_feedforward_dim
        self.transformer_dropout: float = cfg_w_encoder.transformer_dropout
        self.act_cls: ActClass = cfg_w_encoder.act_cls
        self.norm_cls: NormClass = cfg_w_encoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""


class ConvolutionalWEncoder(BaseWEncoder):
    """W-space encoder that uses a convolutional architecture with 1x1 kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            PointsConvBlock([self.embedding_dim, *self.conv_dims], act_cls=self.act_cls, norm_cls=self.norm_cls),
            PointsConvLayer(self.conv_dims[-1], 2 * self.z1_dim, use_trunc_init=True),
        )
        return

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(2, 1)
        x = self.encode(x).transpose(2, 1)
        return x


class TransformerWEncoder(BaseWEncoder):
    """W-space encoder using transformer architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = LinearLayer(self.embedding_dim, self.proj_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.transformer = TransformerEncoder(
            in_dim=self.proj_dim,
            n_heads=self.n_heads,
            feedforward_dim=self.transformer_feedforward_dim,
            act_cls=self.act_cls,
            dropout_rate=self.transformer_dropout,
            n_layers=self.n_transformer_layers,
        )
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z1_dim, use_trunc_init=True)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder."""
        batch_size = x.shape[0]
        x = self.input_proj(x)
        x = self.positional_encoding.expand(batch_size, -1, -1) + x
        x = self.transformer(x)
        return self.to_latent(x)


def get_w_encoder() -> BaseWEncoder:
    """Returns the correct w_encoder."""
    decoder_dict: dict[WEncoders, type[BaseWEncoder]] = {
        WEncoders.Convolutional: ConvolutionalWEncoder,
        WEncoders.Transformer: TransformerWEncoder,
    }
    return decoder_dict[Experiment.get_config().w_autoencoder.model.w_encoder.class_name]()
