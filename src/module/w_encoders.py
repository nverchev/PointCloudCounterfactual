"""Encoder architecture for the W-Autoencoder."""

import abc
import itertools

import torch
from torch import nn as nn

from src.config import Experiment, ActClass
from src.config.options import WEncoders
from src.module.layers import PointsConvLayer, LinearLayer


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
        self.embedding_dim: int = cfg_ae_model.embedding_dim  # Input embedding dimension
        self.z1_dim: int = cfg_wae_model.z1_dim  # Z-space dimension

        # Vector quantization parameters
        self.n_codes: int = cfg_ae_model.n_codes  # Number of codebook vectors
        self.book_size: int = cfg_ae_model.book_size  # Size of each codebook

        # Network architecture parameters
        self.proj_dim: int = cfg_w_encoder.proj_dim
        self.n_heads: int = cfg_w_encoder.n_heads
        self.conv_dims: tuple[int, ...] = cfg_w_encoder.conv_dims  # Hidden dimensions for the convolutional layers
        self.mlp_dims: tuple[int, ...] = cfg_w_encoder.mlp_dims  # Hidden dimensions for mlp layers
        self.dropout_rates: tuple[float, ...] = cfg_w_encoder.dropout_rates  # Dropout probabilities
        self.act_cls: ActClass = cfg_w_encoder.act_cls  # Activation function class
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""


class ConvolutionalWEncoder(BaseWEncoder):
    """W-space encoder that uses a convolutional architecture with 1x1 kernel."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.embedding_dim, *self.conv_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim))

        modules.append(PointsConvLayer(self.conv_dims[-1], 2 * self.z1_dim, batch_norm=False, soft_init=True))
        self.encode = nn.Sequential(*modules)
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
        self.input_proj = LinearLayer(self.embedding_dim, self.proj_dim, batch_norm=False)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        transformer_layers: list[nn.Module] = []
        for hidden_dim, drate in zip(self.mlp_dims, self.dropout_rates, strict=False):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self.n_heads,
                dim_feedforward=hidden_dim,
                dropout=drate,
                activation=self.act_cls(),
                batch_first=True,
                norm_first=True,
            )
            transformer_layers.append(encoder_layer)

        self.transformer = nn.ModuleList(transformer_layers)
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z1_dim, batch_norm=False, soft_init=True)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder."""
        batch_size = x.shape[0]
        x = self.input_proj(x)
        x = self.positional_encoding.expand(batch_size, -1, -1) + x
        for layer in self.transformer:
            x = layer(x)

        return self.to_latent(x)


def get_w_encoder() -> BaseWEncoder:
    """Returns the correct w_encoder."""
    decoder_dict: dict[WEncoders, type[BaseWEncoder]] = {
        WEncoders.Convolutional: ConvolutionalWEncoder,
        WEncoders.Transformer: TransformerWEncoder,
    }
    return decoder_dict[Experiment.get_config().w_autoencoder.model.w_encoder.class_name]()
