"""Encoder architecture for the W-Autoencoder."""

import abc
import itertools

import torch
from torch import nn as nn

from src.config import Experiment, ActClass
from src.config.options import WEncoders
from src.module.layers import PointsConvLayer, LinearLayer


class BaseWEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-space encoders in the autoencoder architecture.

    W-space encoders transform point cloud embeddings into a latent space
    representation. This base class provides common configurations and
    interface for different W-encoder implementations.
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder

        # Dataset parameters
        self.num_classes = cfg.data.dataset.n_classes

        # Model configuration
        cfg_ae_arch = cfg_ae.architecture
        cfg_w_encoder = cfg_ae_arch.encoder.w_encoder

        # Latent space dimensions
        self.w_dim: int = cfg_ae_arch.w_dim  # W-space dimension
        self.embedding_dim: int = cfg_ae_arch.embedding_dim  # Input embedding dimension
        self.z1_dim: int = cfg_ae_arch.z1_dim  # Z-space dimension

        # Vector quantization parameters
        self.n_codes: int = cfg_ae_arch.n_codes  # Number of codebook vectors
        self.book_size: int = cfg_ae_arch.book_size  # Size of each codebook

        # Network architecture parameters
        self.proj_dim: int = cfg_w_encoder.proj_dim
        self.n_heads: int = cfg_w_encoder.n_heads
        self.h_dims_conv: tuple[int, ...] = cfg_w_encoder.hidden_dims_conv  # Hidden dimensions for the linear layers
        self.h_dims_lin: tuple[int, ...] = cfg_w_encoder.hidden_dims_lin  # Hidden dimensions for convolutional layers
        self.dropout: tuple[float, ...] = cfg_w_encoder.dropout  # Dropout probabilities
        self.act_cls: ActClass = cfg_w_encoder.act_cls  # Activation function class

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the W-encoder.

        Args:
            x: Input embedding tensor of shape [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Hidden features for hierarchical VAE
            torch.Tensor: Latent variable
        """


def get_w_encoder() -> BaseWEncoder:
    """Returns the correct w_encoder."""
    decoder_dict: dict[WEncoders, type[BaseWEncoder]] = {
        WEncoders.Convolution: WEncoderConvolution,
        WEncoders.Transformers: WEncoderTransformers,
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.encoder.w_encoder.architecture]()


class WEncoderConvolution(BaseWEncoder):
    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        total_h_dims = [h_dim * self.embedding_dim for h_dim in self.h_dims_conv]
        dim_pairs = itertools.pairwise([self.embedding_dim, *total_h_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim))

        self.conv = nn.Sequential(*modules)
        modules = []
        expand_w_dim = self.w_dim * self.h_dims_conv[-1]
        dim_pairs = itertools.pairwise([expand_w_dim, *self.h_dims_lin])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout, strict=False):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))

        modules.append(LinearLayer(self.h_dims_lin[-1], 2 * self.z1_dim, batch_norm=False))  # change to encode
        self.encode = nn.Sequential(*modules)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = x.view(-1, self.n_codes, self.embedding_dim).transpose(2, 1)
        h = self.conv(x).view(-1, self.w_dim * self.h_dims_conv[-1])
        x = self.encode(h)
        return h, x


class WEncoderTransformers(BaseWEncoder):
    """W-space encoder using transformer architecture.

    This encoder uses self-attention mechanisms to transform point cloud
    embeddings into latent space representations, allowing the model to
    capture global dependencies in the input.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = LinearLayer(self.embedding_dim, self.proj_dim, batch_norm=False)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        transformer_layers: list[nn.Module] = []
        for hidden_dim, do in zip(self.h_dims_lin, self.dropout, strict=False):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self.n_heads,
                dim_feedforward=hidden_dim,
                dropout=do,
                activation=self.act_cls(),
                batch_first=True,
                norm_first=True,
            )
            transformer_layers.append(encoder_layer)

        self.transformer = nn.ModuleList(transformer_layers)
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z1_dim, batch_norm=False)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder."""
        batch_size = x.shape[0]
        x = self.input_proj(x.view(batch_size, self.n_codes, self.embedding_dim))
        h = self.positional_encoding.expand(batch_size, -1, -1) + x
        for layer in self.transformer:
            h = layer(h)

        z = self.to_latent(h)
        return z
