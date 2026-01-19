"""Decoder architecture for the W-Autoencoder."""

import abc
import itertools

import torch
from torch import nn as nn

from src.config import Experiment, ActClass
from src.config.options import WDecoders
from src.module.layers import LinearLayer, PointsConvLayer


class BaseWDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        self.n_classes: int = cfg.data.dataset.n_classes
        cfg_ae_arc = cfg_ae.architecture
        cfg_w_decoder = cfg_ae_arc.decoder.w_decoder
        self.w_dim: int = cfg_ae_arc.w_dim
        self.embedding_dim: int = cfg_ae_arc.embedding_dim
        self.n_codes: int = cfg_ae_arc.n_codes
        self.book_size: int = cfg_ae_arc.book_size
        self.z1_dim: int = cfg_ae_arc.z1_dim
        self.z2_dim: int = cfg_ae_arc.z2_dim
        self.proj_dim: int = cfg_w_decoder.proj_dim
        self.n_heads: int = cfg_w_decoder.n_heads
        self.h_dims: tuple[int, ...] = cfg_w_decoder.hidden_dims
        self.dropout: tuple[float, ...] = cfg_w_decoder.dropout
        self.act_cls: ActClass = cfg_w_decoder.act_cls

    @abc.abstractmethod
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class PriorDecoder(nn.Module):
    """Network for the conditional prior"""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_arc = cfg.autoencoder.architecture
        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_arc.n_codes
        self.z2_dim: int = cfg_ae_arc.z2_dim
        self.prior = LinearLayer(self.n_classes, self.n_codes * 2 * self.z2_dim, batch_norm=False)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prior(x).view(-1, self.n_codes, 2 * self.z2_dim)


class PosteriorDecoder(nn.Module):
    """Network for the conditional posterior"""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_arc = cfg.autoencoder.architecture
        cfg_posterior = cfg_ae_arc.decoder.posterior_decoder
        self.hidden_features: int = cfg_ae_arc.hidden_features
        self.w_dim: int = cfg_ae_arc.w_dim
        self.embedding_dim: int = cfg_ae_arc.embedding_dim
        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_arc.n_codes
        self.z2_dim: int = cfg_ae_arc.z2_dim
        self.h_dims: tuple[int, ...] = cfg_posterior.hidden_dims
        self.dropout: tuple[float, ...] = cfg_posterior.dropout
        self.act_cls: ActClass = cfg_posterior.act_cls
        self.proj_dim: int = cfg_ae_arc.encoder.w_encoder.proj_dim
        self.n_heads: int = cfg_posterior.n_heads
        self.input_proj = LinearLayer(self.embedding_dim, self.proj_dim, batch_norm=False)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.prob_proj = LinearLayer(self.n_classes, self.proj_dim, batch_norm=False)
        transformer_layers: list[nn.Module] = []
        for hidden_dim, do in zip(self.h_dims, self.dropout, strict=False):
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
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z2_dim, batch_norm=False, soft_init=True)
        return

    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        x = self.input_proj(x.view(batch_size, self.n_codes, self.embedding_dim))
        x = self.positional_encoding.expand(batch_size, -1, -1) + x + self.prob_proj(probs).unsqueeze(1)
        for layer in self.transformer:
            x = layer(x)

        return self.to_latent(x)


class WDecoderLinear(BaseWDecoder):
    """W-decoder with linear layers."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        self.dropout: tuple[float, ...] = (0.0, *self.dropout)
        expanded_w_dim = self.w_dim * self.proj_dim
        dim_pairs = itertools.pairwise([self.z1_dim + self.z2_dim, *self.h_dims, expanded_w_dim])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout, strict=False):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        self.decode = nn.Sequential(*modules)
        self.conv = nn.Sequential(
            PointsConvLayer(
                self.proj_dim * self.embedding_dim, self.embedding_dim, groups=self.embedding_dim, batch_norm=False
            )
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """Forward pass."""
        z = torch.cat((z1, z2), dim=1)
        x = self.decode(z).view(-1, self.proj_dim * self.embedding_dim, self.n_codes)
        x = self.conv(x).transpose(2, 1).reshape(-1, self.w_dim)
        return x


class WDecoderConvolution(BaseWDecoder):
    """W-decoder with convolutional layers."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        total_h_dims = [h_dim * self.n_codes for h_dim in self.h_dims]
        dim_pairs = itertools.pairwise([(self.z1_dim + self.z2_dim) * self.n_codes, *total_h_dims])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout, strict=False):
            modules.append(PointsConvLayer(in_dim, out_dim, groups=self.n_codes, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(PointsConvLayer(total_h_dims[-1], self.w_dim, groups=self.n_codes, batch_norm=False))
        self.decode = nn.ModuleList(modules)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """Forward pass."""
        z = torch.cat((z1, z2), dim=1)
        x = self.decode(z.repeat(1, self.n_codes).unsqueeze(2))
        return x.squeeze(2)


class WDecoderTransformers(BaseWDecoder):
    """W-space encoder using transformer architecture.

    This encoder uses self-attention mechanisms to transform point cloud
    embeddings into latent space representations, allowing the model to
    capture global dependencies in the input.
    """

    def __init__(self) -> None:
        super().__init__()
        self.z1_proj = LinearLayer(self.z2_dim, self.proj_dim, batch_norm=False)
        self.z2_proj = LinearLayer(self.z2_dim, self.proj_dim, batch_norm=False)
        self.query_tokens = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.key_tokens = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        transformer_layers: list[nn.Module] = []
        for hidden_dim, do in zip(self.h_dims, self.dropout, strict=False):
            layer = nn.TransformerDecoderLayer(
                d_model=self.proj_dim,
                nhead=self.n_heads,
                dropout=do,
                dim_feedforward=hidden_dim,
                activation=self.act_cls(),
                batch_first=True,
                norm_first=True,
            )
            transformer_layers.append(layer)

        self.transformer = nn.ModuleList(transformer_layers)
        self.compress = LinearLayer(self.proj_dim, self.embedding_dim, batch_norm=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder."""
        batch_size = z1.shape[0]
        z1_proj = self.z1_proj(z1).view(batch_size, self.n_codes, self.proj_dim)
        z2_proj = self.z2_proj(z2).view(batch_size, self.n_codes, self.proj_dim)
        x = z2_proj + self.key_tokens.expand(batch_size, -1, -1)
        y = z1_proj + self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.transformer:
            x = layer(x, y)

        x = self.compress(x)
        return x.view(batch_size, self.n_codes * self.embedding_dim)


def get_w_decoder() -> BaseWDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[WDecoders, type[BaseWDecoder]] = {
        WDecoders.Convolution: WDecoderConvolution,
        WDecoders.Linear: WDecoderLinear,
        WDecoders.TransformerCross: WDecoderTransformers,
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.decoder.w_decoder.architecture]()
