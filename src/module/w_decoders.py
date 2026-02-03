"""Decoder architecture for the W-Autoencoder."""

import abc

import torch
from torch import nn as nn

from src.config import Experiment, ActClass, NormClass
from src.config.options import WDecoders
from src.module.layers import LinearLayer, PointsConvLayer, PointsConvResBlock, TransformerDecoder


class BaseWDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        cfg_ae_model = cfg_ae.model
        cfg_wae_model = cfg.w_autoencoder.model
        cfg_w_decoder = cfg_wae_model.w_decoder
        self.n_classes: int = cfg.data.dataset.n_classes
        self.w_dim: int = cfg_ae_model.w_dim
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.n_codes: int = cfg_ae_model.n_codes
        self.book_size: int = cfg_ae_model.book_size
        self.z1_dim: int = cfg_wae_model.z1_dim
        self.z2_dim: int = cfg_wae_model.z2_dim
        self.proj_dim: int = cfg_w_decoder.proj_dim
        self.n_heads: int = cfg_w_decoder.n_heads
        self.conv_dims: tuple[int, ...] = cfg_w_decoder.conv_dims
        self.n_transformer_layers: int = cfg_w_decoder.n_transformer_layers
        self.feedforward_dim: int = cfg_w_decoder.feedforward_dim
        self.transformer_dropout: float = cfg_w_decoder.transformer_dropout
        self.act_cls: ActClass = cfg_w_decoder.act_cls
        self.norm_cls: NormClass = cfg_w_decoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearWDecoder(BaseWDecoder):
    """W-decoder using grouped linear layers (PointConv for a sequence of 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            PointsConvResBlock(
                [(self.z1_dim + self.z2_dim) * self.n_codes, *self.conv_dims],
                n_groups_layer=self.n_codes,
                act_cls=self.act_cls,
                norm_cls=self.norm_cls,
            ),
            PointsConvLayer(self.conv_dims[-1], self.w_dim, n_groups_dense=self.n_codes),
        )
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        z = torch.cat((z1, z2), dim=2)
        x = self.decode(z.view(-1, self.n_codes * (self.z1_dim + self.z2_dim), 1))
        return x.squeeze(2)


class TransformerWDecoder(BaseWDecoder):
    """W-decoder using transformer architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.z1_proj = LinearLayer(self.z2_dim, self.proj_dim)
        self.z2_proj = LinearLayer(self.z2_dim, self.proj_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.memory_positional_embedding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.transformer = TransformerDecoder(
            embedding_dim=self.proj_dim,
            n_heads=self.n_heads,
            hidden_dim=self.feedforward_dim,
            dropout_rate=self.transformer_dropout,
            act_cls=self.act_cls,
            n_layers=self.n_transformer_layers,
        )
        self.compress = LinearLayer(self.proj_dim, self.embedding_dim)
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        z1_proj = self.z1_proj(z1).view(batch_size, self.n_codes, self.proj_dim)
        z2_proj = self.z2_proj(z2).view(batch_size, self.n_codes, self.proj_dim)
        memory = z1_proj + self.memory_positional_embedding.expand(batch_size, -1, -1)
        x = z2_proj + self.positional_embedding.expand(batch_size, -1, -1)
        x = self.transformer(x, memory)
        x = self.compress(x)
        return x.view(batch_size, self.n_codes * self.embedding_dim)


def get_w_decoder() -> BaseWDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[WDecoders, type[BaseWDecoder]] = {
        WDecoders.Linear: LinearWDecoder,
        WDecoders.Transformer: TransformerWDecoder,
    }
    return decoder_dict[Experiment.get_config().w_autoencoder.model.w_decoder.class_name]()
