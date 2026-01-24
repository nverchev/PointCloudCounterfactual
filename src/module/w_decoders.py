"""Decoder architecture for the W-Autoencoder."""

import abc
import itertools

import torch
from torch import nn as nn

from src.config import Experiment, ActClass, NormClass
from src.config.options import WDecoders
from src.module.layers import LinearLayer, PointsConvLayer


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
        self.mlp_dims: tuple[int, ...] = cfg_w_decoder.mlp_dims
        self.dropout_rates: tuple[float, ...] = cfg_w_decoder.dropout_rates
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
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([(self.z1_dim + self.z2_dim) * self.n_codes, *self.mlp_dims])
        for (in_dim, out_dim), rate in zip(dim_pairs, self.dropout_rates, strict=False):
            modules.append(
                PointsConvLayer(
                    in_dim, out_dim, n_groups_dense=self.n_codes, act_cls=self.act_cls, norm_cls=self.norm_cls
                )
            )
            modules.append(nn.Dropout(rate))

        modules.append(PointsConvLayer(self.mlp_dims[-1], self.w_dim, n_groups_dense=self.n_codes))
        self.decode = nn.Sequential(*modules)
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        z = torch.cat((z1, z2), dim=2)
        x = self.decode(z.view(-1, self.n_codes * (self.z1_dim + self.z2_dim), 1))
        return x.squeeze(2)


class TransformerWDecoder(BaseWDecoder):
    """W-decoder using transformer architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.z1_proj = LinearLayer(self.z2_dim, self.proj_dim, act_cls=self.act_cls)
        self.z2_proj = LinearLayer(self.z2_dim, self.proj_dim, act_cls=self.act_cls)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.memory_positional_embedding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.norm = self.norm_cls(self.proj_dim)
        transformer_layers: list[nn.Module] = []
        for hidden_dim, rate in zip(self.mlp_dims, self.dropout_rates, strict=False):
            layer = nn.TransformerDecoderLayer(
                d_model=self.proj_dim,
                nhead=self.n_heads,
                dropout=rate,
                dim_feedforward=hidden_dim,
                activation=self.act_cls(),
                batch_first=True,
                norm_first=True,
            )
            transformer_layers.append(layer)

        self.transformer = nn.ModuleList(transformer_layers)
        self.compress = LinearLayer(self.proj_dim, self.embedding_dim)
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        z1_proj = self.z1_proj(z1).view(batch_size, self.n_codes, self.proj_dim)
        z2_proj = self.z2_proj(z2).view(batch_size, self.n_codes, self.proj_dim)
        memory = z1_proj + self.memory_positional_embedding.expand(batch_size, -1, -1)
        memory = self.norm(memory.transpose(1, 2)).transpose_(1, 2)
        x = z2_proj + self.positional_embedding.expand(batch_size, -1, -1)
        for layer in self.transformer:
            x = layer(x, memory)

        x = self.compress(x)
        return x.view(batch_size, self.n_codes * self.embedding_dim)


def get_w_decoder() -> BaseWDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[WDecoders, type[BaseWDecoder]] = {
        WDecoders.Linear: LinearWDecoder,
        WDecoders.Transformer: TransformerWDecoder,
    }
    return decoder_dict[Experiment.get_config().w_autoencoder.model.w_decoder.class_name]()
