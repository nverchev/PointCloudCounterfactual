"""Decoder architecture."""

import abc
import itertools
from typing import cast
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ActClass, NormClass
from src.config.experiment import Experiment
from src.config.options import Decoders
from src.data import OUT_CHAN
from src.module.layers import PointsConvLayer, LinearLayer
from src.utils.neighbour_ops import graph_filtering


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for point decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        cfg_decoder = cfg_ae_model.decoder
        self.map_dims: tuple[int, ...] = cfg_decoder.map_dims
        self.conv_dims: tuple[int, ...] = cfg_decoder.conv_dims
        self.mlp_dims: tuple[int, ...] = cfg_decoder.mlp_dims
        self.dropout_rates: tuple[float, ...] = cfg_decoder.dropout_rates
        self.n_codes: int = cfg_ae_model.n_codes
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.n_heads: int = cfg_decoder.n_heads
        self.w_dim: int = cfg_ae_model.w_dim
        self.act_cls: ActClass = cfg_decoder.act_cls
        self.norm_cls: NormClass = cfg_decoder.norm_cls
        self.filtering: bool = cfg_decoder.filter
        self.sample_dim: int = cfg_decoder.sample_dim
        self.n_components: int = cfg_decoder.n_components
        self.tau: float = cfg_decoder.tau

    @abc.abstractmethod
    def forward(self, w: torch.Tensor, n_output_points: int, initial_sampling: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel."""

    _null_tensor: torch.Tensor = torch.empty(0)

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.sample_dim, *self.map_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=torch.nn.ReLU))

        modules.append(PointsConvLayer(self.map_dims[-1], self.w_dim, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)
        self.group_conv = nn.ModuleList()
        self.group_transformer = nn.ModuleList()
        self.group_final = nn.ModuleList()
        self.memory_positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.conv_dims[-1]))
        self.norm = self.norm_cls(self.conv_dims[-1])
        self.proj_w = LinearLayer(self.embedding_dim, self.conv_dims[-1], act_cls=self.act_cls)
        for _ in range(self.n_components):
            modules = []
            dim_pairs = itertools.pairwise([self.w_dim, *self.conv_dims])
            for in_dim, out_dim in dim_pairs:
                modules.append(
                    PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls, use_residual=True)
                )

            self.group_conv.append(nn.Sequential(*modules))
            self.transformer = nn.ModuleList()
            transformer_layers: list[nn.Module] = []
            for hidden_dim, rate in zip(self.mlp_dims, self.dropout_rates, strict=False):
                layer = nn.TransformerDecoderLayer(
                    d_model=self.conv_dims[-1],
                    nhead=self.n_heads,
                    dropout=rate,
                    dim_feedforward=hidden_dim,
                    activation=self.act_cls(),
                    batch_first=True,
                    norm_first=True,
                )
                transformer_layers.append(layer)

            self.group_transformer.append(nn.ModuleList(transformer_layers))
            self.group_final.append(PointsConvLayer(self.conv_dims[-1], OUT_CHAN, use_soft_init=True))

        if self.n_components > 1:
            self.att = PointsConvLayer(self.conv_dims[-1] * self.n_components, self.n_components)

        return

    def _initialize_sampling(
        self, batch: int, n_output_points: int, device: torch.device, initial_sampling: torch.Tensor
    ) -> torch.Tensor:
        """Initialize and normalize the sampling points."""
        if initial_sampling.numel():
            return initial_sampling.to(device)

        return torch.randn(batch, self.sample_dim, n_output_points, device=device)

    def _process_component_groups(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process all component groups and return outputs and attention features."""
        xs_list = []
        group_atts = []

        for group in range(self.n_components):
            x_group = self.group_conv[group](x)
            group_atts.append(x_group)
            transformer = cast(Iterable[nn.TransformerDecoderLayer], self.group_transformer[group])
            x_group = x_group.transpose(2, 1)
            for layer in transformer:
                x_group = layer(x_group, memory=memory)

            x_group = x_group.transpose(2, 1)
            x_group = self.group_final[group](x_group)
            xs_list.append(x_group)

        xs = torch.stack(xs_list, dim=3)
        return xs, group_atts

    def _apply_attention_mixing(self, xs: torch.Tensor, group_atts: list[torch.Tensor]) -> torch.Tensor:
        """Apply attention-based mixing across components if there are more than one, otherwise return the component."""
        if self.n_components > 1:
            x_att = self.att(torch.cat(group_atts, dim=1).contiguous())
            if self.training:
                x_att = F.gumbel_softmax(x_att, tau=self.tau, dim=1)
            else:
                x_att = torch.softmax(x_att / self.tau, dim=1)

            x_att = x_att.transpose(2, 1)
            x = (xs * x_att.unsqueeze(1)).sum(3)
        else:
            x = xs.squeeze(3)

        return x

    def forward(self, w: torch.Tensor, n_output_points: int, initial_sampling: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch = w.size()[0]
        device = w.device

        # Initialize sampling points
        x = self._initialize_sampling(batch, n_output_points, device, initial_sampling)

        # Map and join with latent code
        x = self.map_sample(x)
        x = self._join_operation(x, w)

        # Process component groups
        memory = self.proj_w(w.view(batch, self.n_codes, self.embedding_dim)) + self.memory_positional_encoding
        memory = self.norm(memory.transpose(1, 2)).transpose_(1, 2)

        xs, group_atts = self._process_component_groups(x, memory)

        # Mix components with attention
        x = self._apply_attention_mixing(xs, group_atts)

        # Optional graph filtering
        if self.filtering:
            x = graph_filtering(x)

        return x

    @staticmethod
    def _join_operation(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w.unsqueeze(2) * x


def get_decoder() -> BasePointDecoder:
    """Get decoder according to the configuration."""
    decoder_dict: dict[Decoders, BasePointDecoder] = {
        Decoders.PCGen: PCGen(),
    }
    return decoder_dict[Experiment.get_config().autoencoder.model.decoder.class_name]
