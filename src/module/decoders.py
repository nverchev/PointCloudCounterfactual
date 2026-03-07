"""Decoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ActClass, NormClass
from src.config.options import Decoders
from src.config.specs import DecoderConfig
from src.data import IN_CHAN, OUT_CHAN
from src.module.layers import PointsConvLayer, TransformerDecoder, PointsConvResBlock
from src.utils.neighbour_ops import graph_filtering


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for point decoder."""

    def __init__(self, cfg: DecoderConfig, feature_dim: int) -> None:
        super().__init__()
        self.map_dims: tuple[int, ...] = cfg.map_dims
        self.conv_dims: tuple[int, ...] = cfg.conv_dims
        self.n_heads: int = cfg.n_heads
        self.feedforward_dim: int = cfg.feedforward_dim
        self.n_transformer_layers: int = cfg.n_transformer_layers
        self.transformer_dropout: float = cfg.transformer_dropout
        self.feature_dim: int = feature_dim
        self.act_cls: ActClass = cfg.act_cls
        self.norm_cls: NormClass = cfg.norm_cls
        self.filtering: bool = cfg.filter
        self.n_components: int = cfg.n_components
        self.tau: float = cfg.tau

    @abc.abstractmethod
    def forward(
        self,
        initial_sampling: torch.Tensor,
        features: torch.Tensor,
        n_output_points: int,
        x_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel."""

    _null_tensor: torch.Tensor = torch.empty(0)

    def __init__(self, cfg: DecoderConfig, feature_dim: int) -> None:
        super().__init__(cfg, feature_dim)
        self.embedding_dim: int = cfg.embedding_dim
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([IN_CHAN, *self.map_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=torch.nn.ReLU))

        modules.append(PointsConvLayer(self.map_dims[-1], self.feature_dim, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)
        self.conv_block = PointsConvResBlock(
            [self.feature_dim, *self.conv_dims, self.embedding_dim],
            act_cls=self.act_cls,
            norm_cls=self.norm_cls,
        )
        if self.n_transformer_layers > 0:
            self.proj_initial_sampling = PointsConvLayer(IN_CHAN, self.embedding_dim)
            self.transformer = TransformerDecoder(
                embedding_dim=self.embedding_dim,
                n_heads=self.n_heads,
                feedforward_dim=self.feedforward_dim,
                dropout_rate=self.transformer_dropout,
                act_cls=self.act_cls,
                n_layers=self.n_transformer_layers,
                use_final_norm=True,
            )
        self.out_conv = PointsConvLayer(
            self.embedding_dim, self.n_components * OUT_CHAN, use_trunc_init=True, n_groups_layer=self.n_components
        )
        if self.n_components > 1:
            self.att = PointsConvLayer(self.embedding_dim, self.n_components, n_groups_layer=self.n_components)

        return

    def _process_component_groups(
        self, x: torch.Tensor, initial_sampling: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return component outputs and attention features."""
        x = self.conv_block(x)
        if self.n_transformer_layers > 0:
            x.transpose_(2, 1)
            x_init = self.proj_initial_sampling(initial_sampling)
            x_init = x_init.transpose(2, 1)
            x = self.transformer(x, x_init)
            x.transpose_(2, 1)

        x_out = self.out_conv(x)
        return x, x_out

    def _apply_attention_mixing(self, x: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        """Apply attention-based mixing across components if there are more than one, otherwise return the component."""
        if self.n_components > 1:
            batch, _, n_output_points = x_out.shape
            x_out = x_out.view(batch, self.n_components, OUT_CHAN, n_output_points)
            x_att = self.att(x)
            x_att = x_att.unsqueeze(2)
            if self.training:
                x_att = F.gumbel_softmax(x_att, tau=self.tau, dim=1)
            else:
                x_att = torch.softmax(x_att / self.tau, dim=1)

            x = (x_out * x_att).sum(1)
        else:
            x = x_out

        return x

    def forward(
        self,
        initial_sampling: torch.Tensor,
        features: torch.Tensor,
        n_output_points: int,
        x_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        initial_sampling = initial_sampling.transpose(2, 1)

        x = self.map_sample(initial_sampling)
        x = self._join_operation(x, features)
        x, x_out = self._process_component_groups(x, initial_sampling)
        x = self._apply_attention_mixing(x, x_out)
        if self.filtering:
            x = graph_filtering(x)

        x.transpose_(2, 1)
        return x

    @staticmethod
    def _join_operation(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w.unsqueeze(2) * x


class PCGenClusters(PCGen):
    """PCGen with spatial clustering (mean of 4 contiguous points)."""

    def __init__(self, cfg: DecoderConfig, feature_dim: int) -> None:
        super().__init__(cfg, feature_dim)
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([3 * IN_CHAN, *self.map_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=torch.nn.ReLU))

        modules.append(PointsConvLayer(self.map_dims[-1], self.feature_dim, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)
        if self.n_transformer_layers > 0:
            self.proj_initial_sampling = PointsConvLayer(2 * IN_CHAN, self.embedding_dim)

        return

    def forward(
        self,
        initial_sampling: torch.Tensor,
        features: torch.Tensor,
        n_output_points: int,
        x_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for PCGenClusters."""
        initial_sampling = initial_sampling.transpose(2, 1)
        batch, chan, n_points = initial_sampling.shape
        cluster_size, extra = divmod(n_points, 4)
        assert extra == 0, 'Number of points must be divisible by 4'

        if x_0 is None:
            x_0 = torch.zeros_like(initial_sampling)
        else:
            x_0 = x_0.transpose(2, 1)

        clusters = initial_sampling.view(batch, chan, cluster_size, 4)
        cluster_means = clusters.mean(dim=-1)
        cluster_features = cluster_means.repeat_interleave(4, dim=2)

        x_in = torch.cat([initial_sampling, cluster_features, x_0], dim=1)
        x = self.map_sample(x_in)
        x = self._join_operation(x, features)

        source_reduced = x_0[:, :, ::4]
        transformer_init = torch.cat([cluster_means, source_reduced], dim=1)

        x, x_out = self._process_component_groups(x, transformer_init)
        x = self._apply_attention_mixing(x, x_out)
        if self.filtering:
            x = graph_filtering(x)

        x.transpose_(2, 1)
        return x

    def _process_component_groups(
        self, x: torch.Tensor, initial_sampling: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return component outputs and attention features with pooled initial sampling."""
        x = self.conv_block(x)
        if self.n_transformer_layers > 0:
            x.transpose_(2, 1)
            x_init = self.proj_initial_sampling(initial_sampling)
            x_init = x_init.transpose(2, 1)
            x = self.transformer(x, x_init)
            x.transpose_(2, 1)

        x_out = self.out_conv(x)
        return x, x_out


def get_decoder(cfg: DecoderConfig, feature_dim: int) -> BasePointDecoder:
    """Get decoder according to the configuration."""
    decoder_dict: dict[Decoders, type[BasePointDecoder]] = {
        Decoders.PCGen: PCGen,
        Decoders.PCGenClusters: PCGenClusters,
    }
    return decoder_dict[cfg.class_name](cfg, feature_dim)
