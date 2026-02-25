"""Decoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ActClass, NormClass
from src.config.experiment import Experiment
from src.config.options import Decoders
from src.data import OUT_CHAN
from src.module.layers import PointsConvLayer, TransformerEncoder, PointsConvResBlock
from src.utils.neighbour_ops import graph_filtering


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for point decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        cfg_decoder = cfg_ae_model.decoder
        self.map_dims: tuple[int, ...] = cfg_decoder.map_dims
        self.conv_dims: tuple[int, ...] = cfg_decoder.conv_dims
        self.n_heads: int = cfg_decoder.n_heads
        self.feedforward_dim: int = cfg_decoder.feedforward_dim
        self.n_transformer_layers: int = cfg_decoder.n_transformer_layers
        self.transformer_dropout: float = cfg_decoder.transformer_dropout
        self.feature_dim: int = cfg_ae_model.feature_dim
        self.act_cls: ActClass = cfg_decoder.act_cls
        self.norm_cls: NormClass = cfg_decoder.norm_cls
        self.filtering: bool = cfg_decoder.filter
        self.sample_dim: int = cfg_decoder.sample_dim
        self.n_components: int = cfg_decoder.n_components
        self.tau: float = cfg_decoder.tau

    @abc.abstractmethod
    def forward(self, initial_sampling: torch.Tensor, features: torch.Tensor, n_output_points: int) -> torch.Tensor:
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel."""

    _null_tensor: torch.Tensor = torch.empty(0)

    def __init__(self) -> None:
        super().__init__()
        decoder_cfg = Experiment.get_config().autoencoder.model.decoder
        self.embedding_dim: int = decoder_cfg.embedding_dim
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.sample_dim, *self.map_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=torch.nn.ReLU))

        modules.append(PointsConvLayer(self.map_dims[-1], self.feature_dim, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)
        self.conv_block = PointsConvResBlock(
            [self.feature_dim, *self.conv_dims, self.embedding_dim],
            act_cls=self.act_cls,
            norm_cls=self.norm_cls,
        )
        self.transformer = TransformerEncoder(
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

    def _initialize_sampling(self, batch: int, n_output_points: int, device: torch.device) -> torch.Tensor:
        """Initialize and normalize the sampling points."""
        return torch.randn(batch, self.sample_dim, n_output_points, device=device)

    def _process_component_groups(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return component outputs and attention features."""
        x = self.conv_block(x)
        x.transpose_(2, 1)
        x = self.transformer(x)
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

    def forward(self, initial_sampling: torch.Tensor, features: torch.Tensor, n_output_points: int) -> torch.Tensor:
        """Forward pass.

        Args:
            initial_sampling: Initial sampling points
            features: features of dimension [Batch, feature_dim]
            n_output_points: Number of output points
        """
        batch = features.size()[0]
        device = features.device
        if initial_sampling.numel():
            x = initial_sampling
        else:
            x = self._initialize_sampling(batch, n_output_points, device)

        x = self.map_sample(x)
        x = self._join_operation(x, features)
        x, x_out = self._process_component_groups(x)
        x = self._apply_attention_mixing(x, x_out)
        if self.filtering:
            x = graph_filtering(x)

        x.transpose_(2, 1)
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
