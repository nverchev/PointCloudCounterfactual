"""Decoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ActClass
from src.config.experiment import Experiment
from src.config.options import Decoders
from src.data import OUT_CHAN
from src.module.layers import PointsConvLayer
from src.utils.neighbour_ops import graph_filtering


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for point decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        cfg_decoder = cfg_ae_model.decoder
        self.map_dims: tuple[int, ...] = cfg_decoder.map_dims
        self.conv_dims: tuple[int, ...] = cfg_decoder.conv_dims
        self.w_dim: int = cfg_ae_model.w_dim
        self.act_cls: ActClass = cfg_decoder.act_cls
        self.filtering: bool = cfg_decoder.filter
        self.sample_dim: int = cfg_decoder.sample_dim
        self.n_components: int = cfg_decoder.n_components
        self.tau: float = cfg_decoder.tau

    @abc.abstractmethod
    def forward(self, w: torch.Tensor, n_generated_points: int, initial_sampling: torch.Tensor | None) -> torch.Tensor:
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel."""

    _null_tensor: torch.Tensor = torch.empty(0)

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.sample_dim, *self.map_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, batch_norm=False, act_cls=torch.nn.ReLU))

        modules.append(PointsConvLayer(self.map_dims[-1], self.w_dim, batch_norm=False, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()
        for _ in range(self.n_components):
            modules = []
            dim_pairs = itertools.pairwise([self.w_dim, *self.conv_dims])
            for in_dim, out_dim in dim_pairs:
                modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, residual=True))

            self.group_conv.append(nn.Sequential(*modules))
            self.group_final.append(PointsConvLayer(self.conv_dims[-1], OUT_CHAN, batch_norm=False, soft_init=True))

        if self.n_components > 1:
            self.att = PointsConvLayer(self.conv_dims[-1] * self.n_components, self.n_components, batch_norm=False)

        return

    def _initialize_sampling(
        self, batch: int, n_generated_points: int, device: torch.device, initial_sampling: torch.Tensor | None
    ) -> torch.Tensor:
        """Initialize and normalize the sampling points."""
        if initial_sampling is None:
            x = torch.randn(batch, self.sample_dim, n_generated_points, device=device)
        else:
            x = initial_sampling

        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return x

    def _process_component_groups(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process all component groups and return outputs and attention features."""
        xs_list = []
        group_atts = []

        for group in range(self.n_components):
            x_group = self.group_conv[group](x)
            group_atts.append(x_group)
            x_group = self.group_final[group](x_group)
            xs_list.append(x_group)

        xs = torch.stack(xs_list, dim=3)
        return xs, group_atts

    def _apply_attention_mixing(self, xs: torch.Tensor, group_atts: list[torch.Tensor]) -> torch.Tensor:
        """Apply attention-based mixing across components or select single component."""
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

    def forward(self, w: torch.Tensor, n_generated_points: int, initial_sampling: torch.Tensor | None) -> torch.Tensor:
        """Forward pass."""
        batch = w.size()[0]
        device = w.device

        # Initialize sampling points
        x = self._initialize_sampling(batch, n_generated_points, device, initial_sampling)

        # Map and join with latent code
        x = self.map_sample(x)
        x = self._join_operation(x, w)

        # Process component groups
        xs, group_atts = self._process_component_groups(x)

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
