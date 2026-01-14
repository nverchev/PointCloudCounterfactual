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
        cfg_ae_arc = Experiment.get_config().autoencoder.architecture
        cfg_decoder = cfg_ae_arc.decoder
        self.h_dims_map: tuple[int, ...] = cfg_decoder.hidden_dims_map
        self.h_dims_conv: tuple[int, ...] = cfg_decoder.hidden_dims_conv
        self.w_dim: int = cfg_ae_arc.w_dim
        self.act_cls: ActClass = cfg_decoder.act_cls
        self.filtering: bool = cfg_decoder.filtering
        self.sample_dim: int = cfg_decoder.sample_dim
        self.n_components: int = cfg_decoder.n_components
        self.tau: float = cfg_decoder.tau

    @abc.abstractmethod
    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
        viz_att: torch.Tensor,
        viz_components: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel."""

    concat: bool = False
    _null_tensor: torch.Tensor = torch.empty(0)

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.sample_dim, *self.h_dims_map])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, batch_norm=False, act_cls=torch.nn.ReLU))

        modules.append(PointsConvLayer(self.h_dims_map[-1], self.w_dim, batch_norm=False, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()

        for _ in range(self.n_components):
            modules = []
            dim_pairs = itertools.pairwise([self.w_dim, *self.h_dims_conv])
            for in_dim, out_dim in dim_pairs:
                modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, residual=True))
            self.group_conv.append(nn.Sequential(*modules))
            self.group_final.append(
                PointsConvLayer(self.h_dims_conv[-1], OUT_CHAN, batch_norm=False, act_cls=torch.nn.Identity)
            )
        if self.n_components > 1:
            self.att = PointsConvLayer(self.h_dims_conv[-1] * self.n_components, self.n_components, batch_norm=False)

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
        viz_att: torch.Tensor = _null_tensor,
        viz_components: torch.Tensor = _null_tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        batch = w.size()[0]
        device = w.device
        x = s if s.numel() else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample(x)
        x = self._join_operation(x, w)
        xs_list = []
        group_atts = []
        for group in range(self.n_components):
            x_group = self.group_conv[group](x)
            group_atts.append(x_group)
            x_group = self.group_final[group](x_group)
            xs_list.append(x_group)
        xs = torch.stack(xs_list, dim=3)
        if self.n_components > 1:
            x_att = self.att(torch.cat(group_atts, dim=1).contiguous())

            if self.training:
                x_att = F.gumbel_softmax(x_att, tau=self.tau, dim=1)
            else:
                # good approximation of the expected weights
                x_att = torch.softmax(x_att / self.tau, dim=1)

            x_att = x_att.transpose(2, 1)
            x = (xs * x_att.unsqueeze(1)).sum(3)
            if viz_att.numel():  # accessory information for visualization
                assert x_att.shape == viz_att.shape, (
                    f'Shape tensor_out {viz_att.shape} does not match shape attention {x_att.shape}'
                )
                # side effects
                viz_att.data = x_att
            if viz_components.numel():  # accessory information for visualization
                assert xs.shape == viz_components.shape, (
                    f'Shape tensor_out {viz_components.shape} does not match shape components {xs.shape}'
                )
                # side effects
                viz_components.data = xs
        else:
            x = xs.squeeze(3)
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
    return decoder_dict[Experiment.get_config().autoencoder.architecture.decoder.architecture]
