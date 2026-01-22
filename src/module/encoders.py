"""Encoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn

from src.config import ActClass, Experiment
from src.config.options import Encoders
from src.data import IN_CHAN
from src.module.layers import EdgeConvLayer, PointsConvLayer
from src.utils.neighbour_ops import get_graph_features, graph_max_pooling


class BasePointEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_neighbors: int = cfg_ae_model.encoder.n_neighbors
        self.conv_dims: tuple[int, ...] = cfg_ae_model.encoder.conv_dims
        self.w_dim: int = cfg_ae_model.w_dim
        self.act_cls: ActClass = cfg_ae_model.encoder.act_cls

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class DGCNN(BasePointEncoder):
    """Dynamic Graph Convolutional Neural Network encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.h_dim = (64, 64, 128, 256)
        modules = [EdgeConvLayer(2 * IN_CHAN, self.h_dim[0])]
        for in_dim, out_dim in itertools.pairwise(self.h_dim):
            modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls))

        self.edge_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.h_dim), self.w_dim, batch_norm=False)
        return

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        xs = []
        x = x.transpose(2, 1)
        for conv in self.edge_convolutions:
            indices, x = get_graph_features(x, k=self.n_neighbors, indices=indices)  # [batch, features, num_points, k]
            indices = torch.empty(0)  # finds new neighbors dynamically every iteration
            x = conv(x)
            x = x.max(dim=3, keepdim=False)[0]  # [batch, features, num_points]
            xs.append(x)

        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class LDGCNN(BasePointEncoder):
    """Lighter version of DGCNN where the graph is only calculated once."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.conv_dims[0])
        modules: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise(self.conv_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls))

        self.points_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), self.w_dim, batch_norm=False)
        return

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(2, 1)
        indices, x = get_graph_features(x, k=self.n_neighbors, indices=indices)
        x = self.edge_conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        xs = [x]
        for conv in self.points_convolutions:
            x = graph_max_pooling(x, k=self.n_neighbors, indices=indices)
            x = conv(x)
            xs.append(x)

        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


def get_encoder() -> BasePointEncoder:
    """Get encoder according to the configuration."""
    dict_encoder: dict[Encoders, type[BasePointEncoder]] = {
        Encoders.LDGCNN: LDGCNN,
        Encoders.DGCNN: DGCNN,
    }
    return dict_encoder[Experiment.get_config().autoencoder.model.encoder.class_name]()
