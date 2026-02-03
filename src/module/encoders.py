"""Encoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn

from src.config import ActClass, Experiment, NormClass
from src.config.options import Encoders
from src.data import IN_CHAN
from src.module.layers import EdgeConvLayer, PointsConvLayer, LinearLayer, TransformerDecoder
from src.utils.neighbour_ops import get_graph_features, graph_max_pooling


class BasePointEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_neighbors: int = cfg_ae_model.encoder.n_neighbors
        self.conv_dims: tuple[int, ...] = cfg_ae_model.encoder.conv_dims
        self.w_dim: int = cfg_ae_model.w_dim
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.n_heads: int = cfg_ae_model.encoder.n_heads
        self.proj_dim: int = cfg_ae_model.encoder.proj_dim
        self.n_transformer_layers: int = cfg_ae_model.encoder.n_transformer_layers
        self.feedforward_dim: int = cfg_ae_model.encoder.feedforward_dim
        self.transformer_dropout: float = cfg_ae_model.encoder.transformer_dropout
        self.n_codes: int = cfg_ae_model.n_codes
        self.act_cls: ActClass = cfg_ae_model.encoder.act_cls
        self.norm_cls: NormClass = cfg_ae_model.encoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class DGCNN(BasePointEncoder):
    """Dynamic Graph Convolutional Neural Network encoder."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        for in_dim, out_dim in itertools.pairwise((IN_CHAN, *self.conv_dims)):
            modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))

        self.edge_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), self.w_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        xs = []
        x = x.transpose(2, 1)
        indices = torch.empty(0)
        for conv in self.edge_convolutions:
            indices, x = get_graph_features(x, n_neighbors=self.n_neighbors, indices=indices)
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
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.conv_dims[0], act_cls=self.act_cls, norm_cls=self.norm_cls)
        modules: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise(self.conv_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))

        self.points_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), self.w_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(2, 1)
        indices, x = get_graph_features(x, n_neighbors=self.n_neighbors, indices=torch.empty(0))
        x = self.edge_conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        xs = [x]
        for conv in self.points_convolutions:
            x = graph_max_pooling(x, n_neighbors=self.n_neighbors, indices=indices)
            x = conv(x)
            xs.append(x)

        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class TransformerEncoder(BasePointEncoder):
    """Transformer encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.conv_dims[0], act_cls=self.act_cls, norm_cls=self.norm_cls)
        modules: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise(self.conv_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))

        self.points_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), self.w_dim)
        self.proj_codes = LinearLayer(self.embedding_dim, self.proj_dim, act_cls=self.act_cls)
        self.proj_input = LinearLayer(IN_CHAN, self.proj_dim, act_cls=self.act_cls)
        self.transformer_codes = TransformerDecoder(
            embedding_dim=self.proj_dim,
            n_heads=self.n_heads,
            hidden_dim=self.feedforward_dim,
            dropout_rate=self.transformer_dropout,
            act_cls=self.act_cls,
            n_layers=self.n_transformer_layers,
            use_final_norm=True,
        )
        self.compress = LinearLayer(self.proj_dim, self.embedding_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch = x.shape[0]
        y = x.transpose(2, 1)
        indices, y = get_graph_features(y, n_neighbors=self.n_neighbors, indices=torch.empty(0))
        y = self.edge_conv(y)
        y = y.max(dim=3, keepdim=False)[0]
        xs = [y]
        for conv in self.points_convolutions:
            y = graph_max_pooling(y, n_neighbors=self.n_neighbors, indices=indices)
            y = conv(y)
            xs.append(y)

        y = torch.cat(xs, dim=1).contiguous()
        y = self.final_conv(y)
        x_max = y.max(dim=2, keepdim=False)[0]
        queries = self.proj_codes(x_max.view(batch, self.n_codes, self.embedding_dim))
        memory = self.proj_input(x)
        x = self.transformer_codes(queries, memory)
        x = self.compress(x)
        return x.view(batch, self.w_dim)


def get_encoder() -> BasePointEncoder:
    """Get encoder according to the configuration."""
    dict_encoder: dict[Encoders, type[BasePointEncoder]] = {
        Encoders.LDGCNN: LDGCNN,
        Encoders.DGCNN: DGCNN,
        Encoders.Transformer: TransformerEncoder,
    }
    return dict_encoder[Experiment.get_config().autoencoder.model.encoder.class_name]()
