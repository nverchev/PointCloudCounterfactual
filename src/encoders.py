import abc
from typing import Type

import torch
import torch.nn as nn

from src.layers import PointsConvLayer, LinearLayer, EdgeConvLayer
from src.neighbour_ops import get_graph_features, graph_max_pooling
from src.config_options import ExperimentAE, Encoders, WEncoders
from src.data_structures import IN_CHAN


class BaseWEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = ExperimentAE.get_config()
        cfg_ae = cfg.autoencoder
        cfg_w_encoder = cfg.autoencoder.encoder.w_encoder
        self.w_dim = cfg_ae.w_dim
        self.embedding_dim = cfg_ae.embedding_dim
        self.num_codes = cfg_ae.num_codes
        self.book_size = cfg_ae.book_size
        self.z_dim = cfg.autoencoder.z_dim
        self.h_dims_conv = cfg_w_encoder.hidden_dims_conv
        self.h_dims_lin = cfg_w_encoder.hidden_dims_lin
        self.dropout = cfg_w_encoder.dropout
        self.act_cls = cfg_w_encoder.act_cls

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class WEncoderConvolution(BaseWEncoder):

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        total_h_dims = [h_dim * self.embedding_dim for h_dim in self.h_dims_conv]
        in_dims = [self.embedding_dim] + total_h_dims
        out_dims = total_h_dims
        for in_dim, out_dim, do in zip(in_dims, out_dims, self.dropout):
            modules.append(PointsConvLayer(in_dim, out_dim))
        self.conv = nn.Sequential(*modules)
        modules = []
        in_dims = [self.w_dim * self.h_dims_conv[-1]] + self.h_dims_lin
        out_dims = self.h_dims_lin
        for in_dim, out_dim, do in zip(in_dims, out_dims, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(
            LinearLayer(self.h_dims_lin[-1], 2 * self.z_dim)
        )  # change to encode
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x).view(-1, self.w_dim * self.h_dims_conv[-1])
        x = self.encode(x)
        return x


class BasePointEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = ExperimentAE.get_config()
        self.k = cfg.autoencoder.encoder.k
        self.h_dim = cfg.autoencoder.encoder.hidden_dims
        self.w_dim = cfg.autoencoder.w_dim
        self.act_cls = cfg.autoencoder.encoder.act_cls

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        pass


class LDGCNN(BasePointEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.h_dim[0])
        modules: list[nn.Module] = []
        in_dims = self.h_dim[:-1]
        out_dims = self.h_dim[1:]
        for in_dim, out_dim in zip(in_dims, out_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls))
        self.points_convs = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.h_dim), self.w_dim, batch_norm=False)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 1)
        indices, x = get_graph_features(x, k=self.k, indices=indices)
        x = self.edge_conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        xs = [x]
        for conv in self.points_convs:
            x = graph_max_pooling(x, k=self.k, indices=indices)
            x = conv(x)
            xs.append(x)
        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class DGCNN(BasePointEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.h_dim = [64, 64, 128, 256]
        modules = [EdgeConvLayer(2 * IN_CHAN, self.h_dim[0])]
        in_dims = self.h_dim[:-1]
        out_dims = self.h_dim[1:]
        for in_dim, out_dim in zip(in_dims, out_dims):
            modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls))
        self.edge_convs = nn.Sequential(*modules)
        self.final_conv = nn.Conv1d(sum(self.h_dim), self.w_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        xs = []
        x = x.transpose(2, 1)
        for conv in self.edge_convs:
            indices, x = get_graph_features(x, k=self.k, indices=indices)  # [batch, features, num_points, k]
            indices = torch.empty(0)  # finds new neighbours dynamically every iteration
            x = conv(x)
            x = x.max(dim=3, keepdim=False)[0]  # [batch, features, num_points]
            xs.append(x)
        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        x = x_max
        return x


def get_encoder() -> BasePointEncoder:
    dict_encoder: dict[Encoders, Type[BasePointEncoder]] = {
        Encoders.LDGCNN: LDGCNN,
        Encoders.DGCNN: DGCNN,
    }
    return dict_encoder[ExperimentAE.get_config().autoencoder.encoder.architecture]()


def get_w_encoder() -> BaseWEncoder:
    decoder_dict: dict[WEncoders, Type[BaseWEncoder]] = {
        WEncoders.Convolution: WEncoderConvolution,
    }
    return decoder_dict[ExperimentAE.get_config().autoencoder.encoder.w_encoder.architecture]()
