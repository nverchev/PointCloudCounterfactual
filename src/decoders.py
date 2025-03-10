import abc
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import PointsConvLayer, LinearLayer
from src.neighbour_ops import graph_filtering
from src.config_options import ExperimentAE, Decoders, WDecoders, MainExperiment
from src.data_structures import OUT_CHAN


class PriorDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        cfg = MainExperiment.get_config()
        cfg_model = cfg.autoencoder.model
        cfg_prior = cfg_model.decoder.prior_decoder
        self.num_classes = cfg.data.dataset.n_classes
        self.z_dim = cfg_model.z_dim
        self.h_dims = cfg_prior.hidden_dims
        self.dropout = cfg_prior.dropout
        self.act_cls = cfg_prior.act_cls
        modules: list[nn.Module] = []
        in_dims = [self.num_classes] + self.h_dims
        out_dims = self.h_dims
        for in_dim, out_dim, do in zip(in_dims, out_dims, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            # modules.append(nn.Dropout(do))
        modules.append(LinearLayer(self.h_dims[-1], self.z_dim, batch_norm=False))
        self.prior = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior(x)


class BaseWDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = MainExperiment.get_config()
        cfg_ae = cfg.autoencoder
        self.num_classes = cfg.data.dataset.n_classes
        cfg_model = cfg_ae.model
        cfg_w_decoder = cfg_model.decoder.w_decoder
        self.w_dim = cfg_model.w_dim
        self.embedding_dim = cfg_model.embedding_dim
        self.num_codes = cfg_model.n_codes
        self.book_size = cfg_model.book_size
        self.z_dim = cfg_model.z_dim
        self.expand = cfg_w_decoder.expand
        self.h_dims = cfg_w_decoder.hidden_dims
        self.dropout = cfg_w_decoder.dropout
        self.act_cls = cfg_w_decoder.act_cls

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class WDecoderLinear(BaseWDecoder):

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        self.dropout = [0.] + self.dropout
        in_dims = [self.z_dim] + self.h_dims
        out_dims = self.h_dims + [self.w_dim * self.expand]
        for in_dim, out_dim, do in zip(in_dims, out_dims, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        self.decode = nn.Sequential(*modules)
        self.conv = nn.Sequential(
            PointsConvLayer(self.expand * self.embedding_dim,
                            self.embedding_dim,
                            groups=self.embedding_dim,
                            batch_norm=False))

    def forward(self, x):
        x = self.decode(x).view(-1, self.expand * self.embedding_dim, self.num_codes)
        x = self.conv(x).transpose(2, 1).reshape(-1, self.w_dim)
        return x


class WDecoderConvolution(BaseWDecoder):

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        total_h_dims = [h_dim * self.num_codes for h_dim in self.h_dims]
        in_dims = [self.z_dim * self.num_codes] + total_h_dims
        out_dims = total_h_dims
        for in_dim, out_dim, do in zip(in_dims, out_dims, self.dropout):
            modules.append(PointsConvLayer(in_dim, out_dim, groups=self.num_codes, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(
            PointsConvLayer(total_h_dims[-1], self.w_dim, groups=self.num_codes, batch_norm=False)
        )
        self.decode = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decode(x.repeat(1, self.num_codes).unsqueeze(2))
        return x.squeeze(2)


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg_ae = ExperimentAE.get_config().model
        cfg_decoder = cfg_ae.decoder
        self.h_dims_map = cfg_decoder.hidden_dims_map
        self.h_dims_conv = cfg_decoder.hidden_dims_conv
        self.w_dim = cfg_ae.w_dim
        self.act_cls = cfg_decoder.act_cls
        self.filtering = cfg_decoder.filtering
        self.sample_dim = cfg_decoder.sample_dim
        self.n_components = cfg_decoder.n_components
        self.tau = cfg_decoder.tau

    @abc.abstractmethod
    def forward(self,
                w: torch.Tensor,
                m: int,
                s: torch.Tensor,
                viz_att: torch.Tensor,
                viz_components: torch.Tensor,
                ) -> torch.Tensor:
        ...


class PCGen(BasePointDecoder):
    concat: bool = False

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        in_dims = [self.sample_dim] + self.h_dims_map
        out_dims = self.h_dims_map
        for in_dim, out_dim in zip(in_dims, out_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, batch_norm=False, act_cls=torch.nn.ReLU))
        modules.append(PointsConvLayer(self.h_dims_map[-1], self.w_dim, batch_norm=False, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)

        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()

        in_dims = [self.w_dim] + self.h_dims_conv
        out_dims = self.h_dims_conv
        for _ in range(self.n_components):
            modules = []
            for in_dim, out_dim in zip(in_dims, out_dims):
                modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, residual=True))
            self.group_conv.append(nn.Sequential(*modules))
            self.group_final.append(
                PointsConvLayer(self.h_dims_conv[-1], OUT_CHAN, batch_norm=False, act_cls=torch.nn.Identity)
            )
        if self.n_components > 1:
            self.att = PointsConvLayer(self.h_dims_conv[-1] * self.n_components,
                                       self.n_components,
                                       batch_norm=False,
                                       act_cls=None)

    def forward(self,
                w: torch.Tensor,
                m: int,
                s: torch.Tensor,
                viz_att: torch.Tensor,
                viz_components: torch.Tensor,
                ) -> torch.Tensor:
        batch = w.size()[0]
        device = w.device
        x = s if s.numel() else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample(x)
        x = self.join_operation(x, w)
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
                assert x_att.shape == viz_att.shape, (f'Shape tensor_out {viz_att.shape} does not match shape '
                                                      f'attention {x_att.shape}')
                # side effects
                viz_att.data = x_att
            if viz_components.numel():  # accessory information for visualization
                assert xs.shape == viz_components.shape, (f'Shape tensor_out {viz_components.shape} does '
                                                          f'not match shape components {xs.shape}')
                # side effects
                viz_components.data = xs
        else:
            x = xs.squeeze(3)
        if self.filtering:
            x = graph_filtering(x)
        return x

    @staticmethod
    def join_operation(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w.unsqueeze(2) * x


def get_decoder() -> BasePointDecoder:
    decoder_dict: dict[Decoders, Type[BasePointDecoder]] = {
        Decoders.PCGen: PCGen,
    }
    return decoder_dict[ExperimentAE.get_config().model.decoder.architecture]()


def get_w_decoder() -> BaseWDecoder:
    decoder_dict: dict[WDecoders, Type[BaseWDecoder]] = {
        WDecoders.Convolution: WDecoderConvolution,
        WDecoders.Linear: WDecoderLinear,
    }
    return decoder_dict[ExperimentAE.get_config().model.decoder.w_decoder.architecture]()
