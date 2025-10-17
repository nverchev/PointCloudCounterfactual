"""Contains decoder classes."""

import abc
import itertools
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device

from src.layers import PointsConvLayer, LinearLayer
from src.neighbour_ops import graph_filtering
from src.config_options import Decoders, WDecoders, Experiment
from src.data_structures import OUT_CHAN


class PriorDecoder(nn.Module):
    """Network for the conditional prior"""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_arc = cfg.autoencoder.architecture
        cfg_prior = cfg_ae_arc.decoder.prior_decoder
        self.h_conv = cfg_ae_arc.encoder.w_encoder.hidden_dims_conv
        self.hidden_features = cfg_ae_arc.hidden_features
        self.w_dim = cfg_ae_arc.w_dim
        self.num_classes = cfg.data.dataset.n_classes
        self.z_dim = cfg_ae_arc.z2_dim
        self.h_dims = cfg_prior.hidden_dims
        self.dropout = cfg_prior.dropout
        self.act_cls = cfg_prior.act_cls
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.num_classes, *self.h_dims])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(LinearLayer(self.h_dims[-1], 2 * self.z_dim, batch_norm=False))
        self.prior = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prior(x)


class PosteriorDecoder(PriorDecoder):
    """Network for the conditional posterior"""

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        expand_w_dim = self.w_dim * self.h_conv[-1]
        dim_pairs = itertools.pairwise([self.num_classes + expand_w_dim, *self.h_dims])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(LinearLayer(self.h_dims[-1], 2 * self.z_dim, batch_norm=False))
        self.prior = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prior(x)


class BaseWDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        self.num_classes = cfg.data.dataset.n_classes
        cfg_ae_arc = cfg_ae.architecture
        cfg_w_decoder = cfg_ae_arc.decoder.w_decoder
        self.w_dim = cfg_ae_arc.w_dim
        self.embedding_dim = cfg_ae_arc.embedding_dim
        self.n_codes = cfg_ae_arc.n_codes
        self.book_size = cfg_ae_arc.book_size
        self.z_dim = cfg_ae_arc.z1_dim + cfg_ae_arc.z2_dim
        self.expand = cfg_w_decoder.expand
        self.h_dims = cfg_w_decoder.hidden_dims
        self.dropout = cfg_w_decoder.dropout
        self.act_cls = cfg_w_decoder.act_cls

    @abc.abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class WDecoderLinear(BaseWDecoder):
    """W-decoder with linear layers."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        self.dropout = tuple([0., *self.dropout])
        expanded_w_dim = self.w_dim * self.expand
        dim_pairs = itertools.pairwise([self.z_dim, *self.h_dims, expanded_w_dim])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        self.decode = nn.Sequential(*modules)
        self.conv = nn.Sequential(
            PointsConvLayer(self.expand * self.embedding_dim,
                            self.embedding_dim,
                            groups=self.embedding_dim,
                            batch_norm=False))

    def forward(self, z: torch.Tensor):
        """Forward pass."""
        x = self.decode(z).view(-1, self.expand * self.embedding_dim, self.n_codes)
        x = self.conv(x).transpose(2, 1).reshape(-1, self.w_dim)
        return x


class WDecoderConvolution(BaseWDecoder):
    """W-decoder with convolutional layers."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        total_h_dims = [h_dim * self.n_codes for h_dim in self.h_dims]
        dim_pairs = itertools.pairwise([self.z_dim * self.n_codes, *total_h_dims])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout):
            modules.append(PointsConvLayer(in_dim, out_dim, groups=self.n_codes, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(
            PointsConvLayer(total_h_dims[-1], self.w_dim, groups=self.n_codes, batch_norm=False)
        )
        self.decode = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor):
        """Forward pass."""
        x = self.decode(z.repeat(1, self.n_codes).unsqueeze(2))
        return x.squeeze(2)


class WDecoderTransformers(BaseWDecoder):
    """W-space encoder using transformer architecture.

    This encoder uses self-attention mechanisms to transform point cloud
    embeddings into latent space representations, allowing the model to
    capture global dependencies in the input.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj_dim = 128
        self.decode =PointsConvLayer(self.z_dim * self.n_codes, self.proj_dim * self.n_codes, groups=self.n_codes, batch_norm=False)


        self.input_proj = LinearLayer(self.z_dim, self.proj_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        transformer_layers: list[nn.Module] = []
        for hidden_dim, do in zip(self.h_dims, self.dropout):
            layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead= 4,
                dropout=do,
                dim_feedforward=hidden_dim,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            transformer_layers.append(layer)

        self.transformer = nn.Sequential(*transformer_layers)
        self.compress = nn.Linear(self.proj_dim, self.embedding_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder.

        Args:
            z: Input embedding tensor of shape [batch_size, w_dim]

        Returns:
            w
        """
        batch_size = z.shape[0]
        x = self.decode(z.repeat(1, self.n_codes).unsqueeze(2)).view(batch_size, self.n_codes, self.proj_dim)
        x = self.query_tokens.expand(batch_size, -1, -1) + x
        x = self.transformer(x)
        return self.compress(x).view(batch_size, self.n_codes * self.embedding_dim)

#
# class WDecoderTransformers(BaseWDecoder):
#     """W-space encoder using transformer architecture.
#
#     This encoder uses self-attention mechanisms to transform point cloud
#     embeddings into latent space representations, allowing the model to
#     capture global dependencies in the input.
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.proj_dim = 128
#         self.latent_proj = nn.Linear(self.z_dim,  self.proj_dim)
#         self.query_tokens = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
#         decoder_layers: list[nn.Module] = []
#         for hidden_dim, do in zip(self.h_dims, self.dropout):
#             decoder_layer = nn.TransformerDecoderLayer(
#                 d_model=self.proj_dim,
#                 nhead=4,
#                 dim_feedforward=hidden_dim,
#                 dropout=do,
#                 batch_first=True,
#                 norm_first=True,
#                 activation='gelu'
#             )
#             decoder_layers.append(decoder_layer)
#
#         self.decoder = nn.ModuleList(decoder_layers)
#         encoder_layers: list[nn.Module] = []
#
#         for hidden_dim, do in zip(self.h_dims, self.dropout):
#             encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=self.proj_dim,
#                 nhead=4,
#                 dim_feedforward=hidden_dim,
#                 dropout=do,
#                 batch_first=True,
#                 norm_first=True,
#                 activation='gelu'
#             )
#             encoder_layers.append(encoder_layer)
#
#         self.encoder = nn.ModuleList(encoder_layers)
#         self.output_proj = nn.Linear(self.proj_dim, self.embedding_dim)
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """Forward pass through transformer encoder.
#
#         Args:
#             z: Input embedding tensor of shape [batch_size, w_dim]
#
#         Returns:
#             w
#
#         """
#         batch_size = z.shape[0]
#         z_expanded = self.latent_proj(z).view(batch_size, 1, self.proj_dim)
#         x = self.query_tokens.expand(z.size(0), -1, -1)
#         for decoder, encoder in zip(self.decoder, self.encoder):
#             x = decoder(x, z_expanded)
#             x = encoder(x)
#
#         return self.output_proj(x).view(batch_size, self.n_codes * self.embedding_dim)


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for point decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_arc = Experiment.get_config().autoencoder.architecture
        cfg_decoder = cfg_ae_arc.decoder
        self.h_dims_map = cfg_decoder.hidden_dims_map
        self.h_dims_conv = cfg_decoder.hidden_dims_conv
        self.w_dim = cfg_ae_arc.w_dim
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
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel. """
    concat: bool = False

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
            self.att = PointsConvLayer(self.h_dims_conv[-1] * self.n_components,
                                       self.n_components,
                                       batch_norm=False)

    def forward(self,
                w: torch.Tensor,
                m: int,
                s: torch.Tensor,
                viz_att: torch.Tensor = torch.empty(0),
                viz_components: torch.Tensor = torch.empty(0),
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
    def _join_operation(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return w.unsqueeze(2) * x


def get_decoder() -> BasePointDecoder:
    """Get decoder according to the configuration."""
    decoder_dict: dict[Decoders, BasePointDecoder] = {
        Decoders.PCGen: PCGen(),
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.decoder.architecture]


def get_w_decoder() -> BaseWDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[WDecoders, BaseWDecoder] = {
        WDecoders.Convolution: WDecoderConvolution(),
        WDecoders.Linear: WDecoderLinear(),
        WDecoders.TransformerCross: WDecoderTransformers(),
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.decoder.w_decoder.architecture]
