"""Decoders of the codewords and of the point clouds, and the conditional distributions of the inner VAE."""

import abc
import itertools

from typing import Final, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.experiment import Experiment
from src.config.torch import ActClass
from src.data.structures import OUT_CHAN
from src.module.layers import (
    GraphFilter,
    LinearLayer,
    Modulate,
    PointsConvLayer,
    Repeat,
    SphereProjection,
)

__all__ = [
    'BasePointDecoder',
    'BaseWDecoder',
    'ComponentAttention',
    'ConditionalGaussian',
    'PCGen',
    'WDecoder',
]


class ConditionalGaussian(nn.Module):
    """Mean and log variance of a Gaussian, computed from a conditioning vector.

    Attributes:
        act_cls: Activation of hidden layers.
        bn_momentum: Momentum of batch normalization.
        z_y_dim: Dimension of the latent variable z_y.
        layers: Maps the condition to the mean and the log variance, together.
    """

    act_cls: Final[ActClass]
    bn_momentum: Final[float]
    z_y_dim: Final[int]
    layers: nn.Sequential

    def __init__(self, in_dim: int, h_dims: tuple[int, ...], /) -> None:
        """Initialize.

        Args:
            in_dim: dimension of the condition.
            h_dims: hidden dimensions; empty for a linear layer.
        """
        super().__init__()
        cfg_model = Experiment.get_config().w_autoencoding.model
        cfg_arch = cfg_model.architecture
        self.act_cls = cfg_arch.act_cls
        self.bn_momentum = cfg_arch.bn_momentum
        self.z_y_dim = cfg_model.z_y_dim
        self.layers = self._get_layers(in_dim, h_dims)
        return

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the log variance ``(B, Z_y)`` of the conditions ``(B, in_dim)``."""
        mu, log_var = self.layers(x).chunk(2, 1)
        return mu, log_var

    def _get_layers(self, in_dim: int, h_dims: tuple[int, ...]) -> nn.Sequential:
        dims = (in_dim, *h_dims)
        hidden_layers = [
            LinearLayer(in_d, out_d, act_cls=self.act_cls, bn_momentum=self.bn_momentum)
            for in_d, out_d in itertools.pairwise(dims)
        ]
        final_layer = LinearLayer(dims[-1], 2 * self.z_y_dim, act_cls=None, batch_norm=False)
        return nn.Sequential(*hidden_layers, final_layer)


class BaseWDecoder(nn.Module, abc.ABC):
    """Abstract base of the decoders of the latent variables into the codeword.

    Attributes:
        w_dim: Dimension of the codeword.
        dim_codes: Number of codes of a codeword.
        z_dim: Dimension of the latent variables together.
        h_dims: Channels per code of the hidden layers.
        dropout: Dropout probability.
        act_cls: Activation of the layers.
        bn_momentum: Momentum of the batch normalization.
    """

    w_dim: Final[int]
    dim_codes: Final[int]
    z_dim: Final[int]
    h_dims: Final[tuple[int, ...]]
    dropout: Final[float]
    act_cls: Final[ActClass]
    bn_momentum: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoding.model
        cfg_model = cfg.w_autoencoding.model
        cfg_arch = cfg_model.architecture
        self.w_dim = cfg_ae_model.w_dim
        self.dim_codes = cfg_ae_model.dim_codes
        self.z_dim = cfg_model.z_u_dim + cfg_model.z_y_dim
        self.h_dims = cfg_arch.decoder_h_dims
        self.dropout = cfg_arch.dropout
        self.act_cls = cfg_arch.act_cls
        self.bn_momentum = cfg_arch.bn_momentum
        return

    @abc.abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent variables ``(B, Z)`` into the codewords ``(B, W)``."""


class WDecoder(BaseWDecoder):
    """Decoder of the latent variables into the codeword, as in PCGen.

    Each embedding has its own weights, so the decoding is computed in parallel.

    Attributes:
        layers: Maps the latent variables to the codeword.
    """

    layers: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.layers = nn.Sequential(
            Repeat(1, self.dim_codes),
            nn.Unflatten(1, (self.dim_codes * self.z_dim, 1)),
            *self._get_conv_layers(),
            PointsConvLayer(
                self.dim_codes * self.h_dims[-1],
                self.w_dim,
                act_cls=None,
                batch_norm=False,
                groups=self.dim_codes,
            ),
            nn.Flatten(),
        )
        return

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)

    def _get_conv_layers(self) -> list[nn.Module]:
        conv_layers: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise((self.z_dim, *self.h_dims)):
            conv_layers.append(
                PointsConvLayer(
                    self.dim_codes * in_dim,
                    self.dim_codes * out_dim,
                    act_cls=self.act_cls,
                    bn_momentum=self.bn_momentum,
                    groups=self.dim_codes,
                )
            )
            conv_layers.append(nn.Dropout1d(self.dropout))

        return conv_layers


class ComponentAttention(nn.Module):
    """Choose, for every point, how much each component contributes.

    While training, Gumbel noise makes the choice explore the components;
    evaluation uses the softmax without noise.

    Attributes:
        score: Layer scoring every component from the features of all of them.
        temperature: Temperature of the softmax.
    """

    score: PointsConvLayer
    temperature: Final[float]

    def __init__(self, in_dim: int, n_components: int, *, temperature: float) -> None:
        """Initialize.

        Args:
            in_dim: features of all the components together.
            n_components: components to choose among.
            temperature: temperature of the softmax.
        """
        super().__init__()
        self.score = PointsConvLayer(in_dim, n_components, act_cls=None, batch_norm=False)
        self.temperature = temperature
        return

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the share of every component for every point ``(B, M, C)``."""
        scores = self.score(features)
        if self.training:
            return F.gumbel_softmax(scores, tau=self.temperature, dim=1).transpose(2, 1)

        return F.softmax(scores / self.temperature, dim=1).transpose(2, 1)

    @override
    def extra_repr(self) -> str:
        return f'temperature={self.temperature}'


class BasePointDecoder(nn.Module, abc.ABC):
    """Abstract base of the decoders of the codewords into point clouds.

    Attributes:
        w_dim: Dimension of the codeword.
        sample_dim: Dimension of the sampling of the generating surface.
        map_h_dims: Hidden dimensions of the map of the sampling.
        conv_h_dims: Channels of the point convolutions of each component.
        n_components: Components of the decoder.
        act_cls: Activation of the point convolutions.
        attention_temperature: Temperature of the component selection softmax.
        filtering: Whether the graph filter sharpens the decoded cloud.
        bn_momentum: Momentum of the batch normalization.
    """

    w_dim: Final[int]
    sample_dim: Final[int]
    map_h_dims: Final[tuple[int, ...]]
    conv_h_dims: Final[tuple[int, ...]]
    n_components: Final[int]
    act_cls: Final[ActClass]
    attention_temperature: Final[float]
    filtering: Final[bool]
    bn_momentum: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg_model = Experiment.get_config().autoencoding.model
        cfg_arch = cfg_model.architecture
        cfg_decoder = cfg_arch.decoder
        self.w_dim = cfg_model.w_dim
        self.sample_dim = cfg_decoder.sample_dim
        self.map_h_dims = cfg_decoder.map_h_dims
        self.conv_h_dims = cfg_decoder.conv_h_dims
        self.n_components = cfg_decoder.n_components
        self.act_cls = cfg_decoder.act_cls
        self.attention_temperature = cfg_decoder.attention_temperature
        self.filtering = cfg_decoder.filtering
        self.bn_momentum = cfg_arch.bn_momentum
        return

    @abc.abstractmethod
    def forward(self, w: torch.Tensor, m: int, sampling: torch.Tensor) -> torch.Tensor:
        """Decode the codeword into a cloud.

        Args:
            w: codewords ``(B, W)``.
            m: points of each cloud, when the sampling is drawn here.
            sampling: sampling of the generating surface ``(B, D, M)``; empty to draw one.

        Returns:
            The cloud ``(B, OUT_CHAN, M)``.
        """


class PCGen(BasePointDecoder):
    """Decoder mapping the points of a fixed distribution to a cloud, in parallel.

    Attributes:
        sphere: Projects the sampling on the sphere.
        map_sample: Maps the sampling to the dimension of the codeword.
        join: Conditions the mapped sampling on the codeword.
        group_conv: Layers of each component.
        group_final: Final layer of each component.
        attention: Mixes the components, when there are several.
        filter: Graph filter, or the identity.
    """

    sphere: SphereProjection
    map_sample: nn.Sequential
    join: Modulate
    group_conv: nn.ModuleList
    group_final: nn.ModuleList
    attention: ComponentAttention | None
    filter: nn.Module

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.sphere = SphereProjection()
        self.map_sample = self._get_map_sample()
        self.join = Modulate()
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()
        for _ in range(self.n_components):
            self.group_conv.append(self._get_component_conv())
            self.group_final.append(
                PointsConvLayer(self.conv_h_dims[-1], OUT_CHAN, act_cls=nn.Identity, batch_norm=False)
            )

        self.attention = self._get_attention()
        self.filter = GraphFilter() if self.filtering else nn.Identity()
        return

    @override
    def forward(self, w: torch.Tensor, m: int, sampling: torch.Tensor) -> torch.Tensor:
        """Decode the codeword into a cloud.

        Args:
            w: codewords ``(B, W)``.
            m: points of each cloud, when the sampling is drawn here.
            sampling: sampling of the generating surface ``(B, D, M)``; empty to draw one.

        Returns:
            The coordinates ``(B, OUT_CHAN, M)``.
        """
        x = sampling if sampling.numel() else torch.randn(w.size(0), self.sample_dim, m, device=w.device)
        x = self.join(self.map_sample(self.sphere(x)), w)
        features = [conv(x) for conv in self.group_conv]
        components = torch.stack([final(f) for final, f in zip(self.group_final, features, strict=True)], dim=3)
        if self.attention is None:
            return self.filter(components.squeeze(3))

        attention = self.attention(torch.cat(features, dim=1).contiguous())
        return self.filter((components * attention.unsqueeze(1)).sum(3))

    def _get_map_sample(self) -> nn.Sequential:
        map_layers: list[nn.Module] = [
            PointsConvLayer(in_dim, out_dim, act_cls=nn.ReLU, batch_norm=False)
            for in_dim, out_dim in itertools.pairwise((self.sample_dim, *self.map_h_dims))
        ]
        last_map = PointsConvLayer(self.map_h_dims[-1], self.w_dim, act_cls=nn.Hardtanh, batch_norm=False)
        return nn.Sequential(*map_layers, last_map)

    def _get_component_conv(self) -> nn.Sequential:
        return nn.Sequential(
            *(
                PointsConvLayer(
                    in_dim,
                    out_dim,
                    act_cls=self.act_cls,
                    bn_momentum=self.bn_momentum,
                    residual=True,
                )
                for in_dim, out_dim in itertools.pairwise((self.w_dim, *self.conv_h_dims))
            )
        )

    def _get_attention(self) -> ComponentAttention | None:
        if self.n_components == 1:
            return None

        total_dim = self.conv_h_dims[-1] * self.n_components
        return ComponentAttention(total_dim, self.n_components, temperature=self.attention_temperature)
