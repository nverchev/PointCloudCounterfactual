"""Encoders of the point clouds and of the codewords."""

import abc
import itertools

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.torch import ActClass
from src.data.structures import IN_CHAN
from src.module.layers import EdgeConvLayer, EdgeFeatures, LinearLayer, Max, PointsConvLayer, Transpose

__all__ = [
    'BasePointEncoder',
    'BaseWEncoder',
    'DGCNNEncoder',
    'WEncoder',
]


class BasePointEncoder(nn.Module, abc.ABC):
    """Abstract base of the encoders of the point clouds.

    Attributes:
        w_dim: Dimension of the codeword.
        n_neighbours: Neighbours of every point in the edge convolutions.
        h_dims: Channels of the edge convolutions.
        act_cls: Activation of the layers.
        bn_momentum: Momentum of the batch normalization.
    """

    w_dim: Final[int]
    n_neighbours: Final[int]
    h_dims: Final[tuple[int, ...]]
    act_cls: Final[ActClass]
    bn_momentum: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg_model = Experiment.get_config().autoencoding.model
        cfg_arch = cfg_model.architecture
        self.w_dim = cfg_model.w_dim
        self.n_neighbours = cfg_arch.encoder.n_neighbours
        self.h_dims = cfg_arch.encoder.h_dims
        self.act_cls = cfg_arch.encoder.act_cls
        self.bn_momentum = cfg_arch.bn_momentum
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the coordinates ``(B, N, IN_CHAN)`` into the codeword ``(B, W)``."""


class DGCNNEncoder(BasePointEncoder):
    """Dynamic Graph CNN encoder of the point clouds.

    Attributes:
        transpose: Moves the coordinates to the channel dimension.
        blocks: Edge convolutions, each on the graph of the features of the block before.
        head: Projects the features of all the blocks to the codeword.
    """

    transpose: Transpose
    blocks: nn.ModuleList
    head: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.transpose = Transpose(2, 1)
        self.blocks = self._get_blocks()
        self.head = self._get_head()
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        return self.head(torch.cat(features, dim=1).contiguous())

    def _get_blocks(self) -> nn.ModuleList:
        blocks = [
            nn.Sequential(
                EdgeFeatures(self.n_neighbours),
                EdgeConvLayer(2 * IN_CHAN, self.h_dims[0], act_cls=None, bn_momentum=self.bn_momentum),
                Max(dim=3),
            )
        ]
        for in_dim, out_dim in itertools.pairwise(self.h_dims):
            layer = EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum)
            blocks.append(nn.Sequential(EdgeFeatures(self.n_neighbours), layer, Max(dim=3)))

        return nn.ModuleList(blocks)

    def _get_head(self) -> nn.Sequential:
        return nn.Sequential(nn.Conv1d(sum(self.h_dims), self.w_dim, kernel_size=1), Max(dim=2))


class BaseWEncoder(nn.Module, abc.ABC):
    """Abstract base of the encoders of the codewords.

    Attributes:
        w_dim: Dimension of the codeword.
        embedding_dim: Dimension of each code.
        dim_codes: Number of codes of a codeword.
        n_classes: Number of classes of the evaluations.
        z_u_dim: Dimension of the latent variable unrelated to the classification.
        z_y_dim: Dimension of the latent variable the evaluation conditions.
        h_dims: Expansion of the channels of each code.
        dropout: Dropout probability.
        act_cls: Activation of the layers.
        bn_momentum: Momentum of the batch normalization.
    """

    w_dim: Final[int]
    embedding_dim: Final[int]
    dim_codes: Final[int]
    n_classes: Final[int]
    z_u_dim: Final[int]
    z_y_dim: Final[int]
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
        self.embedding_dim = cfg_ae_model.embedding_dim
        self.dim_codes = cfg_ae_model.dim_codes
        self.n_classes = cfg.data.dataset.n_classes
        self.z_u_dim = cfg_model.z_u_dim
        self.z_y_dim = cfg_model.z_y_dim
        self.h_dims = cfg_arch.encoder_h_dims
        self.dropout = cfg_arch.dropout
        self.act_cls = cfg_arch.act_cls
        self.bn_momentum = cfg_arch.bn_momentum
        return

    @abc.abstractmethod
    def forward(self, w_q: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode the codewords under the evaluation.

        Args:
            w_q: codewords ``(B, W)``.
            condition: evaluation, a point of the probability simplex ``(B, C)``.

        Returns:
            The shifts of the mean and of the log variance of ``z_y`` from its prior ``(B, Z_y)``,
            and the mean and the log variance of ``z_u`` ``(B, Z_u)``.
        """


class WEncoder(BaseWEncoder):
    """Encoder of the codewords, as in PCGen.

    It expands each code with shared weights and concatenates the outputs. Two heads then infer the
    parameters of the latent variables: the one of ``z_y`` also reads the evaluation, so that the
    posterior is ``q(z_y | w, y)``, while the one of ``z_u`` does not.

    Attributes:
        expansion: Maps the codes to the features shared by the two heads.
        head_u: Maps the features to the parameters of ``z_u``.
        head_y: Maps the evaluation and the features to the parameters of ``z_y``.
    """

    expansion: nn.Sequential
    head_u: nn.Sequential
    head_y: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        features_dim = self.w_dim * self.h_dims[-1]
        self.expansion = self._get_expansion()
        self.head_u = self._get_head(features_dim, 2 * self.z_u_dim)
        self.head_y = self._get_head(self.n_classes + features_dim, 2 * self.z_y_dim)
        return

    @override
    def forward(self, w_q: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.expansion(w_q)
        d_mu_y, d_log_var_y = self.head_y(torch.cat((condition, features), dim=1)).chunk(2, 1)
        mu_u, log_var_u = self.head_u(features).chunk(2, 1)
        return d_mu_y, d_log_var_y, mu_u, log_var_u

    def _get_conv_layers(self) -> list[nn.Module]:
        return [
            PointsConvLayer(
                in_dim * self.embedding_dim,
                out_dim * self.embedding_dim,
                act_cls=self.act_cls,
                bn_momentum=self.bn_momentum,
            )
            for in_dim, out_dim in itertools.pairwise((1, *self.h_dims))
        ]

    def _get_expansion(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Unflatten(1, (self.dim_codes, self.embedding_dim)),
            Transpose(2, 1),
            *self._get_conv_layers(),
            nn.Flatten(),
        )

    def _get_head(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(self.dropout),
            LinearLayer(in_dim, self.w_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum),
            nn.Dropout(self.dropout),
            LinearLayer(self.w_dim, out_dim, act_cls=None, batch_norm=False),
        )
