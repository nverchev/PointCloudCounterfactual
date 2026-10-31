"""Classifier under audit."""

import abc
import itertools

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.torch import ActClass
from src.data.structures import IN_CHAN, Inputs
from src.module.layers import EdgeConvLayer, EdgeFeatures, LinearLayer, Max, Mean, PointsConvLayer, Transpose

__all__ = [
    'DGCNN',
    'BaseClassifier',
]


class BaseClassifier(nn.Module, abc.ABC):
    """Abstract base of the classifiers under audit.

    Attributes:
        n_classes: Number of classes to predict.
        n_neighbours: Neighbours of every point in the edge convolutions.
        hidden_dims: Channels of the edge convolutions.
        embedding_dim: Channels of the point features before the global pooling.
        mlp_dims: Hidden dimensions of the classification head.
        dropout: Dropout probability after the first layer of the classification head.
        act_cls: Activation of the layers.
        bn_momentum: Momentum of the batch normalization.
    """

    n_classes: Final[int]
    n_neighbours: Final[int]
    hidden_dims: Final[tuple[int, ...]]
    embedding_dim: Final[int]
    mlp_dims: Final[tuple[int, ...]]
    dropout: Final[float]
    act_cls: Final[ActClass]
    bn_momentum: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_arch = cfg.classification.model.architecture
        self.n_classes = cfg.data.dataset.n_classes
        self.n_neighbours = cfg_arch.n_neighbours
        self.hidden_dims = cfg_arch.hidden_dims
        self.embedding_dim = cfg_arch.embedding_dim
        self.mlp_dims = cfg_arch.mlp_dims
        self.dropout = cfg_arch.dropout
        self.act_cls = cfg_arch.act_cls
        self.bn_momentum = cfg_arch.bn_momentum
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> torch.Tensor:
        """Return the logits of the clouds ``(B, n_classes)``."""


class DGCNN(BaseClassifier):
    """Dynamic Graph CNN classifier.

    Attributes:
        transpose: Moves the coordinates to the channel dimension.
        blocks: Edge convolutions, each on the graph of the features of the block before.
        final_conv: Layer shared by the points before the global pooling.
        max_pool: Global max pooling.
        mean_pool: Global average pooling.
        mlp: Classification head.
    """

    transpose: Transpose
    blocks: nn.ModuleList
    final_conv: PointsConvLayer
    max_pool: Max
    mean_pool: Mean
    mlp: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.transpose = Transpose(2, 1)
        self.blocks = self._get_blocks()
        self.final_conv = PointsConvLayer(
            sum(self.hidden_dims), self.embedding_dim, act_cls=None, bn_momentum=self.bn_momentum
        )
        self.max_pool = Max(dim=2)
        self.mean_pool = Mean(dim=2)
        self.mlp = self._get_mlp()
        return

    def _get_blocks(self) -> nn.ModuleList:
        return nn.ModuleList(
            nn.Sequential(
                EdgeFeatures(self.n_neighbours),
                EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum),
                Max(dim=3),
            )
            for in_dim, out_dim in itertools.pairwise((IN_CHAN, *self.hidden_dims))
        )

    def _get_mlp(self) -> nn.Sequential:
        first_dim, *other_dims = self.mlp_dims
        layers: list[nn.Module] = [
            LinearLayer(2 * self.embedding_dim, first_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum)
        ]
        for in_dim, out_dim in itertools.pairwise((first_dim, *other_dims)):
            layers.append(nn.Dropout(self.dropout))
            layers.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum))

        layers.append(LinearLayer(self.mlp_dims[-1], self.n_classes, act_cls=None, batch_norm=False))
        return nn.Sequential(*layers)

    @override
    def forward(self, inputs: Inputs) -> torch.Tensor:
        """Return the logits of the clouds."""
        x = self.transpose(inputs.cloud)
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        x = self.final_conv(torch.cat(features, dim=1).contiguous())
        return self.mlp(torch.cat((self.max_pool(x), self.mean_pool(x)), dim=1))
