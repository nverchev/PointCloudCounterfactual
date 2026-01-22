"""Classifier architecture."""

import itertools

import torch

from torch import nn
from torch.nn import functional as F

from src.config.experiment import Experiment
from src.config.options import Classifiers
from src.data import IN_CHAN
from src.data.structures import Inputs
from src.module.layers import EdgeConvLayer, LinearLayer, PointsConvLayer
from src.utils.neighbour_ops import get_graph_features


class DGCNN(nn.Module):
    """Standard Dynamic Graph Convolutional Neural Network classifier."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_class_model = cfg.classifier.model
        self.k: int = cfg_class_model.n_neighbors
        self.act_cls = cfg_class_model.act_cls
        self.conv_dims: tuple[int, ...] = cfg_class_model.conv_dims
        self.feature_dim: int = cfg_class_model.feature_dim
        self.mlp_dims: tuple[int, ...] = cfg_class_model.mlp_dims
        self.dropout_rates: tuple[float, ...] = cfg_class_model.dropout_rates
        self.n_classes: int = cfg.data.dataset.n_classes
        conv_modules: list[torch.nn.Module] = [EdgeConvLayer(2 * IN_CHAN, self.conv_dims[0], act_cls=self.act_cls)]
        for in_dim, out_dim in itertools.pairwise(self.conv_dims):
            conv_modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls))

        self.edge_convolutions = nn.Sequential(*conv_modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), self.feature_dim)
        mlp_modules: list[torch.nn.Module] = [LinearLayer(2 * self.feature_dim, self.mlp_dims[0], act_cls=self.act_cls)]

        for (in_dim, out_dim), rate in zip(itertools.pairwise(self.mlp_dims), self.dropout_rates, strict=False):
            mlp_modules.append(nn.Dropout(p=rate))
            mlp_modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))

        mlp_modules.append(LinearLayer(self.mlp_dims[-1], self.n_classes, batch_norm=False))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, inputs: Inputs) -> torch.Tensor:
        """Forward Pass."""
        x = inputs.cloud
        indices = inputs.indices
        xs = []
        x = x.transpose(2, 1)
        for conv in self.edge_convolutions:
            indices, x = get_graph_features(x, k=self.k, indices=indices)  # [batch, features, num_points, k]
            indices = torch.empty(0)  # finds new neighbors dynamically every iteration
            x = conv(x)
            x = x.max(dim=3, keepdim=False)[0]  # [batch, features, num_points]
            xs.append(x)

        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x1 = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x2 = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        x = torch.cat((x1, x2), 1)
        return self.mlp(x)


def get_classifier() -> DGCNN:
    """Get the classifier according to the configuration."""
    dict_classifier: dict[str, type[DGCNN]] = {Classifiers.DGCNN: DGCNN}
    return dict_classifier[Experiment.get_config().classifier.model.class_name]()
