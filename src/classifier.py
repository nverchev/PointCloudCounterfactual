"""Module with the classifier architecture definition."""

import torch

from torch import nn
from torch.nn import functional as F

from src.config_options import Experiment
from src.data_structures import IN_CHAN, Inputs
from src.layers import EdgeConvLayer, LinearLayer, PointsConvLayer
from src.neighbour_ops import get_graph_features


class DGCNN(nn.Module):
    """Standard DGCNN classifier."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_class_ae = cfg.classifier.architecture
        self.k = cfg_class_ae.k
        self.act_cls = cfg_class_ae.act_cls
        self.h_dim = cfg_class_ae.hidden_dims
        self.emb_dim = cfg_class_ae.emb_dim
        self.mlp_dims = cfg_class_ae.mlp_dims
        self.dropout = cfg_class_ae.dropout
        self._classes = cfg_class_ae.out_classes
        self.num_classes = cfg.data.dataset.n_classes
        conv_modules: list[torch.nn.Module] = [EdgeConvLayer(2 * IN_CHAN, self.h_dim[0], act_cls=self.act_cls)]
        in_dims = self.h_dim[:-1]
        out_dims = self.h_dim[1:]
        for in_dim, out_dim in zip(in_dims, out_dims, strict=False):
            conv_modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls))
        self.edge_convs = nn.Sequential(*conv_modules)
        self.final_conv = PointsConvLayer(sum(self.h_dim), self.emb_dim)

        mlp_modules: list[torch.nn.Module] = [LinearLayer(2 * self.emb_dim, self.mlp_dims[0], act_cls=self.act_cls)]
        in_dims = self.mlp_dims[:-1]
        out_dims = self.mlp_dims[1:]
        for in_dim, out_dim, prob in zip(in_dims, out_dims, self.dropout, strict=False):
            mlp_modules.append(nn.Dropout(p=prob))
            mlp_modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
        mlp_modules.append(LinearLayer(self.mlp_dims[-1], self.num_classes, batch_norm=False))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, inputs: Inputs) -> torch.Tensor:
        """Forward Pass."""
        x = inputs.cloud
        indices = inputs.indices
        xs = []
        x = x.transpose(2, 1)
        for conv in self.edge_convs:
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
