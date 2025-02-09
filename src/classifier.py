import torch
from torch import nn
from torch.nn import functional as F

from src.config_options import ExperimentClassifier, Datasets
from src.layers import EdgeConvLayer, PointsConvLayer, LinearLayer
from src.data_structures import IN_CHAN, Inputs
from src.neighbour_ops import get_graph_features


class DGCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = ExperimentClassifier.get_config()
        self.k = cfg.classifier.k
        self.act_cls = cfg.classifier.act_cls
        self.h_dim = cfg.classifier.hidden_dims
        self.emb_dim = cfg.classifier.emb_dim
        self.mlp_dims = cfg.classifier.mlp_dims
        self.dropout = cfg.classifier.dropout
        if cfg.data.dataset.name == Datasets.ModelNet:
            selected_classes = cfg.data.dataset.settings['select_classes']
            if selected_classes == ['All']:
                self.out_dim = 40
            else:
                self.out_dim = len(selected_classes)
        else:
            raise NotImplementedError
        conv_modules: list[torch.nn.Module] = [EdgeConvLayer(2 * IN_CHAN, self.h_dim[0], act_cls=self.act_cls)]
        in_dims = self.h_dim[:-1]
        out_dims = self.h_dim[1:]
        for in_dim, out_dim in zip(in_dims, out_dims):
            conv_modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls))
        self.edge_convs = nn.Sequential(*conv_modules)
        self.final_conv = PointsConvLayer(sum(self.h_dim), self.emb_dim)

        mlp_modules: list[torch.nn.Module] = [LinearLayer(2 * self.emb_dim, self.mlp_dims[0], act_cls=self.act_cls)]
        in_dims = self.mlp_dims[:-1]
        out_dims = self.mlp_dims[1:]
        for in_dim, out_dim, prob in zip(in_dims, out_dims, self.dropout):
            mlp_modules.append(nn.Dropout(p=prob))
            mlp_modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
        mlp_modules.append(LinearLayer(self.mlp_dims[-1], 40, batch_norm=False))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, inputs: Inputs) -> torch.Tensor:
        x = inputs.cloud
        indices = inputs.indices
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
        x1 = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x2 = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        x = torch.cat((x1, x2), 1)
        return self.mlp(x)
