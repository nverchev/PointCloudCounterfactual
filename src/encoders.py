"""Contains encoder classes."""

import abc
import itertools

import torch
import torch.nn as nn

from src.layers import PointsConvLayer, LinearLayer, EdgeConvLayer
from src.neighbour_ops import get_graph_features, graph_max_pooling
from src.config_options import Experiment, Encoders, WEncoders
from src.data_structures import IN_CHAN


class BaseWEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-space encoders in the autoencoder architecture.

    W-space encoders transform point cloud embeddings into a latent space
    representation. This base class provides common configurations and
    interface for different W-encoder implementations.
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder

        # Dataset parameters
        self.num_classes = cfg.data.dataset.n_classes

        # Model configuration
        cfg_model = cfg_ae.model
        cfg_w_encoder = cfg_model.encoder.w_encoder

        # Latent space dimensions
        self.w_dim = cfg_model.w_dim  # W-space dimension
        self.embedding_dim = cfg_model.embedding_dim  # Input embedding dimension
        self.z_dim = cfg_model.z1_dim  # Z-space dimension

        # Vector quantization parameters
        self.num_codes = cfg_model.n_codes  # Number of codebook vectors
        self.book_size = cfg_model.book_size  # Size of each codebook

        # Network architecture parameters
        self.h_dims_conv = cfg_w_encoder.hidden_dims_conv  # Conv layer dimensions
        self.h_dims_lin = cfg_w_encoder.hidden_dims_lin  # Linear layer dimensions
        self.dropout = cfg_w_encoder.dropout  # Dropout probabilities
        self.act_cls = cfg_w_encoder.act_cls  # Activation function class

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the W-encoder.

        Args:
            x: Input embedding tensor of shape [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Hidden features for hierarchical VAE
            torch.Tensor: Latent variable
        """


class WEncoderConvolution(BaseWEncoder):

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        total_h_dims = [h_dim * self.embedding_dim for h_dim in self.h_dims_conv]
        dim_pairs = itertools.pairwise([self.embedding_dim, *total_h_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim))
        self.conv = nn.Sequential(*modules)
        modules = []
        expand_w_dim = self.w_dim * self.h_dims_conv[-1]
        dim_pairs = itertools.pairwise([expand_w_dim, *self.h_dims_lin])
        for (in_dim, out_dim), do in zip(dim_pairs, self.dropout):
            modules.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls))
            modules.append(nn.Dropout(do))
        modules.append(
            LinearLayer(self.h_dims_lin[-1], 2 * self.z_dim, batch_norm=False)
        )  # change to encode
        self.encode = nn.Sequential(*modules)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = x.view(-1, self.embedding_dim, self.w_dim // self.embedding_dim)
        h = self.conv(x).view(-1, self.w_dim * self.h_dims_conv[-1])
        x = self.encode(h)
        return h, x


class BasePointEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        self.k = cfg_ae.model.encoder.k
        self.h_dim = cfg_ae.model.encoder.hidden_dims
        self.w_dim = cfg_ae.model.w_dim
        self.act_cls = cfg_ae.model.encoder.act_cls

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LDGCNN(BasePointEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.h_dim[0])
        modules: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise(self.h_dim):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls))
        self.points_convs = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.h_dim), self.w_dim, batch_norm=False)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
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
        self.h_dim = (64, 64, 128, 256)
        modules = [EdgeConvLayer(2 * IN_CHAN, self.h_dim[0])]
        for in_dim, out_dim in itertools.pairwise(self.h_dim):
            modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls))
        self.edge_convs = nn.Sequential(*modules)
        self.final_conv = nn.Conv1d(sum(self.h_dim), self.w_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
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
        x_max = x.max(dim=2, keepdim=False)[0]
        x = x_max
        return x


def get_encoder() -> BasePointEncoder:
    """Returns correct encoder."""
    dict_encoder: dict[Encoders, type[BasePointEncoder]] = {
        Encoders.LDGCNN: LDGCNN,
        Encoders.DGCNN: DGCNN,
    }
    return dict_encoder[Experiment.get_config().autoencoder.model.encoder.architecture]()


def get_w_encoder() -> BaseWEncoder:
    """Returns the correct w_encoder."""
    decoder_dict: dict[WEncoders, type[BaseWEncoder]] = {
        WEncoders.Convolution: WEncoderConvolution,
    }
    return decoder_dict[Experiment.get_config().autoencoder.model.encoder.w_encoder.architecture]()
