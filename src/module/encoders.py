"""Encoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn

from src.config import ActClass, Experiment, NormClass
from src.config.options import Encoders, LatentEncoders, ConditionalLatentEncoders
from src.data import IN_CHAN
from src.module.layers import (
    EdgeConvLayer,
    PointsConvLayer,
    LinearLayer,
    TransformerDecoder,
    PointsConvResBlock,
    TransformerEncoder as TransformerBlock,
)
from src.utils.neighbour_ops import get_graph_features, graph_max_pooling


class BasePointEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_neighbors: int = cfg_ae_model.encoder.n_neighbors
        self.conv_dims: tuple[int, ...] = cfg_ae_model.encoder.conv_dims
        self.n_transformer_layers: int = cfg_ae_model.encoder.n_transformer_layers
        self.feedforward_dim: int = cfg_ae_model.encoder.feedforward_dim
        self.transformer_dropout: float = cfg_ae_model.encoder.transformer_dropout
        self.act_cls: ActClass = cfg_ae_model.encoder.act_cls
        self.norm_cls: NormClass = cfg_ae_model.encoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class DGCNN(BasePointEncoder):
    """Dynamic Graph Convolutional Neural Network encoder."""

    def __init__(self) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        for in_dim, out_dim in itertools.pairwise((IN_CHAN, *self.conv_dims)):
            modules.append(EdgeConvLayer(2 * in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))

        self.edge_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), sum(self.conv_dims))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        xs = []
        x = x.transpose(2, 1)
        indices = torch.empty(0)
        for conv in self.edge_convolutions:
            indices, x = get_graph_features(x, n_neighbors=self.n_neighbors, indices=indices)
            indices = torch.empty(0)  # finds new neighbors dynamically every iteration
            x = conv(x)
            x = x.max(dim=3, keepdim=False)[0]  # [batch, features, num_points]
            xs.append(x)

        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class LDGCNN(BasePointEncoder):
    """Lighter version of DGCNN where the graph is only calculated once."""

    def __init__(self) -> None:
        super().__init__()
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.conv_dims[0], act_cls=self.act_cls, norm_cls=self.norm_cls)
        modules: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise(self.conv_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))

        self.points_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.conv_dims), sum(self.conv_dims))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(2, 1)
        indices, x = get_graph_features(x, n_neighbors=self.n_neighbors, indices=torch.empty(0))
        x = self.edge_conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        xs = [x]
        for conv in self.points_convolutions:
            x = graph_max_pooling(x, n_neighbors=self.n_neighbors, indices=indices)
            x = conv(x)
            xs.append(x)

        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class TransformerEncoder(BasePointEncoder):
    """Transformer encoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.conv_dims[0], act_cls=self.act_cls, norm_cls=self.norm_cls)
        modules: list[nn.Module] = []
        for in_dim, out_dim in itertools.pairwise(self.conv_dims):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))

        self.points_convolutions = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(
            sum(self.conv_dims), sum(self.conv_dims), act_cls=self.act_cls, norm_cls=self.norm_cls
        )
        self.n_codes: int = cfg_ae_model.n_codes
        self.n_heads: int = cfg_ae_model.encoder.n_heads
        self.proj_dim: int = cfg_ae_model.proj_dim
        self.proj_codes = LinearLayer(self.proj_dim, self.proj_dim, act_cls=self.act_cls)
        self.proj_input = LinearLayer(IN_CHAN, self.proj_dim, act_cls=self.act_cls)
        self.transformer_codes = TransformerDecoder(
            embedding_dim=self.proj_dim,
            n_heads=self.n_heads,
            feedforward_dim=self.feedforward_dim,
            dropout_rate=self.transformer_dropout,
            act_cls=self.act_cls,
            n_layers=self.n_transformer_layers,
            use_final_norm=True,
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch = x.shape[0]
        y = x.transpose(2, 1)
        indices, y = get_graph_features(y, n_neighbors=self.n_neighbors, indices=torch.empty(0))
        y = self.edge_conv(y)
        y = y.max(dim=3, keepdim=False)[0]
        xs = [y]
        for conv in self.points_convolutions:
            y = graph_max_pooling(y, n_neighbors=self.n_neighbors, indices=indices)
            y = conv(y)
            xs.append(y)

        y = torch.cat(xs, dim=1).contiguous()
        y = self.final_conv(y)
        x_max = y.max(dim=2, keepdim=False)[0]
        queries = self.proj_codes(x_max.view(batch, self.n_codes, -1))
        memory = self.proj_input(x)
        x = self.transformer_codes(queries, memory)
        return x.view(batch, -1)


def get_encoder() -> BasePointEncoder:
    """Get encoder according to the configuration."""
    dict_encoder: dict[Encoders, type[BasePointEncoder]] = {
        Encoders.LDGCNN: LDGCNN,
        Encoders.DGCNN: DGCNN,
        Encoders.Transformer: TransformerEncoder,
    }
    return dict_encoder[Experiment.get_config().autoencoder.model.encoder.class_name]()


class BaseWEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-space encoders in the autoencoder architecture."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder

        # Dataset parameters
        self.n_classes = cfg.data.dataset.n_classes

        # Model configuration
        cfg_ae_model = cfg_ae.model
        cfg_w_encoder = cfg_ae_model.latent_encoder

        # Latent space dimensions
        self.input_dim: int = cfg_ae_model.proj_dim
        self.z1_dim: int = cfg_ae_model.z1_dim
        self.n_codes: int = cfg_ae_model.n_codes

        # Network architecture parameters
        self.proj_dim: int = cfg_ae_model.proj_dim
        self.n_heads: int = cfg_w_encoder.n_heads
        self.conv_dims: tuple[int, ...] = cfg_w_encoder.conv_dims
        self.n_transformer_layers: int = cfg_w_encoder.n_transformer_layers
        self.feedforward_dim: int = cfg_w_encoder.feedforward_dim
        self.transformer_dropout: float = cfg_w_encoder.transformer_dropout
        self.act_cls: ActClass = cfg_w_encoder.act_cls
        self.norm_cls: NormClass = cfg_w_encoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""


class ConvolutionalWEncoder(BaseWEncoder):
    """W-space encoder that uses a convolutional architecture with 1x1 kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            PointsConvResBlock([self.input_dim, *self.conv_dims], act_cls=self.act_cls, norm_cls=self.norm_cls),
            PointsConvLayer(self.conv_dims[-1], 2 * self.z1_dim, use_trunc_init=True),
        )
        return

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        # x is [Batch, n_codes * input_dim]
        # Reshape to [Batch, input_dim, n_codes] for convolution
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_codes, self.input_dim).transpose(2, 1)
        x = self.encode(x).transpose(2, 1)
        # Output [Batch, n_codes, 2*z1_dim] -> Flatten?
        return x.reshape(batch_size, -1)


class TransformerWEncoder(BaseWEncoder):
    """W-space encoder using transformer architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = LinearLayer(self.input_dim, self.proj_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.transformer = TransformerBlock(
            embedding_dim=self.proj_dim,
            n_heads=self.n_heads,
            feedforward_dim=self.feedforward_dim,
            act_cls=self.act_cls,
            dropout_rate=self.transformer_dropout,
            n_layers=self.n_transformer_layers,
        )
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z1_dim, use_trunc_init=True)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer encoder."""
        batch_size = x.shape[0]
        # x is [Batch, n_codes * input_dim]
        x = x.view(batch_size, self.n_codes, self.input_dim)
        x = self.input_proj(x)
        x = self.positional_encoding.expand(batch_size, -1, -1) + x
        x = self.transformer(x)
        # x is [Batch, n_codes, proj_dim]. Project to latent params [Batch, n_tokens, 2*z1_dim]
        return self.to_latent(x).reshape(batch_size, -1)


def get_latent_encoder() -> BaseWEncoder:
    """Returns the correct w_encoder."""
    decoder_dict: dict[LatentEncoders, type[BaseWEncoder]] = {
        LatentEncoders.Convolutional: ConvolutionalWEncoder,
        LatentEncoders.Transformer: TransformerWEncoder,
    }
    return decoder_dict[Experiment.get_config().autoencoder.model.latent_encoder.class_name]()


class BaseWConditionalEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_posterior = cfg_ae_model.conditional_latent_encoder

        # Input dim from PointEncoder
        self.input_dim: int = cfg_ae_model.proj_dim

        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_model.n_codes
        self.z2_dim: int = cfg_ae_model.z2_dim
        self.n_transformer_layers: int = cfg_posterior.n_transformer_layers
        self.feedforward_dim: int = cfg_posterior.feedforward_dim
        self.transformer_dropout: float = cfg_posterior.transformer_dropout
        self.act_cls: ActClass = cfg_posterior.act_cls
        self.proj_dim: int = cfg_ae_model.proj_dim
        self.n_heads: int = cfg_posterior.n_heads
        return

    @abc.abstractmethod
    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class TransformerWConditionalEncoder(BaseWConditionalEncoder):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = LinearLayer(self.input_dim, self.proj_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.prob_proj = LinearLayer(self.n_classes, self.proj_dim)
        self.transformer = TransformerBlock(
            embedding_dim=self.proj_dim,
            n_heads=self.n_heads,
            feedforward_dim=self.feedforward_dim,
            dropout_rate=self.transformer_dropout,
            act_cls=self.act_cls,
            n_layers=self.n_transformer_layers,
        )
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z2_dim, use_trunc_init=True)
        return

    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        # x is [Batch, -1]
        x = x.view(batch_size, self.n_codes, self.input_dim)
        x = self.input_proj(x)
        # Condition on probs (global)
        # Add to every token?
        x = self.positional_encoding.expand(batch_size, -1, -1) + x + self.prob_proj(probs).unsqueeze(1)
        x = self.transformer(x)
        return self.to_latent(x).reshape(batch_size, -1)


def get_conditional_latent_encoder() -> BaseWConditionalEncoder:
    """Get the conditional encoder according to the configuration."""
    conditional_dict: dict[ConditionalLatentEncoders, type[BaseWConditionalEncoder]] = {
        ConditionalLatentEncoders.Transformer: TransformerWConditionalEncoder,
    }
    return conditional_dict[Experiment.get_config().autoencoder.model.conditional_latent_encoder.class_name]()


class ConditionalPrior(nn.Module):
    """Network for the conditional prior"""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_model.n_codes
        self.z2_dim: int = cfg_ae_model.z2_dim
        # Predict one z2 per token?
        self.prior = LinearLayer(self.n_classes, self.n_codes * 2 * self.z2_dim)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Output [Batch, n_codes * 2 * z2_dim]
        return self.prior(x).view(-1, self.n_codes * 2 * self.z2_dim)
