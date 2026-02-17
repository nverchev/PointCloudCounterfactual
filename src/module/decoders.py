"""Decoder architecture."""

import abc
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ActClass, NormClass
from src.config.experiment import Experiment
from src.config.options import Decoders, LatentDecoders
from src.data import OUT_CHAN
from src.module.layers import PointsConvLayer, LinearLayer, TransformerDecoder, PointsConvResBlock
from src.utils.neighbour_ops import graph_filtering


class BasePointDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for point decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        cfg_decoder = cfg_ae_model.decoder
        self.map_dims: tuple[int, ...] = cfg_decoder.map_dims
        self.conv_dims: tuple[int, ...] = cfg_decoder.conv_dims
        self.n_heads: int = cfg_decoder.n_heads
        self.feedforward_dim: int = cfg_decoder.feedforward_dim
        self.n_transformer_layers: int = cfg_decoder.n_transformer_layers
        self.transformer_dropout: float = cfg_decoder.transformer_dropout
        self.n_codes: int = cfg_ae_model.n_codes
        self.proj_dim: int = cfg_ae_model.proj_dim
        self.act_cls: ActClass = cfg_decoder.act_cls
        self.norm_cls: NormClass = cfg_decoder.norm_cls
        self.filtering: bool = cfg_decoder.filter
        self.sample_dim: int = cfg_decoder.sample_dim
        self.n_components: int = cfg_decoder.n_components
        self.tau: float = cfg_decoder.tau

    @abc.abstractmethod
    def forward(self, z: torch.Tensor, n_output_points: int, initial_sampling: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class PCGen(BasePointDecoder):
    """Map points from a fixed distribution to a point cloud in parallel."""

    _null_tensor: torch.Tensor = torch.empty(0)

    def __init__(self) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        dim_pairs = itertools.pairwise([self.sample_dim, *self.map_dims])
        for in_dim, out_dim in dim_pairs:
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=torch.nn.ReLU))

        # w_dim is determined by the last map_dim for the join operation
        self.w_dim = self.map_dims[-1]
        modules.append(PointsConvLayer(self.map_dims[-1], self.w_dim, act_cls=nn.Hardtanh))
        self.map_sample = nn.Sequential(*modules)

        self.memory_positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))

        # Projection for the join operation: from flat latent (n_codes * proj_dim) to w_dim
        self.proj_to_w = LinearLayer(self.n_codes * self.proj_dim, self.w_dim)

        # Identity projection for memory if dimensions match, or linear if not
        self.proj_memory = nn.Identity()
        self.group_conv = nn.ModuleList()
        self.group_transformer = nn.ModuleList()
        self.group_final = nn.ModuleList()
        for _ in range(self.n_components):
            block = PointsConvResBlock(
                [self.w_dim, *self.conv_dims, self.proj_dim],
                act_cls=self.act_cls,
                norm_cls=self.norm_cls,
            )
            self.group_conv.append(block)
            transformer_decoder = TransformerDecoder(
                embedding_dim=self.proj_dim,
                n_heads=self.n_heads,
                feedforward_dim=self.feedforward_dim,
                dropout_rate=self.transformer_dropout,
                act_cls=self.act_cls,
                n_layers=self.n_transformer_layers,
                use_final_norm=True,
            )
            self.group_transformer.append(transformer_decoder)
            self.group_final.append(PointsConvLayer(self.proj_dim, OUT_CHAN, use_trunc_init=True))

        if self.n_components > 1:
            self.att = PointsConvLayer(self.proj_dim * self.n_components, self.n_components)

        return

    def _initialize_sampling(
        self, batch: int, n_output_points: int, device: torch.device, initial_sampling: torch.Tensor
    ) -> torch.Tensor:
        """Initialize and normalize the sampling points."""
        if initial_sampling.numel():
            return initial_sampling.to(device)

        return torch.randn(batch, self.sample_dim, n_output_points, device=device)

    def _process_component_groups(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process all component groups and return outputs and attention features."""
        xs_list = []
        group_atts = []

        for group in range(self.n_components):
            x_group = self.group_conv[group](x)
            # x_group = self.group_proj[group](x_group)
            group_atts.append(x_group)
            x_group = x_group.transpose(2, 1)
            x_group = self.group_transformer[group](x_group, memory=memory)
            x_group = x_group.transpose(2, 1)
            x_group = self.group_final[group](x_group)
            xs_list.append(x_group)

        xs = torch.stack(xs_list, dim=3)
        return xs, group_atts

    def _apply_attention_mixing(self, xs: torch.Tensor, group_atts: list[torch.Tensor]) -> torch.Tensor:
        """Apply attention-based mixing across components if there are more than one, otherwise return the component."""
        if self.n_components > 1:
            x_att = self.att(torch.cat(group_atts, dim=1).contiguous())
            if self.training:
                x_att = F.gumbel_softmax(x_att, tau=self.tau, dim=1)
            else:
                x_att = torch.softmax(x_att / self.tau, dim=1)

            x_att = x_att.transpose(2, 1)
            x = (xs * x_att.unsqueeze(1)).sum(3)
        else:
            x = xs.squeeze(3)

        return x

    def forward(self, z: torch.Tensor, n_output_points: int, initial_sampling: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent vector [Batch, n_codes * proj_dim]
            n_output_points: Number of output points
            initial_sampling: Initial sampling points
        """
        batch = z.size()[0]
        device = z.device

        # Initialize sampling points
        x = self._initialize_sampling(batch, n_output_points, device, initial_sampling)

        # Map and join with latent code (using projected w)
        w = self.proj_to_w(z)
        x = self.map_sample(x)
        x = self._join_operation(x, w)

        # Process component groups
        # Reshape z to [Batch, n_codes, proj_dim] for memory
        z_seq = z.view(batch, self.n_codes, self.proj_dim)
        memory = self.proj_memory(z_seq) + self.memory_positional_encoding
        xs, group_atts = self._process_component_groups(x, memory)

        # Mix components with attention
        x = self._apply_attention_mixing(xs, group_atts)

        # Optional graph filtering
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
    return decoder_dict[Experiment.get_config().autoencoder.model.decoder.class_name]


class BaseWDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W-decoder."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoder
        cfg_ae_model = cfg_ae.model
        cfg_w_decoder = cfg_ae_model.latent_decoder

        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_model.n_codes

        self.z1_dim: int = cfg_ae_model.z1_dim
        self.z2_dim: int = cfg_ae_model.z2_dim

        self.proj_dim: int = cfg_ae_model.proj_dim
        self.n_heads: int = cfg_w_decoder.n_heads
        self.conv_dims: tuple[int, ...] = cfg_w_decoder.conv_dims
        self.n_transformer_layers: int = cfg_w_decoder.n_transformer_layers
        self.feedforward_dim: int = cfg_w_decoder.feedforward_dim
        self.transformer_dropout: float = cfg_w_decoder.transformer_dropout
        self.act_cls: ActClass = cfg_w_decoder.act_cls
        self.norm_cls: NormClass = cfg_w_decoder.norm_cls
        return

    @abc.abstractmethod
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class LinearWDecoder(BaseWDecoder):
    """W-decoder using grouped linear layers (PointConv for a sequence of 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            PointsConvResBlock(
                [(self.z1_dim + self.z2_dim) * self.n_codes, *self.conv_dims],
                n_groups_layer=self.n_codes,
                act_cls=self.act_cls,
                norm_cls=self.norm_cls,
            ),
            PointsConvLayer(self.conv_dims[-1], self.proj_dim * self.n_codes, n_groups_layer=self.n_codes),
        )
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # z1, z2 are [Batch, n_codes * z_dim]
        # Reshape to [Batch, n_codes * z, 1]?
        # PointsConvResBlock expects [Batch, Channels, Points].
        # If n_groups_layer=n_codes, input channels must be divisible by n_codes?
        # (z1_dim + z2_dim) * n_codes is divisible.
        # But we need to structure it correctly.
        # Cat z1, z2 per token?
        # Reshape [Batch, n_codes, z1_dim]
        batch_size = z1.shape[0]
        z1 = z1.view(batch_size, self.n_codes, self.z1_dim)
        z2 = z2.view(batch_size, self.n_codes, self.z2_dim)
        z = torch.cat((z1, z2), dim=2)  # [Batch, n_codes, z1+z2]
        z = z.view(batch_size, -1, 1)  # [Batch, n_codes * (z1+z2), 1]

        x = self.decode(z)
        # Output [Batch, n_codes * proj_dim, 1] -> Squeeze -> [Batch, n_codes * proj_dim]
        return x.squeeze(2)


class TransformerWDecoder(BaseWDecoder):
    """W-decoder using transformer architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.z1_proj = LinearLayer(self.z1_dim, self.proj_dim)
        self.z2_proj = LinearLayer(self.z2_dim, self.proj_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.memory_positional_embedding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.transformer = TransformerDecoder(
            embedding_dim=self.proj_dim,
            n_heads=self.n_heads,
            feedforward_dim=self.feedforward_dim,
            dropout_rate=self.transformer_dropout,
            act_cls=self.act_cls,
            n_layers=self.n_transformer_layers,
            use_final_norm=True,
        )
        self.compress = LinearLayer(self.proj_dim, self.proj_dim)
        return

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        # Reshape to [Batch, n_codes, z_dim]
        z1 = z1.view(batch_size, self.n_codes, self.z1_dim)
        z2 = z2.view(batch_size, self.n_codes, self.z2_dim)

        z1_proj = self.z1_proj(z1)  # [Batch, n_codes, proj_dim]
        z2_proj = self.z2_proj(z2)

        memory = z1_proj + self.memory_positional_embedding.expand(batch_size, -1, -1)
        x = z2_proj + self.positional_embedding.expand(batch_size, -1, -1)
        x = self.transformer(x, memory)
        x = self.compress(x)
        return x.view(batch_size, -1)


def get_latent_decoder() -> BaseWDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[LatentDecoders, type[BaseWDecoder]] = {
        LatentDecoders.Linear: LinearWDecoder,
        LatentDecoders.Transformer: TransformerWDecoder,
    }
    return decoder_dict[Experiment.get_config().autoencoder.model.latent_decoder.class_name]()
