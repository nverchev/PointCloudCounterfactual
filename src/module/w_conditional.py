"""Module that contains networks that are conditioned on the class probability"""

import abc

import torch
from torch import nn as nn

from src.config import Experiment, ActClass
from src.config.options import WConditionalEncoders
from src.module.layers import LinearLayer


class ConditionalPrior(nn.Module):
    """Network for the conditional prior"""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_wae_model = cfg.w_autoencoder.model
        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_model.n_codes
        self.z2_dim: int = cfg_wae_model.z2_dim
        self.prior = LinearLayer(self.n_classes, self.n_codes * 2 * self.z2_dim, grouped_norm=False)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.prior(x).view(-1, self.n_codes, 2 * self.z2_dim)


class BaseWConditionalEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_wae_model = cfg.w_autoencoder.model
        cfg_posterior = cfg_wae_model.conditional_w_encoder
        self.w_dim: int = cfg_ae_model.w_dim
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_codes: int = cfg_ae_model.n_codes
        self.z2_dim: int = cfg_wae_model.z2_dim
        self.mlp_dims: tuple[int, ...] = cfg_posterior.mlp_dims
        self.dropout: tuple[float, ...] = cfg_posterior.dropout_rates
        self.act_cls: ActClass = cfg_posterior.act_cls
        self.proj_dim: int = cfg_posterior.proj_dim
        self.n_heads: int = cfg_posterior.n_heads
        return

    @abc.abstractmethod
    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class TransformerWConditionalEncoder(BaseWConditionalEncoder):
    """Network for the difference in mean and log-var between the conditional prior and posterior."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = LinearLayer(self.embedding_dim, self.proj_dim, grouped_norm=False)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.prob_proj = LinearLayer(self.n_classes, self.proj_dim, grouped_norm=False)
        transformer_layers: list[nn.Module] = []
        for hidden_dim, do in zip(self.mlp_dims, self.dropout, strict=False):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.proj_dim,
                nhead=self.n_heads,
                dim_feedforward=hidden_dim,
                dropout=do,
                activation=self.act_cls(),
                batch_first=True,
                norm_first=True,
            )
            transformer_layers.append(encoder_layer)

        self.transformer = nn.ModuleList(transformer_layers)
        self.to_latent = LinearLayer(self.proj_dim, 2 * self.z2_dim, grouped_norm=False, soft_init=True)
        return

    def forward(self, probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        x = self.input_proj(x)
        x = self.positional_encoding.expand(batch_size, -1, -1) + x + self.prob_proj(probs).unsqueeze(1)
        for layer in self.transformer:
            x = layer(x)

        return self.to_latent(x)


def get_conditional_w_encoder() -> BaseWConditionalEncoder:
    """Get the conditional encoder according to the configuration."""
    conditional_dict: dict[WConditionalEncoders, type[BaseWConditionalEncoder]] = {
        WConditionalEncoders.Transformer: TransformerWConditionalEncoder,
    }
    return conditional_dict[Experiment.get_config().w_autoencoder.model.conditional_w_encoder.class_name]()
