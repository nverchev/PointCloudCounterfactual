"""Module that contains networks that are conditioned on the class probability"""

import abc

import torch
from torch import nn as nn

from src.config import Experiment, ActClass
from src.config.options import WConditionalEncoders
from src.module.layers import LinearLayer, TransformerEncoder


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
        self.prior = LinearLayer(self.n_classes, self.n_codes * 2 * self.z2_dim)
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
        self.n_transformer_layers: int = cfg_posterior.n_transformer_layers
        self.feedforward_dim: int = cfg_posterior.feedforward_dim
        self.transformer_dropout: float = cfg_posterior.transformer_dropout
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
        self.input_proj = LinearLayer(self.embedding_dim, self.proj_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.n_codes, self.proj_dim))
        self.prob_proj = LinearLayer(self.n_classes, self.proj_dim)
        self.transformer = TransformerEncoder(
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
        x = self.input_proj(x)
        x = self.positional_encoding.expand(batch_size, -1, -1) + x + self.prob_proj(probs).unsqueeze(1)
        x = self.transformer(x)
        return self.to_latent(x)


def get_conditional_w_encoder() -> BaseWConditionalEncoder:
    """Get the conditional encoder according to the configuration."""
    conditional_dict: dict[WConditionalEncoders, type[BaseWConditionalEncoder]] = {
        WConditionalEncoders.Transformer: TransformerWConditionalEncoder,
    }
    return conditional_dict[Experiment.get_config().w_autoencoder.model.conditional_w_encoder.class_name]()
