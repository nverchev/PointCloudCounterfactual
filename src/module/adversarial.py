"""Module for training with an adversarial loss."""

import torch
from torch import nn

from src.config import Experiment
from src.module.layers import LinearLayer, GradReverse


class InformationEraser(nn.Module):
    """Predict logits but reverse the gradient to erase information with the target."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        self.z1_dim: int = cfg.autoencoder.model.z1_dim
        self.intermediate_dim: int = 512
        self.n_classes: int = cfg.data.dataset.n_classes
        self.mlp = nn.Sequential(
            LinearLayer(self.z1_dim, self.intermediate_dim, act_cls=nn.ReLU),
            LinearLayer(self.intermediate_dim, self.n_classes),
        )
        return

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        reversed_features = GradReverse.apply(features, 0.5)
        return self.mlp(reversed_features)
