"""Package for training classes and logic."""

from src.train.learning_schema import get_learning_schema
from src.train.metrics_and_losses import (
    get_classification_loss,
    get_w_autoencoder_loss,
    get_autoencoder_loss,
    get_diffusion_loss,
)

__all__ = [
    'get_autoencoder_loss',
    'get_classification_loss',
    'get_diffusion_loss',
    'get_learning_schema',
    'get_w_autoencoder_loss',
]
