"""Subpackage for training, evaluation and losses."""

from src.train.learning_schema import get_learning_schema
from src.train.loaders import get_loaders, get_evaluated_loaders
from src.train.metrics_and_losses import get_classification_loss, get_autoencoder_loss, get_flow_loss


__all__ = [
    'get_autoencoder_loss',
    'get_classification_loss',
    'get_evaluated_loaders',
    'get_flow_loss',
    'get_learning_schema',
    'get_loaders',
]
