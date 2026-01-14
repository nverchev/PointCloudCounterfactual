"""Package for architecture modules."""

from src.module.autoencoders import VQVAE, CounterfactualVQVAE, get_autoencoder
from src.module.decoders import get_decoder
from src.module.encoders import get_encoder
from src.module.classifier import DGCNN
from src.module.w_autoencoders import CounterfactualWAutoEncoder
from src.module.w_decoders import get_w_decoder
from src.module.w_encoders import get_w_encoder

__all__ = [
    'DGCNN',
    'VQVAE',
    'CounterfactualVQVAE',
    'CounterfactualWAutoEncoder',
    'get_autoencoder',
    'get_decoder',
    'get_encoder',
    'get_w_decoder',
    'get_w_encoder',
]
