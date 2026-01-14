"""Module package."""

from src.config.experiment import Experiment
from src.config.options import Encoders, Decoders, WEncoders, WDecoders, ModelHead
from src.module.autoencoder import (
    AutoEncoder,
    AE,
    AbstractVQVAE,
    VQVAE,
    CounterfactualVQVAE,
)
from src.module.w_autoencoder import BaseWAutoEncoder, WAutoEncoder, CounterfactualWAutoEncoder
from src.module.classifier import DGCNN
from src.module.decoders import (
    BasePointDecoder,
    PCGen,
    BaseWDecoder,
    WDecoderConvolution,
    WDecoderLinear,
    WDecoderTransformers,
    PriorDecoder,
    PosteriorDecoder,
)
from src.module.encoders import (
    BasePointEncoder,
    LDGCNN,
    BaseWEncoder,
    WEncoderConvolution,
    WEncoderTransformers,
    DGCNN as DGCNNEncoder,
)

__all__ = [
    'AE',
    'DGCNN',
    'LDGCNN',
    'VQVAE',
    'AbstractVQVAE',
    'AutoEncoder',
    'BasePointDecoder',
    'BasePointEncoder',
    'BaseWAutoEncoder',
    'BaseWDecoder',
    'BaseWEncoder',
    'CounterfactualVQVAE',
    'CounterfactualWAutoEncoder',
    'DGCNNEncoder',
    'PCGen',
    'PosteriorDecoder',
    'PriorDecoder',
    'WAutoEncoder',
    'WDecoderConvolution',
    'WDecoderLinear',
    'WDecoderTransformers',
    'WEncoderConvolution',
    'WEncoderTransformers',
    'get_autoencoder',
    'get_decoder',
    'get_encoder',
    'get_w_decoder',
    'get_w_encoder',
]


def get_autoencoder() -> AutoEncoder:
    """Factory function to create the appropriate autoencoder."""
    model_registry = {ModelHead.AE: AE, ModelHead.VQVAE: VQVAE, ModelHead.CounterfactualVQVAE: CounterfactualVQVAE}

    model_head = Experiment.get_config().autoencoder.architecture.head

    if model_head not in model_registry:
        raise ValueError(f'Unknown model head: {model_head}')

    return model_registry[model_head]()


def get_encoder() -> BasePointEncoder:
    """Returns correct encoder."""
    dict_encoder: dict[Encoders, type[BasePointEncoder]] = {
        Encoders.LDGCNN: LDGCNN,
        Encoders.DGCNN: DGCNNEncoder,
    }
    return dict_encoder[Experiment.get_config().autoencoder.architecture.encoder.architecture]()


def get_w_encoder() -> BaseWEncoder:
    """Returns the correct w_encoder."""
    decoder_dict: dict[WEncoders, type[BaseWEncoder]] = {
        WEncoders.Convolution: WEncoderConvolution,
        WEncoders.Transformers: WEncoderTransformers,
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.encoder.w_encoder.architecture]()


def get_decoder() -> BasePointDecoder:
    """Get decoder according to the configuration."""
    decoder_dict: dict[Decoders, BasePointDecoder] = {
        Decoders.PCGen: PCGen(),
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.decoder.architecture]


def get_w_decoder() -> BaseWDecoder:
    """Get W-decoder according to the configuration."""
    decoder_dict: dict[WDecoders, type[BaseWDecoder]] = {
        WDecoders.Convolution: WDecoderConvolution,
        WDecoders.Linear: WDecoderLinear,
        WDecoders.TransformerCross: WDecoderTransformers,
    }
    return decoder_dict[Experiment.get_config().autoencoder.architecture.decoder.w_decoder.architecture]()
