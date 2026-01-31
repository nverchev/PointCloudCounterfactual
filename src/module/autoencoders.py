"""Autoencoder architecture."""

import abc

from typing import Generic, TypeVar, override

import torch
import torch.nn as nn

from src.config import Experiment
from src.config.options import AutoEncoders
from src.data.structures import Inputs, Outputs, WInputs
from src.module.decoders import get_decoder
from src.module.encoders import get_encoder
from src.module.quantize import VectorQuantizer
from src.module.w_autoencoders import BaseWAutoEncoder, WAutoEncoder, CounterfactualWAutoEncoder
from src.module.layers import TransferGrad

WA = TypeVar('WA', bound=BaseWAutoEncoder, covariant=True)


class AbstractAutoEncoder(nn.Module, abc.ABC):
    """Abstract autoencoder for point clouds."""

    def __init__(self):
        super().__init__()
        cfg_ae = Experiment.get_config().autoencoder
        self.n_training_points_training: int = cfg_ae.n_training_output_points
        self.n_inference_output_points: int = cfg_ae.objective.n_inference_output_points
        return

    @property
    def n_output_points(self) -> int:
        """Number of generated points."""
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""


class Oracle(AbstractAutoEncoder):
    """Oracle autoencoder that returns an input subset."""

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        out = Outputs()
        out.recon = inputs.cloud[:, : self.n_output_points, :]
        return out


class BaseAutoencoder(AbstractAutoEncoder):
    """Standard autoencoder for point clouds."""

    def __init__(self):
        super().__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        out = self.encode(inputs)
        return self.decode(out, inputs)

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode point cloud to latent representation."""
        out = Outputs()
        return out

    def decode(self, out: Outputs, inputs: Inputs) -> Outputs:
        """Decode latent representation to point cloud."""
        x = self.decoder(out.word, self.n_output_points, inputs.initial_sampling)
        out.recon = x.transpose(2, 1).contiguous()
        return out


class BaseVQVAE(BaseAutoencoder, abc.ABC, Generic[WA]):
    """Abstract VQVAE with vector quantization."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)
    codebook: torch.Tensor

    def __init__(self):
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        self.n_codes: int = cfg_ae_model.n_codes
        self.book_size: int = cfg_ae_model.book_size
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        codebook = nn.functional.normalize(torch.randn(self.n_codes, self.book_size, self.embedding_dim), dim=1)
        self.register_buffer('codebook', codebook)
        self.w_autoencoder: WA = self._init_w_autoencoder()
        for param in self.w_autoencoder.parameters():
            param.requires_grad = False  # separate training for W autoencoder

        self.quantizer = VectorQuantizer()
        self.transfer = TransferGrad()
        return

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode with optional double encoding."""
        out = Outputs()
        out.word_approx = self.encoder(inputs.cloud)
        return out

    def decode(self, out: Outputs, inputs: Inputs) -> Outputs:
        """Decode with vector quantization."""
        with torch.no_grad():
            out.word_quantised, out.idx, _ = self.quantizer.quantize(out.word_approx, self.codebook)
            if self.training:
                self.quantizer.update_codebook(self.codebook, out.idx, out.word_approx)

            out.one_hot_idx = self.quantizer.create_one_hot(out.idx)

        out.word = self.transfer.apply(out.word_quantised, out.word_approx)
        return super().decode(out, inputs)

    @abc.abstractmethod
    def _init_w_autoencoder(self) -> WA:
        """Create the appropriate W autoencoder."""

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
        probs: torch.Tensor | None = None,
    ) -> Outputs:
        """Generate samples from the model."""
        self.w_autoencoder.update_codebook(self.codebook)
        initial_sampling = initial_sampling.to(self.codebook.device)
        inputs = Inputs(self._null_tensor, initial_sampling)
        z1_bias = z1_bias.to(self.codebook.device)
        out = self.w_autoencoder.generate_discrete_latent_space(z1_bias, batch_size=batch_size, probs=probs)
        out.word_quantised = out.word = self.quantizer.decode_from_indices(out.idx, self.codebook)
        return BaseAutoencoder.decode(self, out, inputs)


class VQVAE(BaseVQVAE[WAutoEncoder]):
    """Standard VQVAE implementation."""

    def double_reconstruct(self, inputs: Inputs) -> Outputs:
        """Reconstruct from the continuous space."""
        self.w_autoencoder.update_codebook(self.codebook)
        w_q = self.encode(inputs).word_approx.view(-1, self.n_codes, self.embedding_dim)
        out = self.w_autoencoder(WInputs(w_q), stochastic=False)
        out.word_quantised = out.word = self.quantizer.decode_from_indices(out.idx, self.codebook)
        return BaseAutoencoder.decode(self, out, inputs)

    @override
    def _init_w_autoencoder(self) -> WAutoEncoder:
        return WAutoEncoder()


class CounterfactualVQVAE(BaseVQVAE[CounterfactualWAutoEncoder]):
    """Counterfactual VQVAE implementation."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def double_reconstruct_with_logits(self, inputs: Inputs, logits: torch.Tensor) -> Outputs:
        """Reconstruct from the continuous space."""
        self.w_autoencoder.update_codebook(self.codebook)
        w_q = self.encode(inputs).word_approx
        out = self.w_autoencoder(WInputs(w_q, logits), stochastic=False)
        out.word_quantised = out.word = self.quantizer.decode_from_indices(out.idx, self.codebook)
        return BaseAutoencoder.decode(self, out, inputs)

    @torch.inference_mode()
    def generate_counterfactual(
        self,
        inputs: Inputs,
        sample_logits: torch.Tensor,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Generate counterfactual samples."""
        self.w_autoencoder.update_codebook(self.codebook)
        w_q = self.encode(inputs).word_approx
        out = self.w_autoencoder.generate_counterfactual(WInputs(w_q, sample_logits), target_dim, target_value)
        out.word_quantised = out.word = self.quantizer.decode_from_indices(out.idx, self.codebook)
        return BaseAutoencoder.decode(self, out, inputs)

    @override
    def _init_w_autoencoder(self) -> CounterfactualWAutoEncoder:
        return CounterfactualWAutoEncoder()


def get_autoencoder() -> AbstractAutoEncoder:
    """Get the correct autoencoder according to the configuration."""
    model_registry = {
        AutoEncoders.AE: BaseAutoencoder,
        AutoEncoders.VQVAE: VQVAE,
        AutoEncoders.CounterfactualVQVAE: CounterfactualVQVAE,
    }
    return model_registry[Experiment.get_config().autoencoder.model.class_name]()
