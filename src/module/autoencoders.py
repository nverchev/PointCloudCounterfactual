"""Autoencoder architecture."""

import abc

from typing import Generic, TypeVar

import torch
import torch.nn as nn

from src.config import Experiment
from src.config.options import ModelHead
from src.data.structures import Inputs, Outputs
from src.module.decoders import get_decoder
from src.module.encoders import get_encoder
from src.module.quantize import VectorQuantizer
from src.module.w_autoencoders import BaseWAutoEncoder, WAutoEncoder, CounterfactualWAutoEncoder
from src.module.layers import TransferGrad, reset_child_params

WA = TypeVar('WA', bound=BaseWAutoEncoder, covariant=True)


class AbstractAutoEncoder(nn.Module, abc.ABC):
    """Abstract autoencoder for point clouds."""

    def __init__(self):
        super().__init__()
        cfg_ae = Experiment.get_config().autoencoder
        self.m_training: int = cfg_ae.architecture.training_output_points
        self.m_test: int = cfg_ae.objective.n_inference_output_points

    @property
    def m(self) -> int:
        """Number of generated points."""
        return self.m_test if torch.is_inference_mode_enabled() else self.m_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        pass

    def recursive_reset_parameters(self) -> None:
        """Reset all parameters."""
        reset_child_params(self)
        return


class Oracle(AbstractAutoEncoder):
    """Oracle autoencoder that returns an input subset."""

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        data = Outputs()
        data.recon = inputs.cloud[:, : self.m, :]
        return data


class BaseAutoencoder(AbstractAutoEncoder):
    """Standard autoencoder for point clouds."""

    def __init__(self):
        super().__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        data = self.encode(inputs)
        return self.decode(data, inputs)

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode point cloud to latent representation."""
        data = Outputs()
        return data

    def decode(self, data: Outputs, inputs: Inputs) -> Outputs:
        """Decode latent representation to point cloud."""
        x = self.decoder(data.w, self.m, inputs.initial_sampling)
        data.recon = x.transpose(2, 1).contiguous()
        return data


class BaseVQVAE(BaseAutoencoder, abc.ABC, Generic[WA]):
    """Abstract VQVAE with vector quantization."""

    def __init__(self):
        super().__init__()
        cfg_ae_arc = Experiment.get_config().autoencoder.architecture
        self.n_codes: int = cfg_ae_arc.n_codes
        self.book_size: int = cfg_ae_arc.book_size
        self.embedding_dim: int = cfg_ae_arc.embedding_dim
        self.codebook = nn.Parameter(torch.randn(self.n_codes, self.book_size, self.embedding_dim))
        self.w_autoencoder: WA = self._init_w_autoencoder()
        for param in self.w_autoencoder.parameters():
            param.requires_grad = False  # separate training for W autoencoder

        self.quantizer = VectorQuantizer()
        self.transfer = TransferGrad()

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode with optional double encoding."""
        data = Outputs()
        data.w_q = self.encoder(inputs.cloud, inputs.indices)
        return data

    def double_encode(self, inputs: Inputs) -> Outputs:
        """Encode in the continuous space."""
        data = self.encode(inputs)
        encoded_data = self.w_autoencoder.encode(data.w_q.detach())
        data.update(encoded_data)
        return data

    def decode(self, data: Outputs, inputs: Inputs) -> Outputs:
        """Decode with vector quantization."""
        data.w_e, data.idx, _ = self.quantizer.quantize(data.w_q, self.codebook)
        data.one_hot_idx = self.quantizer.create_one_hot(data.idx)
        data.w = self.transfer.apply(data.w_e, data.w_q)
        return super().decode(data, inputs)

    def double_decode(self, data: Outputs, inputs: Inputs) -> Outputs:
        """Decode from the continuous space."""
        self.w_autoencoder.update_codebook(self.codebook)
        self.w_autoencoder.decode(data)
        data.w_e = data.w = self.quantizer.decode_from_indices(data.idx, self.codebook)
        return super().decode(data, inputs)

    @abc.abstractmethod
    def _init_w_autoencoder(self) -> WA:
        """Create the appropriate W autoencoder."""
        pass


class VQVAE(BaseVQVAE[WAutoEncoder]):
    """Standard VQVAE implementation."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def double_reconstruct(self, inputs: Inputs) -> Outputs:
        """Reconstruct from the continuous space."""
        data = self.double_encode(inputs)
        return self.double_decode(data, inputs)

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
    ) -> Outputs:
        """Generate samples from the model."""
        z1_bias = z1_bias.to(self.codebook.device)
        initial_sampling = initial_sampling.to(self.codebook.device)
        data = self.w_autoencoder.sample_latent(z1_bias, batch_size=batch_size)
        inputs = Inputs(self._null_tensor, initial_sampling)
        output = self.double_decode(data, inputs)
        return output

    def _init_w_autoencoder(self) -> WAutoEncoder:
        return WAutoEncoder()


class CounterfactualVQVAE(BaseVQVAE[CounterfactualWAutoEncoder]):
    """Counterfactual VQVAE implementation."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def double_reconstruct_with_logits(self, inputs: Inputs, logits: torch.Tensor) -> Outputs:
        """Reconstruct from the continuous space."""
        data = self.double_encode(inputs)
        data.probs = self.w_autoencoder.relaxed_softmax(logits)
        return self.double_decode(data, inputs)

    def generate_counterfactual(
        self,
        inputs: Inputs,
        sample_logits: torch.Tensor,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Generate counterfactual samples."""
        data = self.double_encode(inputs)
        self.w_autoencoder.update_codebook(self.codebook)
        data.probs = self.w_autoencoder.relaxed_softmax(sample_logits)
        data = self.w_autoencoder.counterfactual_decode(data, target_dim, target_value)
        data.w_e = data.w = self.quantizer.decode_from_indices(data.idx, self.codebook)
        return super(BaseVQVAE, self).decode(data, inputs)

    def _init_w_autoencoder(self) -> CounterfactualWAutoEncoder:
        return CounterfactualWAutoEncoder()

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
        probs: torch.Tensor | None = None,
    ) -> Outputs:
        """Generate samples from the model."""
        z1_bias = z1_bias.to(self.codebook.device)
        initial_sampling = initial_sampling.to(self.codebook.device)
        data = self.w_autoencoder.sample_latent(z1_bias, batch_size=batch_size, probs=probs)
        inputs = Inputs(self._null_tensor, initial_sampling)
        output = self.double_decode(data, inputs)
        return output


def get_autoencoder() -> AbstractAutoEncoder:
    """Factory function to create the appropriate autoencoder."""
    model_registry = {
        ModelHead.AE: BaseAutoencoder,
        ModelHead.VQVAE: VQVAE,
        ModelHead.CounterfactualVQVAE: CounterfactualVQVAE,
    }

    model_head = Experiment.get_config().autoencoder.architecture.head

    if model_head not in model_registry:
        raise ValueError(f'Unknown model head: {model_head}')

    return model_registry[model_head]()
