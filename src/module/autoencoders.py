"""Autoencoder architecture."""

import abc

from typing import Generic, TypeVar

import torch
import torch.nn as nn

from src.config import Experiment
from src.config.options import AutoEncoders
from src.data.structures import Inputs, Outputs, WInputs
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
        self.m_training: int = cfg_ae.training_output_points
        self.m_test: int = cfg_ae.objective.n_inference_output_points
        return

    @property
    def m(self) -> int:
        """Number of generated points."""
        return self.m_test if torch.is_inference_mode_enabled() else self.m_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""

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
        return

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

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def __init__(self):
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        self.n_codes: int = cfg_ae_model.n_codes
        self.book_size: int = cfg_ae_model.book_size
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.codebook = nn.Parameter(torch.randn(self.n_codes, self.book_size, self.embedding_dim))
        self.w_autoencoder: WA = self._init_w_autoencoder()
        for param in self.w_autoencoder.parameters():
            param.requires_grad = False  # separate training for W autoencoder

        self.quantizer = VectorQuantizer()
        self.transfer = TransferGrad()
        return

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode with optional double encoding."""
        data = Outputs()
        data.w_q = self.encoder(inputs.cloud, inputs.indices)
        return data

    def decode(self, data: Outputs, inputs: Inputs) -> Outputs:
        """Decode with vector quantization."""
        data.w_e, data.idx, _ = self.quantizer.quantize(data.w_q, self.codebook)
        data.one_hot_idx = self.quantizer.create_one_hot(data.idx)
        data.w = self.transfer.apply(data.w_e, data.w_q)
        return super().decode(data, inputs)

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
        data = self.w_autoencoder.generate_discrete_latent_space(z1_bias, batch_size=batch_size, probs=probs)
        data.w_e = data.w = self.quantizer.decode_from_indices(data.idx, self.codebook)
        return BaseAutoencoder.decode(self, data, inputs)


class VQVAE(BaseVQVAE[WAutoEncoder]):
    """Standard VQVAE implementation."""

    def double_reconstruct(self, inputs: Inputs) -> Outputs:
        """Reconstruct from the continuous space."""
        self.w_autoencoder.update_codebook(self.codebook)
        w_q = self.encode(inputs).w_q.view(-1, self.n_codes, self.embedding_dim)
        data = self.w_autoencoder(WInputs(w_q))
        data.w_e = data.w = self.quantizer.decode_from_indices(data.idx, self.codebook)
        return BaseAutoencoder.decode(self, data, inputs)

    def _init_w_autoencoder(self) -> WAutoEncoder:
        return WAutoEncoder()


class CounterfactualVQVAE(BaseVQVAE[CounterfactualWAutoEncoder]):
    """Counterfactual VQVAE implementation."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def double_reconstruct_with_logits(self, inputs: Inputs, logits: torch.Tensor) -> Outputs:
        """Reconstruct from the continuous space."""
        self.w_autoencoder.update_codebook(self.codebook)
        w_q = self.encode(inputs).w_q
        data = self.w_autoencoder(WInputs(w_q, logits))
        data.w_e = data.w = self.quantizer.decode_from_indices(data.idx, self.codebook)
        return BaseAutoencoder.decode(self, data, inputs)

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
        w_q = self.encode(inputs).w_q
        data = self.w_autoencoder.generate_counterfactual(WInputs(w_q, sample_logits), target_dim, target_value)
        data.w_e = data.w = self.quantizer.decode_from_indices(data.idx, self.codebook)
        return BaseAutoencoder.decode(self, data, inputs)

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
