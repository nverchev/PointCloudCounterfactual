"""Autoencoder architecture."""

import abc

from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data_types import Inputs, Outputs
from src.module import get_decoder, BaseWAutoEncoder, WAutoEncoder, CounterfactualWAutoEncoder
from src.module import get_encoder
from src.module.layers import TransferGrad, reset_child_params
from src.utils.neighbour_ops import pykeops_square_distance
from src.utils.control import UsuallyFalse


class AutoEncoder(nn.Module, abc.ABC):
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


class Oracle(AutoEncoder):
    """Oracle autoencoder that returns an input subset."""

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        data = Outputs()
        data.recon = inputs.cloud[:, : self.m, :]
        return data


class AE(AutoEncoder):
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


class VectorQuantizer:
    """Handles vector quantization operations."""

    def __init__(self, codebook: torch.Tensor, n_codes: int, embedding_dim: int):
        self.codebook = codebook
        self.n_codes: int = n_codes
        self.embedding_dim: int = embedding_dim

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input using codebook."""
        batch, _ = x.size()
        x_flat = x.view(batch * self.n_codes, 1, self.embedding_dim)
        book_repeated = self.codebook.repeat(batch, 1, 1)

        dist = pykeops_square_distance(x_flat, book_repeated)
        idx = dist.argmin(axis=2).view(-1, 1, 1)

        embeddings = self._get_embeddings(idx, book_repeated)
        one_hot = self._create_one_hot(idx, batch, x.device)

        return embeddings, one_hot

    def _get_embeddings(self, idx: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """Get embeddings from indices."""
        idx_expanded = idx.expand(-1, -1, self.embedding_dim)
        embeddings = book.gather(1, idx_expanded)
        return embeddings.view(-1, self.n_codes * self.embedding_dim)

    def _create_one_hot(self, idx: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        """Create one-hot encoding of indices."""
        one_hot = torch.zeros(batch, self.n_codes, self.codebook.shape[1], device=device)
        return one_hot.scatter_(2, idx.view(batch, self.n_codes, 1), 1)


WA = TypeVar('WA', bound=BaseWAutoEncoder, covariant=True)


class AbstractVQVAE(AE, abc.ABC, Generic[WA]):
    """Abstract VQVAE with vector quantization."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def __init__(self):
        super().__init__()
        cfg_ae_arc = Experiment.get_config().autoencoder.architecture
        self.n_codes: int = cfg_ae_arc.n_codes
        self.book_size: int = cfg_ae_arc.book_size
        self.embedding_dim: int = cfg_ae_arc.embedding_dim
        self.double_encoding = UsuallyFalse()
        self.codebook = nn.Parameter(torch.randn(self.n_codes, self.book_size, self.embedding_dim))

        self.w_autoencoder = self._create_w_autoencoder()
        for param in self.w_autoencoder.parameters():
            param.requires_grad = False  # separate training for W autoencoder

        self.quantizer = VectorQuantizer(self.codebook, self.n_codes, self.embedding_dim)
        self.transfer = TransferGrad()

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode with optional double encoding."""
        data = Outputs()
        data.w_q = self.encoder(inputs.cloud, inputs.indices)

        if self.double_encoding:
            encoded_data = self.w_autoencoder.encode(data.w_q.detach())
            data.update(encoded_data)

        return data

    def decode(self, data: Outputs, inputs: Inputs) -> Outputs:
        """Decode with vector quantization."""
        if self.double_encoding:
            self.w_autoencoder.update_codebook(self.codebook)
            self.w_autoencoder.decode(data)
            data.w_e = data.w = self._decode_from_indices(data)
        else:
            data.w_e, data.one_hot_idx = self.quantizer.quantize(data.w_q)
            data.w = self.transfer.apply(data.w_e, data.w_q)

        return super().decode(data, inputs)

    def _decode_from_indices(self, data: Outputs) -> torch.Tensor:
        """Decode embeddings from indices."""
        idx = data.idx
        batch = idx.shape[0]
        book = self.codebook.repeat(batch, 1, 1)
        idx_flat = idx.view(batch * idx.shape[1], 1, 1)
        idx_expanded = idx_flat.expand(-1, -1, self.embedding_dim)
        embeddings = book.gather(1, idx_expanded)

        return embeddings.view(-1, self.n_codes * self.embedding_dim)

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
        **kwargs: Any,
    ) -> Outputs:
        """Generate samples from the model."""
        z1_bias = z1_bias.to(self.codebook.device)
        initial_sampling = initial_sampling.to(self.codebook.device)
        data = self.w_autoencoder.sample_latent(z1_bias, batch_size=batch_size, **kwargs)
        inputs = Inputs(self._null_tensor, initial_sampling)
        with self.double_encoding:
            output = self.decode(data, inputs)

        return output

    @abc.abstractmethod
    def _create_w_autoencoder(self) -> WA:
        """Create the appropriate W autoencoder."""
        pass


class VQVAE(AbstractVQVAE[WAutoEncoder]):
    """Standard VQVAE implementation."""

    def _create_w_autoencoder(self) -> WAutoEncoder:
        return WAutoEncoder()


class CounterfactualVQVAE(AbstractVQVAE[CounterfactualWAutoEncoder]):
    """Counterfactual VQVAE implementation."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def _create_w_autoencoder(self) -> CounterfactualWAutoEncoder:
        return CounterfactualWAutoEncoder()

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
        probs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Outputs:
        """Generate counterfactual samples."""
        return super().generate(batch_size=batch_size, initial_sampling=initial_sampling, z1_bias=z1_bias, probs=probs)
