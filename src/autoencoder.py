"""Autoencoder architecture."""

import abc
from typing import Generic, Optional, TypeVar, Any, Generator, Callable

import numpy as np
import torch
import torch.nn as nn

from src.data_structures import Inputs, Outputs, W_Inputs
from src.encoders import get_encoder, get_w_encoder
from src.decoders import PriorDecoder, PosteriorDecoder, get_decoder, get_w_decoder
from src.neighbour_ops import pykeops_square_distance
from src.layers import TransferGrad, TemperatureScaledSoftmax, frozen_forward
from src.utils import UsuallyFalse
from src.config_options import Experiment, ModelHead

class GaussianSampler:
    """Handles Gaussian sampling logic."""

    @staticmethod
    def sample(mu: torch.Tensor, log_var: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Gaussian sample given mean and variance (returns mean if not training)."""
        if not training:
            return mu

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class DistanceCalculator:
    """Handles distance calculations for quantization."""

    @staticmethod
    def compute_distances(x: torch.Tensor,
                          codebook: torch.Tensor,
                          dim_codes: int,
                          embedding_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate distances from embeddings and return closest indices."""
        batch, _ = x.shape
        x = x.view(batch * dim_codes, 1, embedding_dim)
        book = codebook.detach().repeat(batch, 1, 1)
        dist = pykeops_square_distance(x, book)

        idx = dist.argmin(axis=2)
        dist_sum = dist.sum(1).view(batch, dim_codes, codebook.shape[1])
        idx_reshaped = idx.view(batch, dim_codes, 1)

        return dist_sum, idx_reshaped


class PseudoInputManager:
    """Manages pseudo inputs and their latent representations."""

    def __init__(self, n_pseudo_inputs: int, embedding_dim: int, z_dim: int, dim_codes: int):
        self.n_pseudo_inputs = n_pseudo_inputs
        self.pseudo_inputs = nn.Parameter(torch.empty(n_pseudo_inputs, embedding_dim, dim_codes))
        self.pseudo_mu = nn.Parameter(torch.empty(n_pseudo_inputs, z_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(n_pseudo_inputs, z_dim))
        self.updated = False
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize pseudo input parameters."""
        nn.init.normal_(self.pseudo_inputs)
        nn.init.normal_(self.pseudo_mu)
        nn.init.normal_(self.pseudo_log_var)

    def update_pseudo_latent(self, encoder_func: Callable[[None], Outputs]) -> None:
        """Update pseudo latent parameters based on encoder output."""
        pseudo_data = encoder_func(None)
        self.pseudo_mu.data = pseudo_data.pseudo_mu1
        self.pseudo_log_var.data = pseudo_data.pseudo_log_var1
        self.updated = True
        return

    def get_combined_input(self, x: Optional[torch.Tensor], dim_codes: int, embedding_dim: int) -> torch.Tensor:
        """Combine input with pseudo inputs."""
        if x is None:
            return self.pseudo_inputs

        x_reshaped = x.view(-1, dim_codes, embedding_dim).transpose(2, 1)
        return torch.cat((x_reshaped, self.pseudo_inputs))

    def sample_pseudo_latent(self, batch_size: int) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Sample pseudo latents."""
        for _ in range(batch_size):
            i = np.random.randint(self.n_pseudo_inputs)
            yield self.pseudo_mu[i], self.pseudo_log_var[i]


class BaseWAutoEncoder(nn.Module, abc.ABC):
    """Base class for W autoencoders with common functionality."""

    def __init__(self, codebook: torch.Tensor):
        super().__init__()

        self.cfg_ae_model = Experiment.get_config().autoencoder.model
        self.codebook = codebook
        self.dim_codes, self.book_size, self.embedding_dim = codebook.shape
        self.encoder = get_w_encoder()
        self.decoder = get_w_decoder()
        self.bn = nn.BatchNorm1d(self.cfg_ae_model.z1_dim + self.cfg_ae_model.z2_dim)
        self.sampler = GaussianSampler()
        self.distance_calc = DistanceCalculator()

        if self.cfg_ae_model.n_pseudo_inputs > 0:
            self.pseudo_manager: PseudoInputManager | None = PseudoInputManager(
                self.cfg_ae_model.n_pseudo_inputs,
                self.cfg_ae_model.embedding_dim,
                self.cfg_ae_model.z1_dim,
                self.dim_codes,
            )
        else:
            self.pseudo_manager = None

    def forward(self, x: W_Inputs) -> Outputs:
        """Forward pass."""
        data = self.encode(x.w_q)
        return self.decode(data)

    @abc.abstractmethod
    def encode(self, x: Optional[torch.Tensor]) -> Outputs:
        """Encode input to latent space."""
        pass

    @abc.abstractmethod
    def decode(self, data: Outputs) -> Outputs:
        """Decode from latent space."""
        pass

    @torch.inference_mode()
    def sample_latent(self, z1_bias: torch.Tensor, batch_size: int = 1) -> Outputs:
        """Generate samples from the distribution."""
        self.eval()
        data = Outputs()

        if self.pseudo_manager and self.cfg_ae_model.n_pseudo_inputs > 0:
            pseudo_z_list = []
            for mu, log_var in self.pseudo_manager.sample_pseudo_latent(batch_size):
                pseudo_z_list.append(self.sampler.sample(mu, log_var))
            z1 = torch.stack(pseudo_z_list).contiguous()
        else:
            z1 = torch.randn((batch_size, self.cfg_ae_model.z1_dim), device=z1_bias.device)

        data.z1 = z1 + z1_bias
        return data

    def _get_input(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """Get input tensor, combining with pseudo inputs if available."""
        if self.pseudo_manager is None:
            return x if x is not None else torch.empty(0)

        return self.pseudo_manager.get_combined_input(x, self.dim_codes, self.embedding_dim)

    def recursive_reset_parameters(self):
        """Reset all parameters."""
        if self.pseudo_manager:
            self.pseudo_manager.initialize_parameters()

        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module is not self.codebook:
                module.reset_parameters()


class WAutoEncoder(BaseWAutoEncoder):
    """W autoencoder implementation."""

    def __init__(self, codebook: torch.Tensor):
        cfg = Experiment.get_config()
        super().__init__(codebook)
        self.n_classes = cfg.data.dataset.n_classes

    def encode(self, x: Optional[torch.Tensor]) -> Outputs:
        """Encode with adversarial features."""
        input_tensor = self._get_input(x)
        _, latent = self.encoder(input_tensor)
        data = Outputs()

        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index)
            data.pseudo_mu1, data.pseudo_log_var1 = pseudo_latent.chunk(2, 1)

        data.mu1, data.log_var1 = latent.chunk(2, 1)

        return data

    def decode(self, data: Outputs) -> Outputs:
        """Decode with counterfactual generation."""
        data.z1 = self.sampler.sample(data.mu1, data.log_var1, self.training)
        data.z2 = torch.randn((data.z1.shape[0], self.cfg_ae_model.z2_dim))  # z2_dim should probably be equal 0 here
        z_combined = self.bn(torch.cat((data.z1, data.z2), dim=1))
        data.w_recon = self.decoder(z_combined)
        data.w_dist, data.idx = self.distance_calc.compute_distances(
            data.w_recon, self.codebook, self.dim_codes, self.embedding_dim
        )

        return data


class CounterfactualWAutoEncoder(BaseWAutoEncoder):
    """Counterfactual W autoencoder with adversarial components."""

    def __init__(self, codebook: torch.Tensor):
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_wae = cfg_ae_model.encoder.w_encoder
        super().__init__(codebook)
        self.n_classes = cfg.data.dataset.n_classes

        # Counterfactual-specific components
        self.softmax = nn.Softmax(dim=1)
        self.relaxed_softmax = TemperatureScaledSoftmax(dim=1, temperature=cfg_wae.cf_temperature)
        self.z2_inference = PriorDecoder()
        self.posterior = PosteriorDecoder()
        self.adversarial = nn.Linear(cfg_ae_model.z1_dim, self.n_classes)

    def encode(self, x: Optional[torch.Tensor]) -> Outputs:
        """Encode with adversarial features."""
        input_tensor = self._get_input(x)
        features, latent = self.encoder(input_tensor)

        data = Outputs()

        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index)
            feature, _ = torch.tensor_split(features, split_index, dim=0)
            data.pseudo_mu1, data.pseudo_log_var1 = pseudo_latent.chunk(2, 1)

        data.mu1, data.log_var1 = latent.chunk(2, 1)
        data.h = features

        # Adversarial predictions
        data.y1 = self.adversarial(data.mu1.detach())
        data.y2 = frozen_forward(self.adversarial, data.mu1)
        data.z1 = self.sampler.sample(data.mu1, data.log_var1, self.training)

        return data

    def decode(self, data: Outputs, logits: Optional[torch.Tensor] = None) -> Outputs:
        """Decode with counterfactual generation."""

        # Handle probability computation
        probs = self._get_probabilities(data, logits)
        data.p_mu2, data.p_log_var2 = self.z2_inference(probs).chunk(2, 1)

        # Compute z2 based on whether we have features
        data.z2 = self._compute_z2(data, probs)

        # Combine and decode
        z_combined = self.bn(torch.cat((data.z1, data.z2), dim=1))
        data.w_recon = self.decoder(z_combined)
        data.w_dist, data.idx = self.distance_calc.compute_distances(
            data.w_recon, self.codebook, self.dim_codes, self.embedding_dim
        )

        return data

    def _get_probabilities(self, data: Outputs, logits: Optional[torch.Tensor]) -> torch.Tensor:
        """Get probabilities from logits or stored probs."""
        if logits is not None:
            return self.relaxed_softmax(logits)

        if hasattr(data, 'probs'):
            return data.probs

        elif hasattr(data, 'y1'):  # mostly for testing
            return self.relaxed_softmax(data.y1.detach())

        raise AttributeError('Either logits, approximated logits, or probs must be set.')

    def _compute_z2(self, data: Outputs, probs: torch.Tensor) -> torch.Tensor:
        """Compute z2 latent variable."""
        if not hasattr(data, 'h'):
            # No features available, sample from prior
            return self.sampler.sample(data.p_mu2, data.p_log_var2, self.training)

        # Use posterior with features
        h_combined = torch.cat((probs, data.h), dim=1)
        data.d_mu2, data.d_log_var2 = self.posterior(h_combined).chunk(2, 1)

        mu_combined = data.d_mu2 + data.p_mu2
        log_var_combined = data.d_log_var2 + data.p_log_var2

        return self.sampler.sample(mu_combined, log_var_combined, self.training)

    def forward(self, x: W_Inputs) -> Outputs:
        """Forward pass with logits."""
        data = self.encode(x.w_q)
        return self.decode(data, x.logits)

    @torch.inference_mode()
    def sample_latent(self,
                      z1_bias: torch.Tensor,
                      batch_size: int = 1,
                      probs: Optional[torch.Tensor] = None,
                      ) -> Outputs:
        """Generate samples with counterfactual probabilities."""
        data = super().sample_latent(z1_bias, batch_size)

        if probs is not None and probs.numel() > 0:
            data.probs = probs.to(self.codebook.device)

        else:
            data.probs = self.softmax(torch.rand((batch_size, self.n_classes), device=self.codebook.device))

        return data


# Type variable for W autoencoder
WA = TypeVar('WA', bound=BaseWAutoEncoder, covariant=True)


class AutoEncoder(nn.Module, abc.ABC):
    """Abstract autoencoder for point clouds."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config().autoencoder
        self.m_training = cfg.model.training_output_points
        self.m_test = cfg.objective.n_inference_output_points

    @property
    def m(self) -> int:
        """Number of generated points."""
        return self.m_test if torch.is_inference_mode_enabled() else self.m_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        pass

    def recursive_reset_parameters(self):
        """Reset all parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class Oracle(AutoEncoder):
    """Oracle autoencoder that returns an input subset."""

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        data = Outputs()
        data.recon = inputs.cloud[:, :self.m, :]
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
        data.w = self.encoder(inputs.cloud, inputs.indices)
        return data

    def decode(self, data: Outputs, inputs: Inputs) -> Outputs:
        """Decode latent representation to point cloud."""
        x = self.decoder(data.w, self.m, inputs.initial_sampling)
        data.recon = x.transpose(2, 1).contiguous()
        return data


class VectorQuantizer:
    """Handles vector quantization operations."""

    def __init__(self, codebook: torch.Tensor, num_codes: int, embedding_dim: int):
        self.codebook = codebook
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input using codebook."""
        batch, _ = x.size()
        x_flat = x.view(batch * self.num_codes, 1, self.embedding_dim)
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
        return embeddings.view(-1, self.num_codes * self.embedding_dim)

    def _create_one_hot(self, idx: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        """Create one-hot encoding of indices."""
        one_hot = torch.zeros(batch, self.num_codes, self.codebook.shape[1], device=device)
        return one_hot.scatter_(2, idx.view(batch, self.num_codes, 1), 1)


class AbstractVQVAE(AE, Generic[WA], abc.ABC):
    """Abstract VQVAE with vector quantization."""

    def __init__(self):
        super().__init__()
        self.cfg_ae_model = Experiment.get_config().autoencoder.model

        self.double_encoding = UsuallyFalse()
        self.codebook = nn.Parameter(torch.randn(
            self.cfg_ae_model.n_codes,
            self.cfg_ae_model.book_size,
            self.cfg_ae_model.embedding_dim
        ))

        self.w_autoencoder = self._create_w_autoencoder()
        self.quantizer = VectorQuantizer(
            self.codebook, self.cfg_ae_model.n_codes, self.cfg_ae_model.embedding_dim
        )
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
        idx_expanded = idx_flat.expand(-1, -1, self.cfg_ae_model.embedding_dim)
        embeddings = book.gather(1, idx_expanded)

        return embeddings.view(-1, self.cfg_ae_model.n_codes * self.cfg_ae_model.embedding_dim)

    @torch.inference_mode()
    def generate(self,
                 batch_size: int = 1,
                 initial_sampling: torch.Tensor = torch.empty(0),
                 z1_bias: torch.Tensor = torch.tensor(0),
                 **kwargs: Any) -> Outputs:
        """Generate samples from the model."""
        z1_bias = z1_bias.to(self.codebook.device)
        initial_sampling = initial_sampling.to(self.codebook.device)
        data = self.w_autoencoder.sample_latent(z1_bias, batch_size=batch_size, **kwargs)

        inputs = Inputs(torch.empty(0), initial_sampling)

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
        return WAutoEncoder(self.codebook)


class CounterfactualVQVAE(AbstractVQVAE[CounterfactualWAutoEncoder]):
    """Counterfactual VQVAE implementation."""

    def _create_w_autoencoder(self) -> CounterfactualWAutoEncoder:
        return CounterfactualWAutoEncoder(self.codebook)

    @torch.inference_mode()
    def generate(self,
                 batch_size: int = 1,
                 initial_sampling: torch.Tensor = torch.empty(0),
                 z1_bias: torch.Tensor = torch.empty(0),
                 probs: torch.Tensor = torch.empty(0),
                 **kwargs: Any) -> Outputs:
        """Generate counterfactual samples."""
        return super().generate(
            batch_size=batch_size,
            initial_sampling=initial_sampling,
            z1_bias=z1_bias,
            probs=probs
        )


def get_autoencoder() -> AutoEncoder:
    """Factory function to create the appropriate autoencoder."""
    model_registry = {
        ModelHead.AE: AE,
        ModelHead.VQVAE: VQVAE,
        ModelHead.CounterfactualVQVAE: CounterfactualVQVAE
    }

    model_head = Experiment.get_config().autoencoder.model.head

    if model_head not in model_registry:
        raise ValueError(f"Unknown model head: {model_head}")

    return model_registry[model_head]()
