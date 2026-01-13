"""Autoencoder architecture."""

import abc

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config_options import Experiment, ModelHead
from src.data_structures import Inputs, Outputs, WInputs
from src.decoders import PosteriorDecoder, PriorDecoder, get_decoder, get_w_decoder
from src.encoders import get_encoder, get_w_encoder
from src.layers import TemperatureScaledSoftmax, TransferGrad, reset_child_params
from src.neighbour_ops import pykeops_square_distance
from src.utils import UsuallyFalse


class GaussianSampler:
    """Handles Gaussian sampling logic."""

    @staticmethod
    def sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Gaussian sample given mean and variance (returns mean if not training)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class DistanceCalculator:
    """Handles distance calculations for quantization."""

    @staticmethod
    def compute_distances(
        x: torch.Tensor, codebook: torch.Tensor, dim_codes: int, embedding_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def get_combined_input(self, x: torch.Tensor | None, dim_codes: int, embedding_dim: int) -> torch.Tensor:
        """Combine input with pseudo inputs."""
        if x is None:
            return self.pseudo_inputs

        x_reshaped = x.view(-1, dim_codes, embedding_dim).transpose(2, 1)
        return torch.cat((x_reshaped, self.pseudo_inputs))

    def sample_pseudo_latent(self, batch_size: int) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        """Sample pseudo latents."""
        for _ in range(batch_size):
            i = np.random.randint(self.n_pseudo_inputs)
            yield self.pseudo_mu[i], self.pseudo_log_var[i]


class BaseWAutoEncoder(nn.Module, abc.ABC):
    """Base class for W autoencoders with common functionality."""

    def __init__(self):
        super().__init__()
        self.cfg_ae_arc = Experiment.get_config().autoencoder.architecture
        self.dim_codes = self.cfg_ae_arc.n_codes
        self.book_size = self.cfg_ae_arc.book_size
        self.embedding_dim = self.cfg_ae_arc.embedding_dim
        self.encoder = get_w_encoder()
        self.decoder = get_w_decoder()
        self.sampler = GaussianSampler()
        self.distance_calc = DistanceCalculator()
        self._codebook: torch.Tensor | None = None

        if self.cfg_ae_arc.n_pseudo_inputs > 0:
            self.pseudo_manager: PseudoInputManager | None = PseudoInputManager(
                self.cfg_ae_arc.n_pseudo_inputs,
                self.cfg_ae_arc.embedding_dim,
                self.cfg_ae_arc.z1_dim,
                self.dim_codes,
            )
        else:
            self.pseudo_manager = None

    @property
    def codebook(self) -> torch.Tensor:
        """Codebook tensor."""
        if self._codebook is None:
            raise AttributeError('Codebook not initialized.')

        return self._codebook

    def update_codebook(self, codebook: torch.Tensor) -> None:
        """Update codebook tensor."""
        self.register_buffer('_codebook', codebook.data, persistent=False)

    def forward(self, x: WInputs) -> Outputs:
        """Forward pass."""
        data = self.encode(x.w_q)
        return self.decode(data)

    @abc.abstractmethod
    def encode(self, x: torch.Tensor | None) -> Outputs:
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

        if self.pseudo_manager and self.cfg_ae_arc.n_pseudo_inputs > 0:
            pseudo_z_list = []
            for mu, log_var in self.pseudo_manager.sample_pseudo_latent(batch_size):
                pseudo_z_list.append(self.sampler.sample(mu, log_var))

            z1 = torch.stack(pseudo_z_list).contiguous()
        else:
            z1 = torch.randn((batch_size, 1, self.cfg_ae_arc.z1_dim), device=z1_bias.device)

        data.z1 = z1 + z1_bias
        return data

    def _get_input(self, x: torch.Tensor | None) -> torch.Tensor:
        """Get input tensor, combining with pseudo inputs if available."""
        if self.pseudo_manager is None:
            return x if x is not None else torch.empty(0)

        return self.pseudo_manager.get_combined_input(x, self.dim_codes, self.embedding_dim)

    def recursive_reset_parameters(self):
        """Reset all parameters."""
        if self.pseudo_manager:
            self.pseudo_manager.initialize_parameters()

        reset_child_params(self)
        return


class WAutoEncoder(BaseWAutoEncoder):
    """W autoencoder implementation."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        self.n_classes = cfg.data.dataset.n_classes

    def encode(self, x: torch.Tensor | None) -> Outputs:
        """Encode with adversarial features."""
        input_tensor = self._get_input(x)
        _, latent = self.encoder(input_tensor)
        data = Outputs()
        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index)
            data.pseudo_mu1, data.pseudo_log_var1 = pseudo_latent.chunk(2, 2)

        data.mu1, data.log_var1 = latent.chunk(2, 2)

        return data

    def decode(self, data: Outputs) -> Outputs:
        """Decode with counterfactual generation."""
        data.z1 = self.sampler.sample(data.mu1, data.log_var1)
        data.z2 = torch.zeros((data.z1.shape[0], self.cfg_ae_arc.z2_dim))
        data.w_recon = self.decoder(data.z1, data.z2)
        data.w_dist_2, data.idx = self.distance_calc.compute_distances(
            data.w_recon, self.codebook, self.dim_codes, self.embedding_dim
        )
        return data


class CounterfactualWAutoEncoder(BaseWAutoEncoder):
    """Counterfactual W autoencoder with adversarial components."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_arc = cfg.autoencoder.architecture
        cfg_wae = cfg_ae_arc.encoder.w_encoder
        self.n_classes = cfg.data.dataset.n_classes

        # Counterfactual-specific components
        self.softmax = nn.Softmax(dim=1)
        self.relaxed_softmax = TemperatureScaledSoftmax(dim=1, temperature=cfg_wae.cf_temperature)
        self.z2_inference = PriorDecoder()
        self.posterior = PosteriorDecoder()
        # self.adversarial = nn.Linear(cfg_ae_arc.z1_dim, self.n_classes)

        return

    def encode(self, x: torch.Tensor | None) -> Outputs:
        """Encode with adversarial features."""
        input_tensor = self._get_input(x)
        features, latent = self.encoder(input_tensor)

        data = Outputs()

        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index)
            features, _ = torch.tensor_split(features, split_index, dim=0)
            data.pseudo_mu1, data.pseudo_log_var1 = pseudo_latent.chunk(2, 2)

        data.mu1, data.log_var1 = latent.chunk(2, 2)
        data.h = features

        # Adversarial predictions
        # data.y1 = self.adversarial(data.mu1.detach())
        # data.y2 = frozen_forward(self.adversarial, data.mu1)
        data.z1 = self.sampler.sample(data.mu1, data.log_var1)
        return data

    def decode(self, data: Outputs, logits: torch.Tensor | None = None) -> Outputs:
        """Decode with counterfactual generation."""
        probs = self._get_probabilities(data, logits)
        data.p_mu2, data.p_log_var2 = self.z2_inference(probs).chunk(2, 2)
        data.z2 = self._compute_z2(data, probs)
        data.w_recon = self.decoder(data.z1, data.z2)
        data.w_dist_2, data.idx = self.distance_calc.compute_distances(
            data.w_recon, self.codebook, self.dim_codes, self.embedding_dim
        )
        return data

    def _get_probabilities(self, data: Outputs, logits: torch.Tensor | None) -> torch.Tensor:
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
            return self.sampler.sample(data.p_mu2, data.p_log_var2)

        # Use posterior with features
        data.d_mu2, data.d_log_var2 = self.posterior(probs, data.h).chunk(2, 2)
        mu_combined = data.d_mu2 + data.p_mu2
        log_var_combined = data.d_log_var2 + data.p_log_var2
        return self.sampler.sample(mu_combined, log_var_combined)

    def forward(self, x: WInputs) -> Outputs:
        """Forward pass with logits."""
        data = self.encode(x.w_q)
        data.w_q = x.w_q
        return self.decode(data, x.logits)

    @torch.inference_mode()
    def sample_latent(
        self,
        z1_bias: torch.Tensor,
        batch_size: int = 1,
        probs: torch.Tensor | None = None,
    ) -> Outputs:
        """Generate samples with counterfactual probabilities."""
        data = super().sample_latent(z1_bias, batch_size)
        if probs is not None and probs.numel() > 0:
            data.probs = probs.to(z1_bias.device)

        else:
            alpha = torch.ones(self.n_classes, device=z1_bias.device)
            data.probs = torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))

        return data


class AutoEncoder(nn.Module, abc.ABC):
    """Abstract autoencoder for point clouds."""

    def __init__(self):
        super().__init__()
        cfg_ae = Experiment.get_config().autoencoder
        self.m_training = cfg_ae.architecture.training_output_points
        self.m_test = cfg_ae.objective.n_inference_output_points

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


class AbstractVQVAE[WA: BaseWAutoEncoder](AE, abc.ABC):
    """Abstract VQVAE with vector quantization."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def __init__(self):
        super().__init__()
        cfg_ae_arc = Experiment.get_config().autoencoder.architecture
        self.n_codes = cfg_ae_arc.n_codes
        self.book_size = cfg_ae_arc.book_size
        self.embedding_dim = cfg_ae_arc.embedding_dim
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


def get_autoencoder() -> AutoEncoder:
    """Factory function to create the appropriate autoencoder."""
    model_registry = {ModelHead.AE: AE, ModelHead.VQVAE: VQVAE, ModelHead.CounterfactualVQVAE: CounterfactualVQVAE}

    model_head = Experiment.get_config().autoencoder.architecture.head

    if model_head not in model_registry:
        raise ValueError(f'Unknown model head: {model_head}')

    return model_registry[model_head]()
