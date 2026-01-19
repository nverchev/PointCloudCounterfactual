"""W-autoencoder architecture."""

import abc
from collections.abc import Callable, Generator

import numpy as np
import torch
from torch import nn as nn

from src.config import Experiment
from src.data.structures import Outputs, WInputs
from src.module.w_decoders import PriorDecoder, PosteriorDecoder, get_w_decoder
from src.module.w_encoders import get_w_encoder
from src.module.layers import reset_child_params, TemperatureScaledSoftmax
from src.module.quantize import VectorQuantizer


class GaussianSampler:
    """Handles Gaussian sampling logic."""

    @staticmethod
    def sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Gaussian sample given mean and variance (returns mean if not training)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class PseudoInputManager:
    """Manages pseudo inputs and their latent representations."""

    def __init__(self, n_pseudo_inputs: int, embedding_dim: int, z1_dim: int, n_codes: int):
        self.n_pseudo_inputs: int = n_pseudo_inputs
        self.pseudo_inputs = nn.Parameter(torch.empty(n_pseudo_inputs, n_codes, embedding_dim))
        self.pseudo_mu = nn.Parameter(torch.empty(n_pseudo_inputs, z1_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(n_pseudo_inputs, z1_dim))
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

    def get_combined_input(self, x: torch.Tensor | None, n_codes: int, embedding_dim: int) -> torch.Tensor:
        """Combine input with pseudo inputs."""
        if x is None:
            return self.pseudo_inputs

        x_reshaped = x.view(-1, n_codes, embedding_dim).transpose(2, 1)
        return torch.cat((x_reshaped, self.pseudo_inputs))

    def sample_pseudo_latent(self, batch_size: int) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        """Sample pseudo latents."""
        for _ in range(batch_size):
            i = np.random.randint(self.n_pseudo_inputs)
            yield self.pseudo_mu[i], self.pseudo_log_var[i]


class BaseWAutoEncoder(nn.Module, abc.ABC):
    """Base class for W autoencoders with common settings and interface."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        self.cfg_ae_arc = cfg.autoencoder.architecture
        self.n_codes: int = self.cfg_ae_arc.n_codes
        self.book_size: int = self.cfg_ae_arc.book_size
        self.embedding_dim: int = self.cfg_ae_arc.embedding_dim
        self.n_classes: int = cfg.data.dataset.n_classes
        self._codebook: torch.Tensor | None = None
        self.encoder = get_w_encoder()
        self.decoder = get_w_decoder()
        self.sampler = GaussianSampler()
        self.quantizer = VectorQuantizer()

        if self.cfg_ae_arc.n_pseudo_inputs > 0:
            self.pseudo_manager: PseudoInputManager | None = PseudoInputManager(
                self.cfg_ae_arc.n_pseudo_inputs,
                self.cfg_ae_arc.embedding_dim,
                self.cfg_ae_arc.z1_dim,
                self.n_codes,
            )
        else:
            self.pseudo_manager = None

    @abc.abstractmethod
    def forward(self, x: WInputs) -> Outputs:
        """Forward pass."""

    @property
    def codebook(self) -> torch.Tensor:
        """Codebook tensor."""
        if self._codebook is None:
            raise AttributeError('Codebook not initialized.')

        return self._codebook

    def update_codebook(self, codebook: torch.Tensor) -> None:
        """Update codebook tensor."""
        del self._codebook

        self.register_buffer('_codebook', codebook.data, persistent=False)
        return

    def encode(self, x: torch.Tensor | None) -> Outputs:
        """Encode from discrete to continuous latent space."""
        data = Outputs()
        if x is not None:
            data.h = x  # using the original input for encoding z2

        input_tensor = self._get_input(x)
        latent = self.encoder(input_tensor)

        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index)
            data.pseudo_mu1, data.pseudo_log_var1 = pseudo_latent.chunk(2, 2)

        data.mu1, data.log_var1 = latent.chunk(2, 2)
        data.z1 = self.sampler.sample(data.mu1, data.log_var1)
        return data

    def decode(self, data: Outputs) -> Outputs:
        """Decode from continuous to discrete latent space."""
        data.z2 = self._compute_z2(data)
        data.w_recon = self.decoder(data.z1, data.z2)
        _, data.idx, data.w_dist_2 = self.quantizer.quantize(data.w_recon, self.codebook)
        return data

    def recursive_reset_parameters(self):
        """Reset all parameters."""
        if self.pseudo_manager:
            self.pseudo_manager.initialize_parameters()

        reset_child_params(self)
        return

    @abc.abstractmethod
    def _compute_z2(self, data: Outputs) -> torch.Tensor:
        """Compute z2 latent variable."""

    def _get_input(self, x: torch.Tensor | None) -> torch.Tensor:
        """Get input tensor, combining with pseudo inputs if available."""
        if self.pseudo_manager is None:
            return x if x is not None else torch.empty(0)

        return self.pseudo_manager.get_combined_input(x, self.n_codes, self.embedding_dim)

    @torch.inference_mode()
    def _sample_z1(self, z1_bias: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """Sample z from the prior distribution."""
        self.eval()
        if self.pseudo_manager and self.cfg_ae_arc.n_pseudo_inputs > 0:
            pseudo_z_list = []
            for mu, log_var in self.pseudo_manager.sample_pseudo_latent(batch_size):
                pseudo_z_list.append(self.sampler.sample(mu, log_var))

            z1 = torch.stack(pseudo_z_list).contiguous()
        else:
            z1 = torch.randn((batch_size, 1, self.cfg_ae_arc.z1_dim), device=z1_bias.device)

        return z1 + z1_bias


class WAutoEncoder(BaseWAutoEncoder):
    """Standard W autoencoder."""

    def forward(self, x: WInputs) -> Outputs:
        """Forward pass."""
        data = self.encode(x.w_q)
        return self.decode(data)

    def sample_latent(self, z1_bias: torch.Tensor, batch_size: int = 1) -> Outputs:
        """Sample from the latent space."""
        data = Outputs()
        data.z1 = self._sample_z1(z1_bias, batch_size)
        return data

    def _compute_z2(self, data: Outputs) -> torch.Tensor:
        """Copies z1 into z2"""
        return data.z1


class CounterfactualWAutoEncoder(BaseWAutoEncoder):
    """W autoencoder with condition to classifier logits."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_arc = cfg.autoencoder.architecture
        cfg_wae = cfg_ae_arc.encoder.w_encoder

        # Counterfactual-specific components
        self.softmax = nn.Softmax(dim=1)
        self.relaxed_softmax = TemperatureScaledSoftmax(dim=1, temperature=cfg_wae.cf_temperature)
        self.z2_inference = PriorDecoder()
        self.posterior = PosteriorDecoder()
        return

    def forward(self, x: WInputs) -> Outputs:
        """Forward pass."""
        data = self.encode(x.w_q)
        data.probs = self.relaxed_softmax(x.logits)
        return self.decode(data)

    def counterfactual_decode(
        self,
        data: Outputs,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Create counterfactual embeddings."""
        target = self.get_target(data.probs, target_dim)
        data.probs = self.interpolate_probs(data.probs, target, target_value)
        return self.decode(data)

    @torch.inference_mode()
    def sample_latent(
        self,
        z1_bias: torch.Tensor,
        batch_size: int = 1,
        probs: torch.Tensor | None = None,
    ) -> Outputs:
        """Generate latent space variables with optional target probabilities."""
        data = Outputs()
        data.z1 = self._sample_z1(z1_bias, batch_size)
        if probs is not None and probs.numel() > 0:
            data.probs = probs.to(z1_bias.device)
        else:
            alpha = torch.ones(self.n_classes, device=z1_bias.device)
            data.probs = torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))

        return data

    def _compute_z2(self, data: Outputs) -> torch.Tensor:
        """Compute z2 latent variable."""
        data.p_mu2, data.p_log_var2 = self.z2_inference(data.probs).chunk(2, 2)
        if not hasattr(data, 'h'):
            return self.sampler.sample(data.p_mu2, data.p_log_var2)

        data.d_mu2, data.d_log_var2 = self.posterior(data.probs, data.h).chunk(2, 2)
        mu_combined = data.d_mu2 + data.p_mu2
        log_var_combined = data.d_log_var2 + data.p_log_var2
        return self.sampler.sample(mu_combined, log_var_combined)

    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Create equal probability distribution for each class."""
        if logits.numel() == 0:
            raise ValueError('No logits provided.')

        return self.relaxed_softmax(logits)

    @staticmethod
    def get_target(probs: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Get one-hot encoding for target."""
        target = torch.zeros_like(probs)
        target[:, target_dim] = 1
        return target

    @staticmethod
    def interpolate_probs(probs: torch.Tensor, target: torch.Tensor, target_value: float) -> torch.Tensor:
        """Interpolate latent variables."""
        return (1 - target_value) * probs + target_value * target
