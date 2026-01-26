"""W-autoencoder architecture."""

import abc
from collections.abc import Callable, Generator
from typing import override

import numpy as np
import torch
from torch import nn as nn

from src.config import Experiment
from src.data.structures import Outputs, WInputs
from src.module.w_decoders import get_w_decoder
from src.module.w_conditional import ConditionalPrior, get_conditional_w_encoder
from src.module.w_encoders import get_w_encoder
from src.module.layers import reset_child_params, TemperatureScaledSoftmax
from src.module.quantize import VectorQuantizer


class PseudoInputManager(torch.nn.Module):
    """Manages pseudo inputs and their latent representations."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_wae_model = cfg.w_autoencoder.model
        self.n_codes: int = cfg_ae_model.n_codes
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.n_pseudo_inputs: int = cfg_wae_model.n_pseudo_inputs
        self.z1_dim: int = cfg_wae_model.z1_dim
        self.pseudo_inputs = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.n_codes, self.embedding_dim))
        self.pseudo_mu = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.n_codes, self.z1_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.n_codes, self.z1_dim))
        self.initialize_parameters()
        return

    def initialize_parameters(self):
        """Initialize pseudo input parameters."""
        nn.init.normal_(self.pseudo_inputs)
        nn.init.normal_(self.pseudo_mu)
        nn.init.normal_(self.pseudo_log_var)
        return

    def update_pseudo_latent(self, encoder_func: Callable[[None], Outputs]) -> None:
        """Update pseudo latent parameters based on encoder output."""
        pseudo_data = encoder_func(None)
        self.pseudo_mu.data = pseudo_data.pseudo_mu1
        self.pseudo_log_var.data = pseudo_data.pseudo_log_var1
        return

    def get_combined_input(self, x: torch.Tensor | None) -> torch.Tensor:
        """Combine input with pseudo inputs."""
        if x is None:
            return self.pseudo_inputs

        return torch.cat((x, self.pseudo_inputs))

    def sample_pseudo_latent(self, batch_size: int) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        """Sample pseudo latents."""
        for _ in range(batch_size):
            i = np.random.randint(self.n_pseudo_inputs)
            yield self.pseudo_mu[i], self.pseudo_log_var[i]

        return


class BaseWAutoEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for W autoencoders with common settings and interface."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        cfg_wae_model = cfg.w_autoencoder.model
        self.n_codes: int = cfg_ae_model.n_codes
        self.book_size: int = cfg_ae_model.book_size
        self.embedding_dim: int = cfg_ae_model.embedding_dim
        self.z1_dim: int = cfg_wae_model.z1_dim
        self.n_classes: int = cfg.data.dataset.n_classes
        self.n_pseudo_inputs: int = cfg_wae_model.n_pseudo_inputs
        self._codebook: torch.Tensor | None = None
        self.quantizer = VectorQuantizer()
        self.pseudo_manager = PseudoInputManager() if self.n_pseudo_inputs > 0 else None
        return

    @property
    def codebook(self) -> torch.Tensor:
        """Codebook tensor."""
        if self._codebook is None:
            raise RuntimeError('Codebook not initialized.')

        return self._codebook

    @abc.abstractmethod
    def forward(self, inputs: WInputs, stochastic: bool = True) -> Outputs:
        """Forward pass."""

    @abc.abstractmethod
    def generate_discrete_latent_space(
        self, z1_bias: torch.Tensor, batch_size: int = 1, probs: torch.Tensor | None = None
    ) -> Outputs:
        """Generate discrete latent space samples from the continuous prior distribution."""

    def recursive_reset_parameters(self):
        """Reset all parameters."""
        if self.pseudo_manager:
            self.pseudo_manager.initialize_parameters()

        reset_child_params(self)
        return

    def update_codebook(self, codebook: torch.Tensor) -> None:
        """Update codebook tensor."""
        del self._codebook

        self.register_buffer('_codebook', codebook.data, persistent=False)
        return

    def _get_input(self, x: torch.Tensor | None) -> torch.Tensor:
        """Get input tensor, combining with pseudo inputs if available."""
        if self.pseudo_manager is None:
            if x is None:
                raise ValueError('No input available.')

            return x

        return self.pseudo_manager.get_combined_input(x)

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Gaussian sample given mean and variance."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class WAutoEncoder(BaseWAutoEncoder):
    """Standard W autoencoder."""

    def __init__(self):
        super().__init__()
        self.encoder = get_w_encoder()
        self.decoder = get_w_decoder()
        self.z2_prior = ConditionalPrior()
        self.z2_posterior = get_conditional_w_encoder()
        return

    def forward(self, inputs: WInputs, stochastic: bool = True) -> Outputs:
        """Forward pass."""
        x = inputs.w_q.view(-1, self.n_codes, self.embedding_dim)
        data = self.encode_z1(x)
        data = self.get_probabilities(inputs, data)
        data = self.encode_z2(x, data)
        data = self.sample_posterior(data) if stochastic else self.assign_mean(data)
        return self.decode(data)

    def encode_z1(self, x: torch.Tensor | None) -> Outputs:
        """Encode from discrete to continuous latent space."""
        data = Outputs()
        input_tensor = self._get_input(x)
        latent = self.encoder(input_tensor)
        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index)
            data.pseudo_mu1, data.pseudo_log_var1 = pseudo_latent.chunk(2, 2)

        data.mu1, data.log_var1 = latent.chunk(2, 2)
        return data

    def encode_z2(self, x: torch.Tensor, data: Outputs) -> Outputs:
        """Encode from discrete to continuous latent space conditioned on classifier probabilities."""
        data.p_mu2, data.p_log_var2 = self.z2_prior(data.probs).chunk(2, 2)
        data.d_mu2, data.d_log_var2 = self.z2_posterior(data.probs, x).chunk(2, 2)
        return data

    def sample_posterior(self, data: Outputs) -> Outputs:
        """Sample from posterior distribution."""
        data.z1 = self.sample_gaussian(data.mu1, data.log_var1)
        mu2_combined = data.d_mu2 + data.p_mu2
        log_var2_combined = data.d_log_var2 + data.p_log_var2
        data.z2 = self.sample_gaussian(mu2_combined, log_var2_combined)
        return data

    def decode(self, data: Outputs) -> Outputs:
        """Decode from continuous to discrete latent space."""
        data.w_recon = self.decoder(data.z1, data.z2)
        _, data.idx, data.w_dist_2 = self.quantizer.quantize(data.w_recon, self.codebook)
        return data

    @torch.inference_mode()
    def generate_discrete_latent_space(
        self, z1_bias: torch.Tensor, batch_size: int = 1, probs: torch.Tensor | None = None
    ) -> Outputs:
        """Generate discrete latent space samples from the continuous prior distribution."""
        data = Outputs()
        self.eval()
        data.z1 = self.sample_z1_prior() + z1_bias
        data.probs = probs.to(self.codebook.device) if probs is not None else self.sample_prob(batch_size)
        data.z2 = self.sample_z2_prior(data.probs)
        return self.decode(data)

    def get_probabilities(self, inputs: WInputs, data: Outputs) -> Outputs:
        """Get probabilities for the forward pass."""
        data.probs = self.get_uniform_probabilities(data.z1.shape[0])
        return data

    def get_uniform_probabilities(self, batch_size: int) -> torch.Tensor:
        """Get uniform probabilities over classes."""
        return torch.ones(batch_size, self.n_classes, device=self.codebook.device) / self.n_classes

    def sample_z1_prior(self, batch_size: int = 1) -> torch.Tensor:
        """Sample z1 from the prior distribution."""
        if self.pseudo_manager is not None and self.n_pseudo_inputs > 0:
            self.pseudo_manager.update_pseudo_latent(self.encode_z1)
            pseudo_z_list = []
            for mu, log_var in self.pseudo_manager.sample_pseudo_latent(batch_size):
                pseudo_z_list.append(self.sample_gaussian(mu, log_var))

            return torch.stack(pseudo_z_list).contiguous()

        return torch.randn((batch_size, 1, self.z1_dim), device=self.codebook.device)

    def sample_prob(self, batch_size: int = 1) -> torch.Tensor:
        """Sample a probability vector."""
        return self.get_uniform_probabilities(batch_size)

    def sample_z2_prior(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample z2 from the prior distribution."""
        mu2_combined, log_var2_combined = self.z2_prior(probs).chunk(2, 2)
        return self.sample_gaussian(mu2_combined, log_var2_combined)

    @staticmethod
    def assign_mean(data: Outputs) -> Outputs:
        """Sample z2 from the prior distribution."""
        data.z1 = data.mu1
        data.z2 = data.p_mu2 + data.d_mu2
        return data


class CounterfactualWAutoEncoder(WAutoEncoder):
    """W autoencoder with condition to classifier logits."""

    def __init__(self):
        super().__init__()
        cfg_wae_model = Experiment.get_config().w_autoencoder.model
        self.relaxed_softmax = TemperatureScaledSoftmax(dim=1, temperature=cfg_wae_model.cf_temperature)
        return

    def generate_counterfactual(
        self,
        inputs: WInputs,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Create counterfactual discrete variables."""
        x = inputs.w_q.view(-1, self.n_codes, self.embedding_dim)
        data = self.encode_z1(x)
        old_probs = self.get_probabilities_from_logits(inputs.logits)
        target = self.get_target(old_probs, target_dim)
        data.probs = self.interpolate_probs(old_probs, target, target_value)
        data = self.encode_z2(x, data)
        data = self.assign_mean(data)  # no sampling to keep fidelity with input and consistency during interpolation
        return self.decode(data)

    @override
    def get_probabilities(self, inputs: WInputs, data: Outputs) -> Outputs:
        data.probs = self.get_probabilities_from_logits(inputs.logits)
        return data

    def get_probabilities_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Get probabilities from classifier logits."""
        return self.relaxed_softmax(logits)

    @override
    def sample_prob(self, batch_size: int = 1) -> torch.Tensor:
        alpha = torch.ones(self.n_classes, device=self.codebook.device)
        return torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))

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
