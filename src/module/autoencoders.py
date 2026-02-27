"""Autoencoder architecture."""

import abc
from collections.abc import Callable, Generator
from typing import override

import numpy as np
import torch
import torch.nn as nn

from src.config import Experiment
from src.config.options import AutoEncoders
from src.data.structures import Inputs, Outputs
from src.module.decoders import get_decoder
from src.module.latent_decoders import get_latent_decoder
from src.module.encoders import get_encoder
from src.module.latent_encoders import get_latent_encoder, get_conditional_latent_encoder, ConditionalPrior
from src.module.layers import TemperatureScaledSoftmax


class PseudoInputManager(nn.Module):
    """Manages pseudo inputs and their latent representations."""

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.feature_dim: int = cfg_ae_model.feature_dim
        self.n_pseudo_inputs: int = cfg_ae_model.n_pseudo_inputs
        self.z1_dim: int = cfg_ae_model.z1_dim

        # Pseudo inputs matching the encoder output dimension (flattened codes * proj_dim)
        self.pseudo_inputs = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.feature_dim))

        # Pseudo latent parameters for z1 [n_pseudo, z1_dim]
        self.pseudo_mu = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.z1_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.z1_dim))
        self.initialize_inputs()
        return

    def initialize_inputs(self) -> None:
        """Initialize parameters."""
        nn.init.normal_(self.pseudo_inputs)
        return

    def get_combined_input(self, x: torch.Tensor) -> torch.Tensor:
        """Combine input with pseudo inputs."""
        return torch.cat((x, self.pseudo_inputs))

    def sample_pseudo_latent(self, batch_size: int) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        """Sample pseudo latents."""
        for _ in range(batch_size):
            i = np.random.randint(self.n_pseudo_inputs)
            yield self.pseudo_mu[i], self.pseudo_log_var[i]

        return

    def update_pseudo_latent(self, encoder_func: Callable[[None], Outputs]) -> None:
        """Update pseudo latent parameters based on encoder output."""
        pseudo_out = encoder_func(None)
        self.pseudo_mu.data = pseudo_out.pseudo_mu1
        self.pseudo_log_var.data = pseudo_out.pseudo_log_var1
        return


class AbstractAE(nn.Module, abc.ABC):
    """Abstract autoencoder for point clouds."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae = Experiment.get_config().autoencoder
        self.n_training_points_training: int = cfg_ae.objective.n_training_output_points
        self.n_inference_output_points: int = cfg_ae.objective.n_inference_output_points
        return

    @property
    def device(self) -> torch.device:
        """Device of the model."""
        return next(self.parameters()).device

    @property
    def n_output_points(self) -> int:
        """Number of generated points."""
        return self.n_inference_output_points if torch.is_inference_mode_enabled() else self.n_training_points_training

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""


class Oracle(AbstractAE):
    """Oracle autoencoder that returns an input subset."""

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        out = Outputs()
        batch_size, n_points, _ = inputs.cloud.shape
        perm = torch.randperm(n_points, device=inputs.cloud.device)
        random_indices = perm[: self.n_output_points]
        batch_indices = torch.arange(batch_size, device=inputs.cloud.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, self.n_output_points)
        out.recon = inputs.cloud[batch_indices, random_indices]
        return out


class AE(AbstractAE):
    """Standard autoencoder for point clouds."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass."""
        out = self.encode(inputs)
        return self.decode(out, inputs)

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode point cloud to features."""
        out = Outputs()
        out.features = self.encoder(inputs.cloud)
        return out

    def decode(self, out: Outputs, inputs: Inputs) -> Outputs:
        """Decode features to point cloud."""
        initial_cloud = self._initialize_cloud(inputs.cloud.shape[0])
        x = self.decoder(initial_cloud, out.features, self.n_output_points)
        out.recon = x
        return out

    def _initialize_cloud(self, batch: int) -> torch.Tensor:
        """Initialize and normalize the sampling points."""
        return torch.randn(batch, self.n_output_points, 3, device=self.device)


class BaseVAE(AE):
    """Base VAE class with integrated Latent-Autoencoder logic."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def __init__(self) -> None:
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_classes: int = cfg.data.dataset.n_classes
        self.z1_dim: int = cfg_ae_model.z1_dim
        self.z2_dim: int = cfg_ae_model.z2_dim
        self.n_pseudo_inputs: int = cfg_ae_model.n_pseudo_inputs
        self.latent_encoder = get_latent_encoder()
        self.latent_decoder = get_latent_decoder()
        self.z2_prior = ConditionalPrior()
        self.z2_posterior = get_conditional_latent_encoder()
        self.pseudo_manager: PseudoInputManager | None = PseudoInputManager() if self.n_pseudo_inputs > 0 else None
        return

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode point cloud to latent variables."""
        out = super().encode(inputs)

        # Z1 encoding
        out = self.encode_z1(out)

        # probabilities for z2 conditioning
        out = self.get_probabilities(inputs, out)

        # Z2 encoding
        out = self.encode_z2(out)

        # sampling
        out = self.sample_posterior(out)

        return out

    def decode(self, out: Outputs, inputs: Inputs) -> Outputs:
        """Decode latent variables to point cloud."""

        # latent variables to features
        out.features = self.latent_decoder(out.z1, out.z2)

        # features to point cloud
        return super().decode(out, inputs)

    def encode_z1(self, out: Outputs | None = None) -> Outputs:
        """Encode from dense features to z1."""
        if out is None:
            input_tensor = self._get_input(out)
            out = Outputs()
        else:
            input_tensor = out.features

        latent = self.latent_encoder(input_tensor)
        if self.pseudo_manager is not None:
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index, dim=0)
            out.pseudo_mu1, out.pseudo_log_var1 = pseudo_latent.chunk(2, 1)

        out.mu1, out.log_var1 = latent.chunk(2, 1)
        return out

    def encode_z2(self, out: Outputs) -> Outputs:
        """Encode from dense features to z2, conditioned on probabilities."""
        p_latent = self.z2_prior(out.probs)
        d_latent = self.z2_posterior(out.probs, out.features)
        out.p_mu2, out.p_log_var2 = p_latent.chunk(2, 1)
        out.d_mu2, out.d_log_var2 = d_latent.chunk(2, 1)
        return out

    def sample_posterior(self, out: Outputs) -> Outputs:
        """Sample from posterior distribution."""
        out.z1 = self.sample_gaussian(out.mu1, out.log_var1)
        mu2_combined = out.d_mu2 + out.p_mu2
        log_var2_combined = out.d_log_var2 + out.p_log_var2
        out.z2 = self.sample_gaussian(mu2_combined, log_var2_combined)
        return out

    def get_probabilities(self, inputs: Inputs, out: Outputs) -> Outputs:
        """Get probabilities for the forward pass. Default is uniform."""
        out.probs = self.get_uniform_probabilities(out.mu1.shape[0])
        return out

    def get_uniform_probabilities(self, batch_size: int) -> torch.Tensor:
        """Get uniform probabilities over classes."""
        return torch.ones(batch_size, self.n_classes, device=self.device) / self.n_classes

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
        probs: torch.Tensor | None = None,
    ) -> Outputs:
        """Generate samples from the model."""
        inputs = Inputs(self._null_tensor, initial_sampling)
        z1_bias = z1_bias.to(self.device)
        out = Outputs()
        out.z1 = self.sample_z1_prior(batch_size) + z1_bias
        out.probs = probs.to(self.device) if probs is not None else self.sample_prob(batch_size)
        out.z2 = self.sample_z2_prior(out.probs)
        return self.decode(out, inputs)

    def sample_z1_prior(self, batch_size: int = 1) -> torch.Tensor:
        """Sample z1 from the prior distribution."""
        if self.pseudo_manager is not None and self.n_pseudo_inputs > 0:
            self.pseudo_manager.update_pseudo_latent(self.encode_z1)
            pseudo_z_list = []
            for mu, log_var in self.pseudo_manager.sample_pseudo_latent(batch_size):
                pseudo_z_list.append(self.sample_gaussian(mu, log_var))

            return torch.stack(pseudo_z_list).contiguous()

        return torch.randn((batch_size, self.z1_dim), device=self.device)

    def sample_prob(self, batch_size: int = 1) -> torch.Tensor:
        """Sample a probability vector."""
        return self.get_uniform_probabilities(batch_size)

    def sample_z2_prior(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample z2 from the prior distribution."""
        mu2_combined, log_var2_combined = self.z2_prior(probs).chunk(2, 1)
        return self.sample_gaussian(mu2_combined, log_var2_combined)

    def _get_input(self, out: Outputs | None = None) -> torch.Tensor:
        """Get input tensor, combining with pseudo inputs if available."""
        if self.pseudo_manager is None:
            if out is None:
                raise ValueError('No input available.')

            return out.features

        if out is None:
            return self.pseudo_manager.pseudo_inputs

        return self.pseudo_manager.get_combined_input(out.features)

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Gaussian sample given mean and variance."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class VAE(BaseVAE):
    """Standard VAE implementation."""

    @override
    def get_probabilities(self, inputs: Inputs, out: Outputs) -> Outputs:
        out.probs = self.get_uniform_probabilities(out.mu1.shape[0])
        return out


class CounterfactualVAE(BaseVAE):
    """Counterfactual VAE implementation."""

    def __init__(self) -> None:
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        temperature = cfg_ae_model.cf_temperature if cfg_ae_model.cf_temperature is not None else 1.0
        self.relaxed_softmax = TemperatureScaledSoftmax(dim=1, temperature=temperature)
        return

    @override
    def get_probabilities(self, inputs: Inputs, out: Outputs) -> Outputs:
        if inputs.logits.numel() > 0:
            out.probs = self.get_probabilities_from_logits(inputs.logits)
            return out

        return super().get_probabilities(inputs, out)

    def get_probabilities_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Get probabilities from classifier logits."""
        return self.relaxed_softmax(logits)

    @override
    def sample_prob(self, batch_size: int = 1) -> torch.Tensor:
        alpha = torch.ones(self.n_classes, device=self.device)
        return torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))

    @torch.inference_mode()
    def generate_counterfactual(
        self,
        inputs: Inputs,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Generate counterfactual samples."""
        # Encode
        out = Outputs()
        out.features = self.encoder(inputs.cloud)
        out = self.encode_z1(out)

        # Probabilities interpolation
        old_probs = self.get_probabilities_from_logits(inputs.logits)
        target = self.get_target(old_probs, target_dim)
        out.probs = self.interpolate_probs(old_probs, target, target_value)

        # Encode z2 (using interpolated probs)
        out = self.encode_z2(out)

        # Assign mean (no sampling for counterfactual)
        out.z1 = out.mu1
        out.z2 = out.p_mu2 + out.d_mu2

        return self.decode(out, inputs)

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


def get_autoencoder() -> AbstractAE:
    """Get the correct autoencoder according to the configuration."""
    model_registry = {
        AutoEncoders.AE: AE,
        AutoEncoders.VAE: VAE,
        AutoEncoders.CounterfactualVAE: CounterfactualVAE,
    }
    return model_registry[Experiment.get_config().autoencoder.model.class_name]()
