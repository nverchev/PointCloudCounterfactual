"""Autoencoder architecture."""

import abc
import math
from collections.abc import Callable, Generator
from typing import override

import numpy as np
import torch
import torch.nn as nn

from src.config import Experiment
from src.config.options import AutoEncoders
from src.data.structures import Inputs, Outputs
from src.module.decoders import get_decoder, get_latent_decoder
from src.module.encoders import get_encoder, get_latent_encoder, get_conditional_latent_encoder, ConditionalPrior
from src.module.layers import TemperatureScaledSoftmax


class PseudoInputManager(nn.Module):
    """Manages pseudo inputs and their latent representations."""

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model
        self.n_codes: int = cfg_ae_model.n_codes
        self.proj_dim: int = cfg_ae_model.encoder.proj_dim
        self.n_pseudo_inputs: int = cfg_ae_model.n_pseudo_inputs
        self.z1_dim: int = cfg_ae_model.z1_dim

        # Pseudo inputs matching the encoder output dimension (flattened codes * proj_dim)
        self.pseudo_inputs = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.n_codes * self.proj_dim))

        # Pseudo latent parameters for z1 [n_pseudo, n_codes, z1_dim]
        self.pseudo_mu = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.n_codes, self.z1_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(self.n_pseudo_inputs, self.n_codes, self.z1_dim))
        self.initialize_inputs()
        return

    def initialize_inputs(self):
        """Initialize parameters."""
        nn.init.normal_(self.pseudo_inputs, std=1 / math.sqrt(self.proj_dim))
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

    def update_pseudo_latent(self, encoder_func: Callable[[None], Outputs]) -> None:
        """Update pseudo latent parameters based on encoder output."""
        pseudo_out = encoder_func(None)
        self.pseudo_mu.data = pseudo_out.pseudo_mu1
        self.pseudo_log_var.data = pseudo_out.pseudo_log_var1
        return


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
        x = self.decoder(out.latent_features, self.n_output_points, inputs.initial_sampling)
        out.recon = x.transpose(2, 1).contiguous()
        return out


class BaseVAE(BaseAutoencoder):
    """Base VAE class with integrated Latent-Autoencoder logic."""

    _null_tensor = torch.empty(0)
    _zero_tensor = torch.tensor(0.0)

    def __init__(self):
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae_model = cfg.autoencoder.model

        self.n_classes: int = cfg.data.dataset.n_classes
        self.z1_dim: int = cfg_ae_model.z1_dim
        self.z2_dim: int = cfg_ae_model.z2_dim
        self.n_codes: int = cfg_ae_model.n_codes
        self.n_pseudo_inputs: int = cfg_ae_model.n_pseudo_inputs

        self.latent_encoder = get_latent_encoder()
        self.latent_decoder = get_latent_decoder()
        self.z2_prior = ConditionalPrior()
        self.z2_posterior = get_conditional_latent_encoder()

        self.pseudo_manager = PseudoInputManager() if self.n_pseudo_inputs > 0 else None
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

    def encode(self, inputs: Inputs) -> Outputs:
        """Encode point cloud to VAE latents."""
        out = Outputs()
        # Initial point encoding
        features = self.encoder(inputs.cloud)  # [Batch, n_codes * proj_dim]

        # Latent-Encoder logic (z1)
        out = self.encode_z1(features)

        # Get probabilities (for z2 conditioning)
        out = self.get_probabilities(inputs, out)

        # Z2 encoding
        out = self.encode_z2(features, out)

        # Sampling
        out = self.sample_posterior(out)

        return out

    def decode(self, out: Outputs, inputs: Inputs) -> Outputs:
        """Decode VAE latents to point cloud."""
        # Combine z1 and z2 using Latent-Decoder to get 'latent_features' (uncompressed w)
        out.latent_features = self.latent_decoder(out.z1, out.z2)

        # Decode latent_features using point decoder
        return super().decode(out, inputs)

    def encode_z1(self, x: torch.Tensor | None) -> Outputs:
        """Encode from dense features to z1."""
        out = Outputs()
        input_tensor = self._get_input(x)
        latent = self.latent_encoder(input_tensor)  # [Batch, n_codes * 2*z1_dim]

        # Reshape to 3D [Batch, n_codes, 2*z1_dim]
        batch_size = latent.shape[0]
        latent = latent.view(batch_size, self.n_codes, -1)

        if self.pseudo_manager is not None and x is not None:
            # Handle pseudo inputs batch splitting
            split_index = [-self.pseudo_manager.n_pseudo_inputs]
            latent, pseudo_latent = torch.tensor_split(latent, split_index, dim=0)
            out.pseudo_mu1, out.pseudo_log_var1 = pseudo_latent.chunk(2, 2)  # Chunk on dim 2 (features)
        elif self.pseudo_manager is not None and x is None:
            # Only pseudo inputs update
            out.pseudo_mu1, out.pseudo_log_var1 = latent.chunk(2, 2)
            return out

        out.mu1, out.log_var1 = latent.chunk(2, 2)
        return out

    def encode_z2(self, x: torch.Tensor, out: Outputs) -> Outputs:
        """Encode from dense features to z2, conditioned on probabilities."""
        # Prior and posterior return flattened [Batch, n_codes * 2 * z2_dim]
        # Reshape to [Batch, n_codes, 2 * z2_dim] then chunk
        p_latent = self.z2_prior(out.probs).view(out.probs.shape[0], self.n_codes, -1)
        d_latent = self.z2_posterior(out.probs, x).view(x.shape[0], self.n_codes, -1)

        out.p_mu2, out.p_log_var2 = p_latent.chunk(2, 2)
        out.d_mu2, out.d_log_var2 = d_latent.chunk(2, 2)
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
        out.probs = self.get_uniform_probabilities(out.z1.shape[0], out.z1.device)
        return out

    def get_uniform_probabilities(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get uniform probabilities over classes."""
        return torch.ones(batch_size, self.n_classes, device=device) / self.n_classes

    @torch.inference_mode()
    def generate(
        self,
        batch_size: int = 1,
        initial_sampling: torch.Tensor = _null_tensor,
        z1_bias: torch.Tensor = _zero_tensor,
        probs: torch.Tensor | None = None,
    ) -> Outputs:
        """Generate samples from the model."""
        device = next(self.parameters()).device
        if initial_sampling.numel():
            initial_sampling = initial_sampling.to(device)
        inputs = Inputs(self._null_tensor, initial_sampling)
        z1_bias = z1_bias.to(device)

        out = Outputs()
        out.z1 = self.sample_z1_prior(batch_size, device) + z1_bias
        out.probs = probs.to(device) if probs is not None else self.sample_prob(batch_size, device)
        out.z2 = self.sample_z2_prior(out.probs)

        return self.decode(out, inputs)

    def sample_z1_prior(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Sample z1 from the prior distribution."""
        if device is None:
            device = torch.device('cpu')

        if self.pseudo_manager is not None and self.n_pseudo_inputs > 0:
            self.pseudo_manager.update_pseudo_latent(self.encode_z1)
            pseudo_z_list = []
            for mu, log_var in self.pseudo_manager.sample_pseudo_latent(batch_size):
                pseudo_z_list.append(self.sample_gaussian(mu, log_var))
            return torch.stack(pseudo_z_list).contiguous()

        return torch.randn((batch_size, self.n_codes, self.z1_dim), device=device)

    def sample_prob(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        """Sample a probability vector."""
        if device is None:
            device = torch.device('cpu')

        return self.get_uniform_probabilities(batch_size, device)

    def sample_z2_prior(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample z2 from the prior distribution."""
        mu2_combined, log_var2_combined = self.z2_prior(probs).chunk(2, 1)
        return self.sample_gaussian(mu2_combined, log_var2_combined)


class VAE(BaseVAE):
    """Standard VAE implementation."""

    @override
    def get_probabilities(self, inputs: Inputs, out: Outputs) -> Outputs:
        out.probs = self.get_uniform_probabilities(out.z1.shape[0], out.z1.device)
        return out


class CounterfactualVAE(BaseVAE):
    """Counterfactual VAE implementation."""

    def __init__(self):
        super().__init__()
        cfg_ae_model = Experiment.get_config().autoencoder.model
        self.relaxed_softmax = TemperatureScaledSoftmax(dim=1, temperature=cfg_ae_model.cf_temperature)
        return

    @override
    def get_probabilities(self, inputs: Inputs, out: Outputs) -> Outputs:
        # Check if logits are provided and not empty
        if hasattr(inputs, 'logits') and inputs.logits.numel() > 0:
            out.probs = self.get_probabilities_from_logits(inputs.logits)
        else:
            # Fallback to uniform if no logits (e.g. inference without logits or non-CF training)
            # Note: For CF training, logits should be provided in Inputs.
            out.probs = self.get_uniform_probabilities(out.z1.shape[0], out.z1.device)
        return out

    def get_probabilities_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Get probabilities from classifier logits."""
        return self.relaxed_softmax(logits)

    @override
    def sample_prob(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = torch.device('cpu')

        alpha = torch.ones(self.n_classes, device=device)
        return torch.distributions.Dirichlet(concentration=alpha).sample((batch_size,))

    @torch.inference_mode()
    def generate_counterfactual(
        self,
        inputs: Inputs,
        sample_logits: torch.Tensor,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Generate counterfactual samples."""
        # Encode
        # Encode
        out = Outputs()
        features = self.encoder(inputs.cloud)
        out = self.encode_z1(features)

        # Probabilities interpolation
        old_probs = self.get_probabilities_from_logits(sample_logits)
        target = self.get_target(old_probs, target_dim)
        out.probs = self.interpolate_probs(old_probs, target, target_value)

        # Encode z2 (using interpolated probs)
        out = self.encode_z2(features, out)

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


def get_autoencoder() -> AbstractAutoEncoder:
    """Get the correct autoencoder according to the configuration."""
    model_registry = {
        AutoEncoders.AE: BaseAutoencoder,
        AutoEncoders.VAE: VAE,
        AutoEncoders.CounterfactualVAE: CounterfactualVAE,
    }
    return model_registry[Experiment.get_config().autoencoder.model.class_name]()
