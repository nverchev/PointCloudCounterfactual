"""Outer and inner autoencoders chained into the counterfactual model."""

from typing import override

import torch.nn as nn

from src.data.structures import Inputs, Outputs, WInputs
from src.module.autoencoders import VQVAE
from src.module.w_autoencoders import WAutoEncoder

__all__ = ['CounterfactualVQVAE']


class CounterfactualVQVAE(nn.Module):
    """VQ-VAE decoding the codes that the inner VAE chooses under an evaluation.

    The outer VQ-VAE is not conditioned: the evaluation reaches the cloud only through the codes.
    With the original evaluation this is the double reconstruction, with a target evaluation it is
    the counterfactual. The two parts are trained and saved separately: this module only chains
    them, and holds no weights of its own.

    Attributes:
        autoencoder: Trained outer VQ-VAE.
        w_autoencoder: Trained inner VAE.
    """

    autoencoder: VQVAE
    w_autoencoder: WAutoEncoder

    def __init__(self, autoencoder: VQVAE, w_autoencoder: WAutoEncoder, /) -> None:
        """Initialize.

        Args:
            autoencoder: trained outer VQ-VAE.
            w_autoencoder: trained inner VAE.
        """
        super().__init__()
        self.autoencoder = autoencoder
        self.w_autoencoder = w_autoencoder
        return

    @override
    def forward(self, inputs: Inputs) -> Outputs:
        """Decode the clouds through the codes chosen under the evaluation in the inputs."""
        w_q = self.autoencoder.encode(inputs.cloud)
        w_recon = self.w_autoencoder(WInputs(w_q, inputs.condition)).w_recon
        return self.autoencoder.decode_nearest_codes(w_recon, inputs.initial_sampling)
