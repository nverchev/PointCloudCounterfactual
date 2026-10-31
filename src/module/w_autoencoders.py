"""Inner autoencoder: the VAE of the codewords, conditioned on the evaluation of the classifier.

The latent space is partitioned into ``z_u``, with a standard Gaussian prior, and ``z_y``, whose prior
a linear layer computes from the evaluation, so that ``z_y`` isolates what drives the decision of
the classifier. The encoder reads the evaluation too and gives the shift of the posterior of ``z_y``
from that prior: evaluating both at a target evaluation in place of the original one gives the
intervention of a counterfactual. Otherwise, the architecture is the one of PCGen, without the
VampPrior.
"""

import abc

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data.structures import WInputs, WOutputs
from src.module.decoders import ConditionalGaussian, WDecoder
from src.module.encoders import WEncoder

__all__ = [
    'BaseWAutoEncoder',
    'WAutoEncoder',
]


class BaseWAutoEncoder(nn.Module, abc.ABC):
    """Abstract base of the variational autoencoders of the codewords.

    Attributes:
        w_dim: Dimension of the codeword.
        dim_codes: Number of codes of a codeword.
        embedding_dim: Dimension of each code.
        n_classes: Number of classes of the evaluations.
        z_u_dim: Dimension of the latent variable unrelated to the classification.
        z_y_dim: Dimension of the latent variable the evaluation conditions.
        z_dim: Dimension of the latent variables together.
        prior_h_dims: Hidden dimensions of the conditional prior.
    """

    w_dim: Final[int]
    dim_codes: Final[int]
    embedding_dim: Final[int]
    n_classes: Final[int]
    z_u_dim: Final[int]
    z_y_dim: Final[int]
    z_dim: Final[int]
    prior_h_dims: Final[tuple[int, ...]]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_ae = cfg.autoencoding
        cfg_w = cfg.w_autoencoding
        assert cfg_w is not None
        self.w_dim = cfg_ae.model.w_dim
        self.dim_codes = cfg_ae.model.dim_codes
        self.embedding_dim = cfg_ae.model.embedding_dim
        self.n_classes = cfg.data.dataset.n_classes
        self.z_u_dim = cfg_w.model.z_u_dim
        self.z_y_dim = cfg_w.model.z_y_dim
        self.z_dim = self.z_u_dim + self.z_y_dim
        self.prior_h_dims = cfg_w.model.architecture.prior_h_dims
        return

    @abc.abstractmethod
    def forward(self, inputs: WInputs) -> WOutputs:
        """Reconstruct the codewords under the evaluation."""


class WAutoEncoder(BaseWAutoEncoder):
    """Variational autoencoder of the codewords with a conditional prior.

    Attributes:
        encoder: Encoder of the codewords.
        prior: Conditional prior of ``z_y``.
        decoder: Decoder of the latent variables.
    """

    encoder: WEncoder
    prior: ConditionalGaussian
    decoder: WDecoder

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.encoder = WEncoder()
        self.prior = ConditionalGaussian(self.n_classes, self.prior_h_dims)
        self.decoder = WDecoder()
        return

    @override
    def forward(self, inputs: WInputs) -> WOutputs:
        """Reconstruct the codewords under the evaluation.

        In evaluation mode, the latent variables are the means of their posteriors, so that no
        sampling noise is introduced.

        Args:
            inputs: codewords and evaluation, a point of the probability simplex.

        Returns:
            The reconstructed codewords and the distributions behind them.
        """
        d_mu_y, d_log_var_y, mu_u, log_var_u = self.encoder(inputs.w_q, inputs.condition)
        p_mu_y, p_log_var_y = self.prior(inputs.condition)
        z_y = self._sample(p_mu_y + d_mu_y, p_log_var_y + d_log_var_y)
        z_u = self._sample(mu_u, log_var_u)
        w_recon = self.decode(torch.cat((z_y, z_u), dim=1))
        return WOutputs(w_recon, mu_u, log_var_u, d_mu_y, d_log_var_y, p_log_var_y)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent variables into the reconstructed codeword."""
        return self.decoder(z)

    def _sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
