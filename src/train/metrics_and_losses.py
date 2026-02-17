"""Metrics and losses used for training and evaluation."""

import math

import numpy as np
import torch

from structural_losses import match_cost
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

from drytorch.lib.objectives import Loss, LossBase, Metric

from src.config.experiment import Experiment
from src.config.options import AutoEncoders, ReconLosses
from src.data.structures import Outputs, Targets
from src.utils.neighbour_ops import pykeops_square_distance, torch_square_distance


def pykeops_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """The following code is currently not supported for backprop:

        ```python
        def pykeops_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
                    dist = pykeops_square_distance(t1, t2)
                    return (dist.min(axis = 2) + dist.min(axis = 1)).sum(axis=(1, 2)
        ```

    We use the retrieved index on torch
    """
    dist = pykeops_square_distance(t1, t2)

    # match t1 to t2
    idx1 = dist.argmin(axis=1).expand(-1, -1, t1.shape[2])
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).sum(2).mean(1)

    # match t2 to t1
    idx2 = dist.argmin(axis=2).expand(-1, -1, t2.shape[2])
    m2 = t2.gather(1, idx2)
    squared2 = ((t1 - m2) ** 2).sum(2).mean(1)

    # sum squared distances
    squared = squared1 + squared2
    return squared


def torch_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Calculate Chamfer distance between two point clouds using PyTorch backend."""
    dist = torch_square_distance(t1, t2)
    return torch.min(dist, dim=-1)[0].sum(1) + torch.min(dist, dim=-2)[0].sum(1)


def get_emd_loss() -> LossBase[Outputs, Targets]:
    """Calculate earthmover's distance between two point clouds using PyTorch backend."""

    def _emd(out: Outputs, targets: Targets) -> torch.Tensor:
        return match_cost(out.recon, targets.ref_cloud)

    return Loss(_emd, name='EMD')


def get_chamfer_loss() -> LossBase[Outputs, Targets]:
    """Calculate Chamfer distance between two point clouds using PyTorch backend."""
    cfg = Experiment.get_config()
    chamfer_backend = pykeops_chamfer if torch.cuda.is_available() and not cfg.user.cpu else torch_chamfer

    def _chamfer(out: Outputs, targets: Targets) -> torch.Tensor:
        return chamfer_backend(out.recon, targets.ref_cloud)

    return Loss(_chamfer, name='Chamfer')


def get_recon_loss() -> LossBase[Outputs, Targets]:
    """Calculate reconstruction loss based on configuration settings."""
    cfg = Experiment.get_config()
    cfg_autoencoder = cfg.autoencoder
    recon_loss = cfg_autoencoder.objective.recon_loss

    if recon_loss == ReconLosses.ChamferEMD and torch.cuda.is_available() and not cfg.user.cpu:
        return get_chamfer_loss() + get_emd_loss()

    return get_chamfer_loss()


def gaussian_ll(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Calculate the Gaussian log-likelihood."""
    return -0.5 * (log_var + torch.pow(x - mu, 2) / torch.exp(log_var)) + np.log(2 * math.pi)


def gaussian_kld(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence between Gaussian distributions."""
    return 0.5 * (-1 - log_var + log_var.exp() + (mu**2))


def diff_gaussian_kld(d_mu: torch.Tensor, d_log_var: torch.Tensor, p_log_var: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence between two Gaussian distributions with different parameters."""
    return 0.5 * (-1 - d_log_var + d_log_var.exp() + (d_mu**2) / p_log_var.exp())


def get_kld1_loss() -> LossBase[Outputs, Targets]:
    """Get KL divergence loss for the first latent variable in the variational autoencoder."""

    def _kld1(out: Outputs, _: Targets) -> torch.Tensor:
        return gaussian_kld(mu=out.mu1, log_var=out.log_var1).sum((1, 2))

    return Loss(_kld1, name='KLD1')


def get_kld2_loss() -> LossBase[Outputs, Targets]:
    """Get KL divergence loss for the second latent variable in the variational autoencoder."""

    def _kld2(out: Outputs, _: Targets) -> torch.Tensor:
        return diff_gaussian_kld(d_mu=out.d_mu2, d_log_var=out.d_log_var2, p_log_var=out.p_log_var2).sum((1, 2))

    return Loss(_kld2, name='KLD2')


def get_kld_vamp_loss() -> LossBase[Outputs, Targets]:
    """Get KL divergence loss for the variational autoencoder from aggregated posterior."""
    cfg = Experiment.get_config()
    n_pseudo_inputs = cfg.autoencoder.model.n_pseudo_inputs

    def _kld2_vamp(out: Outputs, _: Targets) -> torch.Tensor:
        """Calculate KL divergence loss for VAMP prior."""
        z_kld = out.z1
        mu_kld = out.mu1
        log_var_kld = out.log_var1
        pseudo_mu = out.pseudo_mu1
        pseudo_log_var = out.pseudo_log_var1
        batch = mu_kld.shape[0]
        # z1 is [Batch, n_codes, Dim]. Pseudo is [Pseudo, n_codes, Dim].
        # Expand z for pseudo: [Batch, Pseudo, n_codes, Dim]
        z = z_kld.unsqueeze(1).expand(-1, n_pseudo_inputs, -1, -1)

        # Posterior ll: sum over dim (1, 2).
        posterior_ll = gaussian_ll(z_kld, mu_kld, log_var_kld).sum((1, 2))

        # Prior ll:
        # pseudo_mu: [Pseudo, n_codes, Dim] -> [1, Pseudo, n_codes, Dim] -> [Batch, Pseudo, n_codes, Dim]
        pseudo_mu = pseudo_mu.unsqueeze(0).expand(batch, -1, -1, -1)
        pseudo_log_var = pseudo_log_var.unsqueeze(0).expand(batch, -1, -1, -1)

        # ll(z|pseudo): sum over dim (2, 3). [Batch, Pseudo]
        # logsumexp over pseudo (1).
        prior_ll = torch.logsumexp(gaussian_ll(z, pseudo_mu, pseudo_log_var).sum((2, 3)), dim=1)
        total = posterior_ll - prior_ll + np.log(n_pseudo_inputs)
        return total

    return Loss(_kld2_vamp, name='KLD2_VAMP')


def get_annealing() -> LossBase[Outputs, Targets]:
    """(Reverse) Annealing component for loss."""
    period = Experiment.get_config().autoencoder.objective.kld_restart_interval

    def _annealing(outputs: Outputs, _: Targets) -> torch.Tensor:
        t_epoch = outputs.model_epoch % period
        time_fraction = torch.tensor(t_epoch / period, device=outputs.recon.device)
        time_fraction = torch.clamp(time_fraction, 0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(time_fraction * math.pi))

    return Loss(_annealing, name='Annealing')


def get_kld_loss() -> LossBase[Outputs, Targets]:
    """Get KL divergence loss for the first latent variable in the variational autoencoder."""
    cfg_ae = Experiment.get_config().autoencoder
    # Access n_pseudo_inputs from model or config? It's in AutoEncoderConfig now.
    vamp = cfg_ae.model.n_pseudo_inputs > 0
    c_kld1 = cfg_ae.objective.c_kld1
    c_kld2 = cfg_ae.objective.c_kld2
    return get_annealing() * (c_kld1 * (get_kld_vamp_loss() if vamp else get_kld1_loss()) + c_kld2 * get_kld2_loss())


def get_cross_entropy_loss() -> LossBase[torch.Tensor, Targets]:
    """Get cross-entropy loss."""
    torch_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return torch_loss(outputs, targets.label)

    return Loss(_cross_entropy, name='CrossEntropy')


def get_accuracy() -> Metric[torch.Tensor, Targets]:
    """Get accuracy metric."""

    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return multiclass_accuracy(outputs, targets.label)

    return Metric(_cross_entropy, name='Accuracy', higher_is_better=True)


def get_macro_accuracy() -> Metric[torch.Tensor, Targets]:
    """Get macro-averaged accuracy metric."""

    def _macro_accuracy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return multiclass_accuracy(outputs, targets.label, average='macro', num_classes=outputs.shape[1])

    # triggering deprecated warning (remove this in the future if not fixed by torcheval)
    zero_output = torch.FloatTensor([[1, 0]])
    zero_tensor = torch.LongTensor([0])
    _macro_accuracy(zero_output, Targets(ref_cloud=zero_tensor, scale=zero_tensor, label=zero_tensor))

    return Metric(_macro_accuracy, name='Macro Accuracy', higher_is_better=True)


def get_f1() -> Metric[torch.Tensor, Targets]:
    """Get F1 score metric."""

    def _f1(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return multiclass_f1_score(outputs, targets.label)

    return Metric(_f1, name='F1_Score', higher_is_better=True)


def get_classification_loss() -> LossBase[torch.Tensor, Targets]:
    """Get combined classification loss and metrics."""
    return get_cross_entropy_loss() | get_accuracy() | get_macro_accuracy()


def get_autoencoder_loss() -> LossBase[Outputs, Targets]:
    """Get autoencoder loss combining reconstruction and KLD losses."""
    cfg_ae = Experiment.get_config().autoencoder
    if cfg_ae.model.class_name is not AutoEncoders.AE:
        return get_recon_loss() + get_kld_loss()

    return get_recon_loss()
