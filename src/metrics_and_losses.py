"""Module containing metrics and losses used for training and evaluation."""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from drytorch.lib.objectives import Metric, Loss, LossBase

# from emd import emdModule
from structural_losses import match_cost
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

from src.neighbour_ops import pykeops_square_distance, torch_square_distance
from src.data_structures import Outputs, Targets, W_Targets
from src.config_options import ReconLosses, ModelHead, Experiment


def pykeops_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """The following code is currently not supported for backprop:

        ```python
        def pykeops_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
                    dist = pykeops_square_distance(t1, t2)
                    return (dist.min(axis = 2) + dist.min(axis = 1)).sum(axis=(1, 2)
        ```

    We use the retrieved index on torch """
    dist = pykeops_square_distance(t1, t2)
    idx1 = dist.argmin(axis=1).expand(-1, -1, t1.shape[2])
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).sum(2).mean(1)
    idx2 = dist.argmin(axis=2).expand(-1, -1, t1.shape[2])
    m2 = t2.gather(1, idx2)
    squared2 = ((t1 - m2) ** 2).sum(2).mean(1)
    # forward + reverse
    squared = squared1 + squared2
    return squared


def torch_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Calculate Chamfer distance between two point clouds using PyTorch backend."""
    dist = torch_square_distance(t1, t2)
    return torch.min(dist, dim=-1)[0].sum(1) + torch.min(dist, dim=-2)[0].sum(1)


def get_emd_loss() -> LossBase[Outputs, Targets]:
    """Calculate earthmover's distance between two point clouds using PyTorch backend."""
    # emd_dist = emdModule()

    def _emd(data: Outputs, targets: Targets) -> torch.Tensor:
        # return torch.sqrt(emd_dist(data.recon.contiguous(), targets.ref_cloud.contiguous(), 0.005, 50)[0]).sum(1)
        return match_cost(data.recon, targets.ref_cloud)

    return Loss(_emd, name='EMD')


def get_chamfer_loss() -> LossBase[Outputs, Targets]:
    """Calculate Chamfer distance between two point clouds using PyTorch backend."""
    cfg_user = Experiment.get_config().user
    device = cfg_user.device

    chamfer_backend = pykeops_chamfer if device.type == 'cuda' else torch_chamfer

    def _chamfer(data: Outputs, targets: Targets) -> torch.Tensor:
        return chamfer_backend(data.recon, targets.ref_cloud)

    return Loss(_chamfer, name='Chamfer')


def get_recon_loss() -> LossBase[Outputs, Targets]:
    """Calculate reconstruction loss based on configuration settings."""
    cfg = Experiment.get_config().autoencoder
    cfg_user = Experiment.get_config().user
    recon_loss = cfg.objective.recon_loss
    device = cfg_user.device
    chamfer_loss = get_chamfer_loss()
    if recon_loss == ReconLosses.ChamferEMD and device.type == 'cuda':
        return chamfer_loss + get_emd_loss()
    return chamfer_loss


def get_embed_loss() -> Loss[Outputs, Targets]:
    """Calculate mean squared error between quantized (w_q) and encoded (w_e) embeddings."""
    mse_loss = nn.MSELoss(reduction='none')

    def _embed_loss(data: Outputs, _not_used: Targets) -> torch.Tensor:
        return mse_loss(data.w_q, data.w_e).mean(dim=1)

    embed_loss = Loss(_embed_loss, name='Embed. Loss')
    return embed_loss


def gaussian_ll(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Calculate the Gaussian log-likelihood."""
    return -0.5 * (log_var + torch.pow(x - mu, 2) / torch.exp(log_var)) + np.log(2 * math.pi)


def gaussian_kld(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence between Gaussian distributions."""
    return 0.5 * (-1 - log_var + log_var.exp() + (mu ** 2))


def diff_gaussian_kld(d_mu: torch.Tensor, d_log_var: torch.Tensor, p_log_var: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence between two Gaussian distributions with different parameters."""
    return 0.5 * (-1 - d_log_var + d_log_var.exp() + (d_mu ** 2) / p_log_var.exp())


def get_kld_loss() -> LossBase[Outputs, W_Targets]:
    """Get KL divergence loss for variational autoencoder."""
    cfg_w_ae = Experiment.get_config().w_autoencoder

    def _kld1(data: Outputs, _: W_Targets) -> torch.Tensor:
        return diff_gaussian_kld(d_mu=data.d_mu2, d_log_var=data.d_log_var2, p_log_var=data.p_log_var2).sum(1)

    def _kld2(data: Outputs, _: W_Targets) -> torch.Tensor:
        return gaussian_kld(mu=data.mu1, log_var=data.log_var1).sum(1)

    def _kld2_vamp(data: Outputs, _: W_Targets) -> torch.Tensor:
        """Calculate KL divergence loss for VAMP prior."""
        z_kld = data.z1
        mu_kld = data.mu1
        log_var_kld = data.log_var1
        pseudo_mu = data.pseudo_mu1
        pseudo_log_var = data.pseudo_log_var1

        k = pseudo_mu.shape[0]
        b = mu_kld.shape[0]
        z = z_kld.unsqueeze(1).expand(-1, k, -1)  # create a copy for each pseudo input
        posterior_ll = gaussian_ll(z_kld, mu_kld, log_var_kld).sum(1)  # sum dimensions
        pseudo_mu = pseudo_mu.unsqueeze(0).expand(b, -1, -1)  # expand to match the batch size
        pseudo_log_var = pseudo_log_var.unsqueeze(0).expand(b, -1, -1)  # expand to match the batch size
        prior_ll = torch.logsumexp(gaussian_ll(z, pseudo_mu, pseudo_log_var).sum(2), dim=1)
        total = posterior_ll - prior_ll + np.log(k)
        return total

    def _combined(data: Outputs, targets: W_Targets) -> torch.Tensor:
        _kld2_final = _kld2_vamp if cfg_w_ae.objective.vamp else _kld2
        return _kld1(data, targets) + _kld2_final(data, targets)

    return Loss(_combined, name='KLD')


def get_adversarial_loss() -> LossBase[Outputs, W_Targets]:
    """Get adversarial loss for training."""
    kld = nn.KLDivLoss(reduction='none', log_target=True)

    def _corr(data: Outputs, targets: W_Targets) -> torch.Tensor:
        return kld(F.log_softmax(data.y1, dim=1), F.log_softmax(targets.logits, dim=1)).sum(1)

    def _max_entropy(data: Outputs, _: W_Targets) -> torch.Tensor:
        return -torch.sum(F.softmax(data.y2, dim=1) * F.log_softmax(data.y2, dim=1), dim=1)

    return Loss(_corr, name='Correlation') + Loss(_max_entropy, name='Max entropy')


def get_nll_loss() -> LossBase[Outputs, W_Targets]:
    """Get negative log likelihood loss."""
    log_softmax = torch.nn.LogSoftmax(dim=2)

    def _nll(data: Outputs, targets: W_Targets) -> torch.Tensor:
        sqrt_dist = torch.sqrt(data.w_dist)
        w_neg_dist = -sqrt_dist + sqrt_dist.min(2, keepdim=True)[0].detach()  # second term for numerical stability
        nll = (-log_softmax(w_neg_dist) * targets.one_hot_idx).sum((1, 2))
        return nll

    return Loss(_nll, name='NLL')


def get_w_accuracy() -> Metric[Outputs, W_Targets]:
    """Get accuracy metric for quantization."""

    def _accuracy(data: Outputs, targets: W_Targets) -> torch.Tensor:
        one_hot_predictions = F.one_hot(data.w_dist.argmin(2), num_classes=targets.one_hot_idx.shape[2])
        return (targets.one_hot_idx * one_hot_predictions).sum(2).mean(1)

    return Metric(_accuracy, name='Quantisation Accuracy', higher_is_better=True)


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


def get_w_encoder_loss() -> LossBase[Outputs, W_Targets]:
    """Get encoder loss combining NLL, KLD and adversarial losses."""
    c_kld = Experiment.get_config().w_autoencoder.objective.c_kld
    c_adv = Experiment.get_config().w_autoencoder.objective.c_counterfactual
    loss = get_nll_loss() + c_kld * get_kld_loss() + c_adv * get_adversarial_loss() | get_w_accuracy()
    return loss


def get_autoencoder_loss() -> LossBase[Outputs, Targets]:
    """Get autoencoder loss combining reconstruction and embedding losses."""
    cfg_ae = Experiment.get_config().autoencoder
    c_embed = cfg_ae.objective.c_embedding
    loss = get_recon_loss()
    if cfg_ae.model.head is not ModelHead.AE:
        return loss + c_embed * get_embed_loss()
    return loss
