from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from dry_torch.metrics import Metric, Loss, LossBase, MetricBase

from emd import emdModule
from structural_losses import match_cost
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

from src.neighbour_ops import pykeops_square_distance, torch_square_distance
from src.data_structures import Outputs, Targets, W_Targets
from src.config_options import ExperimentAE, ExperimentWAE, ReconLosses, ModelHead, MainExperiment


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
    dist = torch_square_distance(t1, t2)
    return torch.min(dist, dim=-1)[0].sum(1) + torch.min(dist, dim=-2)[0].sum(1)


def get_emd_loss() -> LossBase[Outputs, Targets]:
    emd_dist = emdModule()

    def _emd(data: Outputs, targets: Targets) -> torch.Tensor:
        # return torch.sqrt(emd_dist(data.recon.contiguous(), targets.ref_cloud.contiguous(), 0.005, 50)[0]).sum(1)
        return match_cost(data.recon, targets.ref_cloud)

    return Loss(_emd, name='EMD')


def get_chamfer_loss() -> LossBase[Outputs, Targets]:
    cfg = ExperimentAE.get_config()
    cfg_user = MainExperiment.get_config().user
    device = cfg_user.device

    chamfer_backend = pykeops_chamfer if device.type == 'cuda' else torch_chamfer

    def _chamfer(data: Outputs, targets: Targets) -> torch.Tensor:
        return chamfer_backend(data.recon, targets.ref_cloud)

    return Loss(_chamfer, name='Chamfer')


def get_recon_loss() -> LossBase[Outputs, Targets]:
    cfg = ExperimentAE.get_config()
    cfg_user = MainExperiment.get_config().user
    recon_loss = cfg.objective.recon_loss
    device = cfg_user.device
    chamfer_loss = get_chamfer_loss()
    if recon_loss == ReconLosses.ChamferEMD and device.type == 'cuda':
        return chamfer_loss + get_emd_loss()
    return chamfer_loss


def get_embed_loss() -> Loss[Outputs, Targets]:
    cfg = ExperimentAE.get_config()
    cfg_loss = cfg.objective
    mse_loss = nn.MSELoss(reduction='none')

    def _embed_loss(data: Outputs, _not_used: Targets) -> torch.Tensor:
        return mse_loss(data.w_q, data.w_e).mean(dim=1)

    embed_loss = Loss(_embed_loss, name='Embed. Loss')
    return embed_loss


def gaussian_ll(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * (log_var + torch.pow(x - mu, 2) / torch.exp(log_var))


def gaussian_kld(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return 0.5 * (-1 - log_var + log_var.exp() + (mu ** 2))


def get_kld_loss() -> LossBase[Outputs, W_Targets]:

    cfg_w_ae = ExperimentWAE.get_config()

    def _kld_vamp(data: Outputs, targets: W_Targets) -> torch.Tensor:
        num_classes = targets.logits.shape[1]
        z_kld = data.z[:, num_classes:]
        mu_kld = data.mu[:, num_classes:]
        log_var_kld = data.log_var[:, num_classes:]
        pseudo_mu = data.pseudo_mu[:, num_classes:]
        pseudo_log_var = data.pseudo_log_var[:, num_classes:]

        posterior_ll = gaussian_ll(z_kld, mu_kld, log_var_kld).sum(1)  # sum dimensions
        k = pseudo_mu.shape[0]
        b = mu_kld.shape[0]
        z = z_kld.unsqueeze(1).expand(-1, k, -1)  # create a copy for each pseudo input
        pseudo_mu = pseudo_mu.unsqueeze(0).expand(b, -1, -1)  # expand to match the batch size
        pseudo_log_var = pseudo_log_var.unsqueeze(0).expand(b, -1, -1)  # expand to match the batch size
        prior_ll = torch.logsumexp(gaussian_ll(z, pseudo_mu, pseudo_log_var).sum(2), dim=1)
        total = posterior_ll - prior_ll + np.log(k)
        return total

    def _kld(data: Outputs, _: W_Targets) -> torch.Tensor:
        return gaussian_kld(mu=data.mu, log_var=data.log_var).sum(1)

    return Loss(_kld_vamp if cfg_w_ae.objective.vamp else _kld, name='KLD')


def get_counterfactual_loss() -> LossBase[Outputs, W_Targets]:
    log_softmax = nn.LogSoftmax(dim=1)
    kld = nn.KLDivLoss(reduction='none', log_target=True)


    def _counterfactual(data: Outputs, targets: W_Targets) -> torch.Tensor:
        mu_counterfactual = torch.log(data.y)
        prob = log_softmax(targets.logits)
        return +kld(mu_counterfactual, prob).sum(1)

    return Loss(_counterfactual, name='Counterfactual')


def get_nll_loss() -> LossBase[Outputs, W_Targets]:
    log_softmax = torch.nn.LogSoftmax(dim=2)

    def _nll(data: Outputs, targets: W_Targets) -> torch.Tensor:
        sqrt_dist = torch.sqrt(data.w_dist)
        w_neg_dist = - sqrt_dist + sqrt_dist.min(2, keepdim=True)[0]
        return (-log_softmax(w_neg_dist) * targets.one_hot_idx).sum((1, 2))

    return Loss(_nll, name='NLL')


def get_w_accuracy() -> Metric[Outputs, W_Targets]:
    def _accuracy(data: Outputs, targets: W_Targets) -> torch.Tensor:
        one_hot_predictions = F.one_hot(data.w_dist.argmin(2), num_classes=targets.one_hot_idx.shape[2])
        return (targets.one_hot_idx * one_hot_predictions).sum(2).mean(1)

    return Metric(_accuracy, name='Quantisation Accuracy', higher_is_better=True)


def get_cross_entropy_loss() -> LossBase[torch.Tensor, Targets]:
    torch_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return torch_loss(outputs, targets.label)

    return Loss(_cross_entropy, name='CrossEntropy')


def get_accuracy() -> MetricBase[torch.Tensor, Targets]:
    def _cross_entropy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return multiclass_accuracy(outputs, targets.label)

    return Metric(_cross_entropy, name='Accuracy', higher_is_better=True)


def get_macro_accuracy() -> MetricBase[torch.Tensor, Targets]:
    def _macro_accuracy(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return multiclass_accuracy(outputs, targets.label, average='macro', num_classes=outputs.shape[1])

    # triggering deprecated warning (remove this in the future if not fixed by torcheval)
    zero_output = torch.FloatTensor([[1, 0]])
    zero_tensor = torch.LongTensor([0])
    _macro_accuracy(zero_output, Targets(ref_cloud=zero_tensor, scale=zero_tensor, label=zero_tensor))

    return Metric(_macro_accuracy, name='Macro Accuracy', higher_is_better=True)


def get_f1() -> MetricBase[torch.Tensor, Targets]:
    def _f1(outputs: torch.Tensor, targets: Targets) -> torch.Tensor:
        return multiclass_f1_score(outputs, targets.label)

    return Metric(_f1, name='F1_Score', higher_is_better=True)


def get_classification_loss() -> LossBase[torch.Tensor, Targets]:
    return get_cross_entropy_loss() | get_accuracy() | get_macro_accuracy()


def get_w_encoder_loss() -> LossBase[Outputs, W_Targets]:
    c_kld = ExperimentWAE.get_config().objective.c_kld
    c_counterfactual = ExperimentWAE.get_config().objective.c_counterfactual
    return get_nll_loss() + c_kld * get_kld_loss() + c_counterfactual * get_counterfactual_loss() | get_w_accuracy()


def get_autoencoder_loss() -> LossBase[Outputs, Targets]:
    cfg = ExperimentAE.get_config()
    c_embed = cfg.objective.c_embedding
    loss = get_recon_loss()
    if cfg.model.head is ModelHead.VQVAE:
        return loss + c_embed * get_embed_loss()
    return loss
