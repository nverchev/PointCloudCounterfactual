"""Losses and metrics of the three training phases."""

from __future__ import annotations

import warnings

from typing import ClassVar, Final, override

import torch
import torch.nn.functional as F

from torch import Tensor

from drytorch.lib import objectives
from drytorch.lib.accumulators import AbstractAccumulator

from src.config import Experiment
from src.data.structures import Outputs, Targets, WOutputs, get_quantization
from src.module import VQVAE
from src.utils import distances

__all__ = [
    'AggregatedAccuracy',
    'ConfusionMatrixAccumulator',
    'gaussian_kld',
    'get_accuracy',
    'get_autoencoder_loss',
    'get_chamfer_loss',
    'get_classification_loss',
    'get_cross_entropy',
    'get_embedding_loss',
    'get_emd_loss',
    'get_reconstruction_metrics',
    'get_success',
    'get_w_loss',
    'shifted_gaussian_kld',
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def get_accuracy() -> objectives.Metric[Tensor, Targets]:
    """Get the per-sample accuracy."""
    return objectives.Metric(_is_label_predicted, name='Accuracy', higher_is_better=True)


def get_cross_entropy() -> objectives.Loss[Tensor, Targets]:
    """Get the per-sample cross-entropy."""

    def _cross_entropy(logits: Tensor, targets: Targets) -> Tensor:
        return F.cross_entropy(logits, targets.label, reduction='none')

    return objectives.Loss(_cross_entropy, name='CrossEntropy')


def get_classification_loss() -> objectives.LossWithMetrics[Tensor, Targets]:
    """Get the cross-entropy, also reporting the accuracy."""
    return objectives.join(get_cross_entropy(), get_accuracy())


def get_success() -> objectives.Metric[Tensor, Targets]:
    """Get whether each counterfactual reached the class in its label."""
    return objectives.Metric(_is_label_predicted, name='Success', higher_is_better=True)


class ConfusionMatrixAccumulator(AbstractAccumulator[Tensor, Tensor]):
    """Accumulator of confusion matrices.

    Attributes:
        matrix: total counts by label (rows) and prediction (columns) ``(n_classes, n_classes)``.
    """

    matrix: Tensor

    def __init__(self, matrix: Tensor) -> None:
        """Initialize.

        Args:
            matrix: counts of a batch, with labels on the rows ``(n_classes, n_classes)``.
        """
        self.matrix = matrix
        return

    @classmethod
    @override
    def from_value(cls, value: Tensor) -> ConfusionMatrixAccumulator:
        return cls(value.clone())

    @override
    def merge(self, other: ConfusionMatrixAccumulator) -> None:
        self.matrix += other.matrix
        return

    @override
    def sync(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.matrix)

        return

    @override
    def reduce(self) -> Tensor:
        return self.matrix.float()


class AggregatedAccuracy(objectives.Objective[Tensor, Targets]):
    """Accuracy over a full evaluation set, keeping its confusion matrix.

    Attributes:
        n_classes: Number of classes.
    """

    _KEY: ClassVar[str] = 'confusion_matrix'
    n_classes: Final[int]
    _matrix: Tensor | None

    def __init__(self, n_classes: int) -> None:
        """Initialize.

        Args:
            n_classes: number of classes.
        """
        super().__init__()
        self.n_classes = n_classes
        self._matrix = None
        return

    @property
    def confusion_matrix(self) -> Tensor:
        """Counts of the last computed evaluation, labels on the rows ``(n_classes, n_classes)``.

        Raises:
            ValueError: if the metrics have not been computed yet.
        """
        if self._matrix is None:
            msg = 'The confusion matrix is available after computing the metrics.'
            raise ValueError(msg)

        return self._matrix

    @classmethod
    @override
    def _get_accumulator_cls(cls) -> type[ConfusionMatrixAccumulator]:
        return ConfusionMatrixAccumulator

    @override
    def calculate(self, outputs: Tensor, targets: Targets) -> dict[str, Tensor]:
        indices = targets.label * self.n_classes + outputs.argmax(dim=-1)
        counts = torch.bincount(indices, minlength=self.n_classes**2)
        return {self._KEY: counts.view(self.n_classes, self.n_classes)}

    @override
    def _compute(self) -> dict[str, Tensor]:
        if self._KEY not in self._aggregator.accumulators:
            return {}

        matrix = self._aggregator.accumulators[self._KEY].reduce()
        self._matrix = matrix
        total_by_class = matrix.sum(dim=1)
        per_class = matrix.diag() / total_by_class.clamp(min=1)
        return {'Accuracy': matrix.diag().sum() / matrix.sum(), 'MacroAccuracy': per_class[total_by_class > 0].mean()}


# ---------------------------------------------------------------------------
# Outer VQ-VAE
# ---------------------------------------------------------------------------


def get_chamfer_loss() -> objectives.Loss[Outputs, Targets]:
    """Get the Chamfer distance, averaged over the points."""

    def _chamfer(outputs: Outputs, targets: Targets) -> Tensor:
        return distances.chamfer(outputs.recon, targets.ref_cloud) / targets.ref_cloud.shape[1]

    return objectives.Loss(_chamfer, name='Chamfer')


def get_emd_loss() -> objectives.Loss[Outputs, Targets]:
    """Get the earth mover distance, averaged over the points."""

    def _emd(outputs: Outputs, targets: Targets) -> Tensor:
        return distances.emd(outputs.recon, targets.ref_cloud) / targets.ref_cloud.shape[1]

    return objectives.Loss(_emd, name='EMD')


def get_embedding_loss() -> objectives.Loss[Outputs, Targets]:
    """Get the mean squared distance between the codewords and the codes that replace them."""

    def _embedding(outputs: Outputs, _: Targets) -> Tensor:
        quantization = get_quantization(outputs)
        return F.mse_loss(quantization.w_q, quantization.w_e, reduction='none').mean(dim=1)

    return objectives.Loss(_embedding, name='Embedding')


def get_autoencoder_loss() -> objectives.LossWithMetrics[Outputs, Targets]:
    """Get the loss of the outer VQ-VAE.

    Returns:
        The weighted sum of the reconstruction distances and of the embedding loss.

    Side Effects:
        Warns when the earth mover distance is left out.
    """
    cfg_loss = Experiment.get_config().autoencoding.train.loss
    loss = cfg_loss.c_chamfer * get_chamfer_loss() + cfg_loss.c_embedding * get_embedding_loss()
    if _on_accelerator():
        return loss + cfg_loss.c_emd * get_emd_loss()

    warnings.warn('The earth mover distance needs the accelerator: leaving it out.', stacklevel=2)
    return loss


def get_reconstruction_metrics() -> objectives.MetricCollection[Outputs, Targets]:
    """Get the distances between a reconstruction and its reference cloud, averaged over the points.

    Returns:
        The Chamfer distance and, on the accelerator, the earth mover distance.
    """
    with_emd = _on_accelerator()

    def _distances(outputs: Outputs, targets: Targets) -> dict[str, Tensor]:
        n_points = targets.ref_cloud.shape[1]
        values = {'Chamfer': distances.chamfer(outputs.recon, targets.ref_cloud) / n_points}
        if with_emd:
            values['EMD'] = distances.emd(outputs.recon, targets.ref_cloud) / n_points

        return values

    return objectives.MetricCollection(_distances)


# ---------------------------------------------------------------------------
# Inner VAE
# ---------------------------------------------------------------------------


def gaussian_kld(mu: Tensor, log_var: Tensor) -> Tensor:
    """Get the divergence of a diagonal Gaussian from the standard one.

    Args:
        mu: mean.
        log_var: log variance.

    Returns:
        The divergence of each coordinate.
    """
    return 0.5 * (-1 - log_var + log_var.exp() + mu**2)


def shifted_gaussian_kld(d_mu: Tensor, d_log_var: Tensor, p_log_var: Tensor) -> Tensor:
    """Get the divergence of a diagonal Gaussian from a prior, given their differences.

    Args:
        d_mu: mean of the posterior minus mean of the prior.
        d_log_var: log variance of the posterior minus log variance of the prior.
        p_log_var: log variance of the prior.

    Returns:
        The divergence of each coordinate.
    """
    return 0.5 * (-1 - d_log_var + d_log_var.exp() + d_mu**2 / p_log_var.exp())


def get_w_loss(autoencoder: VQVAE, /) -> objectives.DictLoss[WOutputs, Tensor]:
    """Get the loss of the inner VAE.

    The inner VAE predicts which code of every group the outer VQ-VAE chose, so the reconstruction
    term is a likelihood on the discrete latent space. The divergence of ``z_u`` is from the
    standard Gaussian, the one of ``z_y`` from its conditional prior.

    Args:
        autoencoder: trained VQ-VAE, whose codebook scores the predictions.

    Returns:
        The loss, reporting the divergence, the likelihood and the share of codes predicted correctly.
    """
    c_kld = Experiment.get_config().w_autoencoding.train.loss.c_kld

    def _w_loss(outputs: WOutputs, one_hot_idx: Tensor) -> dict[str, Tensor]:
        kld_u = gaussian_kld(outputs.mu_u, outputs.log_var_u).sum(1)
        kld_y = shifted_gaussian_kld(outputs.d_mu_y, outputs.d_log_var_y, outputs.p_log_var_y).sum(1)
        square_distance = autoencoder.square_distances(outputs.w_recon)
        distance = torch.sqrt(square_distance)
        scores = -distance + distance.min(2, keepdim=True)[0].detach()
        likelihood = -(scores.log_softmax(dim=2) * one_hot_idx).sum((1, 2))
        predicted = F.one_hot(square_distance.argmin(2), num_classes=one_hot_idx.shape[2]).float()
        accuracy = (one_hot_idx * predicted).sum(2).mean(1)
        return {
            'Criterion': likelihood + c_kld * (kld_u + kld_y),
            'KLD': kld_u + kld_y,
            'NLL': likelihood,
            'Quantization Accuracy': accuracy,
        }

    return objectives.DictLoss(_w_loss, name='Criterion')


def _is_label_predicted(logits: Tensor, targets: Targets) -> Tensor:
    return (logits.argmax(dim=-1) == targets.label).float()


def _on_accelerator() -> bool:
    # the exact matching that stands in for the extension is too slow for an epoch
    device = Experiment.get_config().user.device
    return torch.accelerator.is_available() if device == 'auto' else device.type != 'cpu'
