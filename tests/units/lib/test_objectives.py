"""Tests for the losses and the metrics."""

import math

import torch

from src.data.structures import Outputs, Targets, WOutputs
from src.lib.objectives import (
    AggregatedAccuracy,
    gaussian_kld,
    get_accuracy,
    get_chamfer_loss,
    get_success,
    get_w_loss,
    shifted_gaussian_kld,
)
from src.module import VQVAE


def test_get_accuracy() -> None:
    """A sample is correct when the largest logit is its label."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
    targets = Targets(torch.empty(3), torch.tensor([0, 0, 0]))

    accuracy = get_accuracy().calculate(logits, targets)

    torch.testing.assert_close(accuracy['Accuracy'], torch.tensor([1.0, 0.0, 1.0]))


def test_get_success() -> None:
    """A counterfactual succeeds when the largest logit is the class in its label."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    targets = Targets(torch.empty(2), torch.tensor([0, 0]))

    success = get_success().calculate(logits, targets)

    torch.testing.assert_close(success['Success'], torch.tensor([1.0, 0.0]))


def test_aggregated_accuracy_counts_the_confusion_matrix() -> None:
    """The matrix has the labels on the rows, and the accuracy comes from its diagonal."""
    objective = AggregatedAccuracy(2)
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 2.0]])

    objective.update(logits, Targets(torch.empty(4), torch.tensor([0, 0, 1, 1])))
    metrics = objective.compute()

    torch.testing.assert_close(objective.confusion_matrix, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    assert metrics['Accuracy'] == 0.5


def test_chamfer_is_averaged_over_the_points() -> None:
    """The distance does not depend on how many points the clouds hold."""
    cloud = torch.zeros(1, 8, 3)
    shifted = cloud + torch.tensor([1.0, 0.0, 0.0])

    chamfer = get_chamfer_loss().calculate(Outputs(shifted), Targets(cloud, torch.zeros(1)))

    torch.testing.assert_close(chamfer['Chamfer'], torch.tensor([2.0]))


def test_gaussian_kld_vanishes_on_the_standard_gaussian() -> None:
    """A standard Gaussian has no divergence from itself, and a shift of the mean has one."""
    assert gaussian_kld(torch.zeros(2), torch.zeros(2)).sum() == 0
    assert gaussian_kld(torch.tensor([2.0]), torch.zeros(1)) == 2


def test_shifted_gaussian_kld_matches_the_closed_form() -> None:
    """The divergence given by the differences is the one between the two Gaussians."""
    p_mu, p_log_var = torch.tensor([0.5]), torch.tensor([0.3])
    q_mu, q_log_var = torch.tensor([1.5]), torch.tensor([-0.4])
    expected = 0.5 * (p_log_var - q_log_var + (q_log_var.exp() + (q_mu - p_mu) ** 2) / p_log_var.exp() - 1)

    divergence = shifted_gaussian_kld(q_mu - p_mu, q_log_var - p_log_var, p_log_var)

    torch.testing.assert_close(divergence, expected)


def test_w_loss_sums_the_likelihood_and_the_weighted_divergence() -> None:
    """A codeword on the chosen codes is predicted exactly, and the criterion adds both terms."""
    module = VQVAE()
    w_recon = module.codebook.data[:, 1].reshape(1, -1)
    one_hot_idx = torch.zeros(1, module.dim_codes, module.book_size)
    one_hot_idx[:, :, 1] = 1
    no_latent = torch.zeros(1, 2)
    outputs = WOutputs(w_recon, no_latent, no_latent, no_latent, no_latent, no_latent)

    values = get_w_loss(module).calculate(outputs, one_hot_idx)

    assert values['Quantization Accuracy'] == 1
    assert values['KLD'] == 0
    assert math.isclose(values['Criterion'].item(), values['NLL'].item())
