"""Tests for the target evaluations of the counterfactuals."""

import torch

from src.utils.counterfactuals import interpolate_evaluation


def test_interpolate_evaluation() -> None:
    """The strength moves the evaluation in a straight line, staying on the simplex."""
    evaluations = torch.tensor([[0.8, 0.2], [0.5, 0.5]])
    targets = torch.tensor([1, 0])

    unchanged = interpolate_evaluation(evaluations, targets, strength=0.0)
    halfway = interpolate_evaluation(evaluations, targets, strength=0.5)
    vertex = interpolate_evaluation(evaluations, targets, strength=1.0)

    torch.testing.assert_close(unchanged, evaluations)
    torch.testing.assert_close(halfway, torch.tensor([[0.4, 0.6], [0.75, 0.25]]))
    torch.testing.assert_close(vertex, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
