"""Target evaluations of the counterfactuals."""

import torch
import torch.nn.functional as F

__all__ = ['interpolate_evaluation']


def interpolate_evaluation(evaluations: torch.Tensor, targets: torch.Tensor, /, *, strength: float) -> torch.Tensor:
    """Move the evaluations along a straight path on the simplex, towards the vertex of the target class.

    Args:
        evaluations: original evaluations ``(N, C)``.
        targets: target class of each evaluation ``(N,)``.
        strength: counterfactual strength: 0 keeps the evaluation, 1 reaches the vertex.

    Returns:
        The target evaluations ``(N, C)``.
    """
    vertices = F.one_hot(targets, num_classes=evaluations.shape[1]).to(evaluations.dtype)
    return (1 - strength) * evaluations + strength * vertices
