"""Events published during training and evaluation."""

from __future__ import annotations

import dataclasses

from torch import Tensor

from drytorch.core import log_events

__all__ = ['ConfusionMatrixEvent']


@dataclasses.dataclass(frozen=True)
class ConfusionMatrixEvent(log_events.Event):
    """Event logged when an evaluation computes a confusion matrix.

    Attributes:
        source_name: Name of the runner that evaluated the model.
        model_name: Name of the evaluated model.
        epoch: Epoch of the evaluated weights.
        matrix: Counts, labels on the rows ``(n_classes, n_classes)``.
    """

    source_name: str
    model_name: str
    epoch: int
    matrix: Tensor
