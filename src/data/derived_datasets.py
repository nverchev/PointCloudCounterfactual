"""Datasets of what the trained models derive from the clouds."""

from __future__ import annotations

import abc
import contextlib

from collections.abc import Iterator
from typing import TYPE_CHECKING, Final, Protocol

import torch
import torch.nn as nn

from torch.utils import data

from src.data.split import CloudSplit
from src.data.structures import Inputs, Targets, WInputs

if TYPE_CHECKING:
    from src.lib.models import VQVAEModel

__all__ = [
    'EvaluatedSplit',
    'GeneratedSplit',
    'WDataset',
]

_BATCH_NORMS: Final = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


class WDataset(data.Dataset[tuple[WInputs, torch.Tensor]]):
    """Codewords of a split, with the evaluations of the classifier and the codes chosen for them.

    The clouds are perturbed again at every access, so the codewords are computed batch by batch.

    Attributes:
        split: Clouds to encode.
        model: Outer VQ-VAE encoding the clouds.
        evaluator: Classifier evaluating the clouds.
        device: Device on which the models reside.
    """

    split: Final[CloudSplit]
    model: Final[VQVAEModel]
    evaluator: Final[_EvaluatorProtocol]
    device: Final[torch.device]

    def __init__(self, split: CloudSplit, model: VQVAEModel, evaluator: _EvaluatorProtocol) -> None:
        """Initialize.

        Args:
            split: clouds to encode.
            model: outer VQ-VAE encoding the clouds.
            evaluator: classifier evaluating the clouds.
        """
        super().__init__()
        self.split = split
        self.model = model
        self.evaluator = evaluator
        self.device = model.device
        return

    def __getitem__(self, index: int) -> tuple[WInputs, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[tuple[WInputs, torch.Tensor]]:
        self.model.module.train(not torch.is_inference_mode_enabled())
        clouds = torch.stack([self.split[index][0].cloud for index in indices]).to(self.device)
        return self._encode(clouds)

    def __len__(self) -> int:
        return len(self.split)

    @torch.inference_mode()
    def _encode(self, clouds: torch.Tensor) -> list[tuple[WInputs, torch.Tensor]]:
        with _batch_statistics_only(self.model.module):
            w_q = self.model.module.encode(clouds)

        conditions = self.evaluator.evaluate(clouds)
        one_hot_idx = self.model.module.quantise(w_q)[1]
        return [
            (WInputs(w_q_i, condition), one_hot_idx_i)
            for w_q_i, condition, one_hot_idx_i in zip(w_q, conditions, one_hot_idx, strict=True)
        ]


class EvaluatedSplit(data.Dataset[tuple[Inputs, Targets]]):
    """Clouds with the evaluations that condition the inner VAE.

    Attributes:
        split: The clouds.
        conditions: Evaluation of each cloud ``(n_clouds, C)``.
    """

    split: Final[CloudSplit]
    conditions: Final[torch.Tensor]

    def __init__(self, split: CloudSplit, conditions: torch.Tensor, /) -> None:
        """Initialize.

        Args:
            split: the clouds.
            conditions: evaluation of each cloud ``(n_clouds, C)``.
        """
        super().__init__()
        self.split = split
        self.conditions = conditions
        return

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        inputs, targets = self.split[index]
        return inputs._replace(condition=self.conditions[index]), targets

    def __len__(self) -> int:
        return len(self.split)


class GeneratedSplit(data.Dataset[tuple[Inputs, Targets]]):
    """Clouds the counterfactual model generated, computed once, with the labels they are scored on.

    Attributes:
        clouds: Generated clouds ``(n_clouds, n_points, OUT_CHAN)``.
        labels: Class each cloud should be assigned to, true or requested ``(n_clouds,)``.
    """

    clouds: Final[torch.Tensor]
    labels: Final[torch.Tensor]

    def __init__(self, clouds: torch.Tensor, labels: torch.Tensor, /) -> None:
        """Initialize.

        Args:
            clouds: generated clouds ``(n_clouds, n_points, OUT_CHAN)``.
            labels: class each cloud should be assigned to, true or requested ``(n_clouds,)``.
        """
        super().__init__()
        self.clouds = clouds
        self.labels = labels
        return

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        cloud = self.clouds[index]
        return Inputs(cloud=cloud), Targets(ref_cloud=cloud, label=self.labels[index])

    def __len__(self) -> int:
        return len(self.clouds)


class _EvaluatorProtocol(Protocol):
    @abc.abstractmethod
    def evaluate(self, clouds: torch.Tensor, /) -> torch.Tensor:
        """Return the evaluation of the clouds ``(B, C)``, a point of the probability simplex."""


@contextlib.contextmanager
def _batch_statistics_only(module: nn.Module, /) -> Iterator[None]:
    # in training mode, batch normalization then uses the statistics of the batch without updating the running ones
    norms = [layer for layer in module.modules() if isinstance(layer, _BATCH_NORMS) and layer.track_running_stats]
    for layer in norms:
        layer.track_running_stats = False

    try:
        yield
    finally:
        for layer in norms:
            layer.track_running_stats = True

    return
