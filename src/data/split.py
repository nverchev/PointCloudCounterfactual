"""Partitions of a dataset and the split feeding the models."""

import enum

from typing import Final

import numpy as np
import torch

from numpy import typing as npt
from torch.utils import data

from src.config.experiment import Experiment
from src.data.augmentations import jitter
from src.data.structures import Inputs, Targets

__all__ = [
    'CloudSplit',
    'Partitions',
]


class Partitions(enum.Enum):
    """Named dataset splits, ``train_val`` joining ``train`` and ``val``."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class CloudSplit(data.Dataset[tuple[Inputs, Targets]]):
    """Clouds of one partition with their labels.

    With augmentation, every access perturbs the coordinates; the model reconstructs the same
    perturbed cloud it reads.

    Attributes:
        clouds: Clouds ``(n_clouds, n_points, IN_CHAN)``.
        labels: Label of each cloud ``(n_clouds,)``.
        augment: Whether every access perturbs the cloud.
        jitter_sigma: Standard deviation of the noise added to the coordinates.
        jitter_clip: Largest absolute value of the noise.
    """

    clouds: Final[npt.NDArray[np.float32]]
    labels: Final[npt.NDArray[np.int64]]
    augment: Final[bool]
    jitter_sigma: Final[float]
    jitter_clip: Final[float]

    def __init__(self, clouds: npt.NDArray[np.float32], labels: npt.NDArray[np.int64], /, *, augment: bool) -> None:
        """Initialize.

        Args:
            clouds: clouds ``(n_clouds, n_points, IN_CHAN)``.
            labels: label of each cloud ``(n_clouds,)``.
            augment: whether every access perturbs the cloud.
        """
        super().__init__()
        cfg_augmentation = Experiment.get_config().data.augmentation
        self.clouds = clouds
        self.labels = labels
        self.augment = augment
        self.jitter_sigma = cfg_augmentation.jitter_sigma
        self.jitter_clip = cfg_augmentation.jitter_clip
        return

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        cloud = torch.from_numpy(self.clouds[index])
        if self.augment:
            cloud = jitter(cloud, sigma=self.jitter_sigma, clip=self.jitter_clip)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return Inputs(cloud=cloud), Targets(ref_cloud=cloud, label=label)

    def __len__(self) -> int:
        return len(self.clouds)
