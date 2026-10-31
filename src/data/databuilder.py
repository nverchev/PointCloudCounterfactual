"""Base builder preparing a point cloud dataset for its splits."""

import abc
import dataclasses
import json
import pathlib

from typing import Any, ClassVar, Final

import numpy as np

from numpy import typing as npt

from src.config.experiment import Experiment
from src.data.split import CloudSplit, Partitions

__all__ = [
    'ConfigCached',
    'DataBuilder',
]


class ConfigCached(abc.ABCMeta):
    """Metaclass reusing one instance per class, data directory and data configuration."""

    _instances: ClassVar[dict[tuple[type, str], Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        key = (cls, cls._get_config_key())
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)

        return cls._instances[key]

    @staticmethod
    def _get_config_key() -> str:
        cfg = Experiment.get_config()
        settings = {'data_dir': cfg.user.path.data_dir, 'data': dataclasses.asdict(cfg.data)}
        return json.dumps(settings, sort_keys=True, default=str)


class DataBuilder(metaclass=ConfigCached):
    """Dataset preparing its clouds once and handing them to the splits: (download →) preprocess → split.

    Attributes:
        dir_raw: Folder of the files of the dataset.
        class_names: Names of the kept classes, by label.
    """

    folder_raw: ClassVar[str]

    dir_raw: Final[pathlib.Path]
    class_names: Final[tuple[str, ...]]
    _all_names: Final[tuple[str, ...]]

    def __init__(self) -> None:
        """Initialize.

        Raises:
            ValueError: if the configuration selects a class the dataset misses.

        Side Effects:
            Downloads the dataset when its folder is missing.
        """
        cfg = Experiment.get_config()
        self.dir_raw = cfg.user.path.data_dir / self.folder_raw
        self._download()
        self._all_names = self._get_all_names()
        selected = cfg.data.dataset.selected_classes
        unknown = [name for name in selected if name not in self._all_names]
        if unknown:
            msg = f'{unknown} are not classes of {type(self).__name__}.'
            raise ValueError(msg)

        self.class_names = selected or self._all_names
        self._preprocess()
        return

    def get_split(self, partition: Partitions, /, *, augment: bool) -> CloudSplit:
        """Get the split of a partition.

        Args:
            partition: partition to draw the clouds from.
            augment: whether every access perturbs the cloud.

        Returns:
            A split over the clouds of that partition.
        """
        clouds, labels = self._get_partition_data(partition)
        return CloudSplit(clouds, labels, augment=augment)

    @abc.abstractmethod
    def _download(self) -> None:
        """Fetch the files of the dataset when they are missing."""

    @abc.abstractmethod
    def _get_all_names(self) -> tuple[str, ...]:
        """Return the names of all the classes of the dataset, by their original label."""

    @abc.abstractmethod
    def _get_partition_data(self, partition: Partitions, /) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Return the clouds ``(N, n_points, 3)`` and the labels ``(N,)`` of a partition."""

    @abc.abstractmethod
    def _preprocess(self) -> None:
        """Make the clouds of the selected classes available for the splits."""
