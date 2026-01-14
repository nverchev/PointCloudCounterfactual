import abc
import enum
from abc import ABCMeta, abstractmethod
from typing import ClassVar, Any

from torch.utils.data import Dataset

from src.data.structures import Inputs, Targets


class PointCloudDataset(Dataset[tuple[Inputs, Targets]], metaclass=ABCMeta):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Inputs, Targets]: ...


class Partitions(enum.Enum):
    """Splits of the dataset."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class Singleton(type):
    """A metaclass that ensures only one instance of a class is created."""

    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class AbstractSingleton(Singleton, abc.ABCMeta):
    """Combining abstract and singleton metaclass."""


class SplitCreator(abc.ABC, metaclass=AbstractSingleton):
    """Abstract class that creates the splits for a dataset. Instantiated only once for efficiency."""

    @abc.abstractmethod
    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        """Retrieve the split."""
