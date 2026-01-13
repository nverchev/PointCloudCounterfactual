"""Package containing dataset classes."""

from typing import Any

from torch import distributed as dist

from src.config_options import Experiment, Datasets
from src.dataset.protocols import PointCloudDataset, Partitions
from src.dataset.modelnet import ModelNet40Dataset
from src.dataset.shapenet import ShapeNetDatasetFlow
from src.dataset.encoded import (
    WDatasetEncoder,
    WDatasetWithLogits,
    ReconstructedDatasetEncoder,
    BoundaryDataset,
    WDatasetWithLogitsFrozen,
    ReconstructedDatasetWithLogits,
    CounterfactualDatasetEncoder,
)

__all__ = [
    'BoundaryDataset',
    'CounterfactualDatasetEncoder',
    'ModelNet40Dataset',
    'Partitions',
    'PointCloudDataset',
    'ReconstructedDatasetEncoder',
    'ReconstructedDatasetWithLogits',
    'ShapeNetDatasetFlow',
    'WDatasetEncoder',
    'WDatasetWithLogits',
    'WDatasetWithLogitsFrozen',
    'get_dataset',
    'get_dataset_multiprocess_safe',
    'get_datasets',
]


def get_dataset(partition: Partitions) -> PointCloudDataset:
    """Getter for the dataset."""
    cfg = Experiment.get_config()
    user_cfg = Experiment.get_config().user
    user_cfg.path.data_dir.mkdir(exist_ok=True)

    dataset_name = cfg.data.dataset.name
    dataset_dict: dict[Datasets, Any] = {
        Datasets.ModelNet: ModelNet40Dataset,
        Datasets.ShapenetFlow: ShapeNetDatasetFlow,
    }
    dataset = dataset_dict[dataset_name]().split(partition)
    return dataset


def get_datasets() -> tuple[PointCloudDataset, PointCloudDataset]:
    """Get the correct datasets for training and testing."""
    cfg = Experiment.get_config()
    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    return train_dataset, test_dataset


def get_dataset_multiprocess_safe() -> tuple[PointCloudDataset, PointCloudDataset]:
    """Get the correct datasets for training and testing, but in a multiprocess safe way."""
    cfg = Experiment.get_config()
    datasets: tuple[PointCloudDataset, PointCloudDataset] | None = None
    if cfg.user.n_subprocesses:
        rank = dist.get_rank()
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                datasets = get_datasets()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])
    else:
        datasets = get_datasets()

    if datasets is None:
        raise RuntimeError('Datasets could not be created.')

    return datasets
