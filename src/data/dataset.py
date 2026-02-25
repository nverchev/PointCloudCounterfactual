"""Module containing dataset getters."""

from typing import Any

from torch import distributed as dist

from src.config import Experiment
from src.config.options import Datasets
from src.data.modelnet import ModelNet40Dataset
from src.data.split import Partitions, PointCloudSplit
from src.data.shapenet import ShapeNetFlowDataset


def get_dataset(partition: Partitions) -> PointCloudSplit:
    """Getter for the dataset."""
    cfg = Experiment.get_config()
    user_cfg = Experiment.get_config().user
    user_cfg.path.data_dir.mkdir(exist_ok=True)

    dataset_name = cfg.data.dataset.name
    dataset_dict: dict[Datasets, Any] = {
        Datasets.ModelNet: ModelNet40Dataset,
        Datasets.ShapeNetFlow: ShapeNetFlowDataset,
    }
    dataset = dataset_dict[dataset_name]().split(partition)
    return dataset


def _get_datasets() -> tuple[PointCloudSplit, PointCloudSplit]:
    """Get the correct datasets for training and testing."""
    cfg = Experiment.get_config()
    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    return train_dataset, test_dataset


def get_datasets() -> tuple[PointCloudSplit, PointCloudSplit]:
    """Get the correct datasets for training and testing, but in a multiprocess safe way."""
    cfg = Experiment.get_config()
    datasets: tuple[PointCloudSplit, PointCloudSplit] | None = None
    if cfg.user.n_subprocesses:
        rank = dist.get_rank()  # type: ignore
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                datasets = _get_datasets()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])  # type: ignore
    else:
        datasets = _get_datasets()

    if datasets is None:
        raise RuntimeError('Datasets could not be created.')

    return datasets
