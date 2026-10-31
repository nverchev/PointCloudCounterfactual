"""Data loaders of the experiment."""

import torch

from torch.utils import data

from drytorch.lib import loading

from src.config import Experiment
from src.data import WDataset
from src.data.structures import Inputs, Targets, WInputs

__all__ = [
    'get_cloud_loader',
    'get_w_loader',
]


def get_cloud_loader(
    split: data.Dataset[tuple[Inputs, Targets]], /, *, batch_size: int
) -> loading.DataLoader[tuple[Inputs, Targets]]:
    """Get a loader of clouds.

    Args:
        split: clouds to load.
        batch_size: size of the batches.

    Returns:
        A loader with the configured number of worker processes.
    """
    n_workers = Experiment.get_config().user.n_workers
    return loading.DataLoader(split, batch_size=batch_size, n_workers=n_workers)


def get_w_loader(split: WDataset, /) -> loading.DataLoader[tuple[WInputs, torch.Tensor]]:
    """Get the loader of the codeword phase.

    The codewords come out of the models already on the device, so they are not pinned and not
    loaded in another process.

    Args:
        split: codewords to load.

    Returns:
        A loader drawing the configured batch size.
    """
    batch_size = Experiment.get_config().w_autoencoding.train.batch_size
    return loading.DataLoader(split, batch_size=batch_size, pin_memory=False)
