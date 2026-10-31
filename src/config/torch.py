"""Torch config utilities."""

import random

from typing import Final, Literal

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'ActClass',
    'SlopedLeakyReLU',
    'get_activation_cls',
    'get_device',
    'get_optim_cls',
    'set_seed',
]

NEGATIVE_SLOPE: Final = 0.2
"""Slope of the activation of the published encoders."""

type ActClass = type[nn.Module]


class SlopedLeakyReLU(nn.LeakyReLU):
    """Leaky ReLU with the slope of the published encoders."""

    def __init__(self, inplace: bool = False) -> None:
        """Initialize.

        Args:
            inplace: whether to modify the input.
        """
        super().__init__(negative_slope=NEGATIVE_SLOPE, inplace=inplace)
        return


def get_activation_cls(act_name: str | ActClass, /) -> ActClass:
    """Get an activation class from the custom one and the ones available in PyTorch.

    Args:
        act_name: name of the class, or the class itself, which is looked up by its name.

    Returns:
        The activation class.

    Raises:
        ValueError: if no activation class has that name.
    """
    if isinstance(act_name, type):
        act_name = act_name.__name__

    if act_name == SlopedLeakyReLU.__name__:
        return SlopedLeakyReLU

    try:
        return getattr(nn.modules.activation, act_name)
    except AttributeError as ae:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.') from ae


def get_device(device: str | torch.device, /) -> torch.device | Literal['auto']:
    """Get the device where the models run.

    Args:
        device: ``auto`` for the accelerator when available, or a device, as a name such as ``cpu`` or ``cuda:1``.

    Returns:
        The device, or ``'auto'``.

    Raises:
        ValueError: if the name is not ``auto`` or the name of a device.
    """
    if isinstance(device, torch.device) or device == 'auto':
        return device

    try:
        return torch.device(device)
    except RuntimeError as re:
        raise ValueError(f'Input device "{device}" is not auto or the name of a device.') from re


def get_optim_cls(optimizer_name: str | type[torch.optim.Optimizer], /) -> type[torch.optim.Optimizer]:
    """Get an optimizer class from the ones available in PyTorch.

    Args:
        optimizer_name: name of the class, or the class itself, which is looked up by its name.

    Returns:
        The optimizer class.

    Raises:
        ValueError: if no optimizer class has that name.
    """
    if isinstance(optimizer_name, type):
        optimizer_name = optimizer_name.__name__

    try:
        return getattr(torch.optim, optimizer_name)
    except AttributeError as ae:
        raise ValueError(f'Input opt_name "{optimizer_name}" is not the name of a pytorch optimizer.') from ae


def set_seed(seed: int, /) -> None:
    """Set seed for reproducibility.

    Args:
        seed: seed for the random generators of Python, NumPy and PyTorch.

    Side Effects:
        Seeds the random generators and makes cuDNN choose deterministic algorithms.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return
