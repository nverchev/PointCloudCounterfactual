"""Torch config utilities."""

import functools
import math

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

type ActClass = Callable[[], nn.Module]
type NormClass = Callable[[int], nn.Module]


def get_grouped_norm(n_channels: int) -> nn.Module:
    """Get grouped normalization layer."""
    n_groups = int(math.sqrt(n_channels))
    while n_channels % n_groups:
        n_groups -= 1

    return nn.GroupNorm(num_groups=n_groups, num_channels=n_channels)


DEFAULT_ACT: ActClass = functools.partial(nn.LeakyReLU, negative_slope=0.2)
DEFAULT_NORM: NormClass = get_grouped_norm


def get_activation_cls(act_name: str) -> ActClass:
    """Get activation class from name."""
    try:
        return getattr(nn.modules.activation, act_name)
    except AttributeError as ae:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.') from ae


def get_norm_cls(norm_name: str) -> NormClass:
    """Get normalization class from name."""
    try:
        return getattr(nn, norm_name)
    except AttributeError as ae:
        raise ValueError(f'Input norm_name "{norm_name}" is not the name of a pytorch nn.Module.') from ae


def get_optim_cls(optimizer_name: str) -> type[torch.optim.Optimizer]:
    """Get optimizer class from name."""
    try:
        return getattr(torch.optim, optimizer_name)
    except AttributeError as ae:
        raise ValueError(f'Input opt_name "{optimizer_name}" is not the name of a pytorch optimizer.') from ae


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
