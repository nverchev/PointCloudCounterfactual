"""Torch config utilities."""

import functools
from collections.abc import Callable

import numpy as np
import torch

type ActClass = Callable[[], torch.nn.Module]

DEFAULT_ACT: ActClass = functools.partial(torch.nn.LeakyReLU, negative_slope=0.2)


def get_activation_cls(act_name: str) -> ActClass:
    """Get activation class from name."""
    try:
        return getattr(torch.nn.modules.activation, act_name)
    except AttributeError as ae:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.') from ae


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
