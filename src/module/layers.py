"""Classes for various layers in the neural network."""

import abc
import functools
import pdb
import sys

from collections.abc import Callable
from typing import Any, Protocol, cast, override, runtime_checkable

import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Function

DEBUG_MODE = sys.gettrace()

type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor
type ActClass = Callable[[], nn.Module]
type NormClass = Callable[[int], nn.Module]


@runtime_checkable
class CanReset(Protocol):
    """Protocol for classes that can be reset to their initial state."""

    def reset_parameters(self) -> None:
        """Reset parameters of the layer."""


@runtime_checkable
class HasInplace(Protocol):
    """Protocol for classes that have an inplace option."""

    inplace: bool


class View(nn.Module):
    """A simple module that reshapes the input tensor according to the specified shape.

    Args:
    *shape (int): A tuple of integers specifying the desired shape for the output tensor.

    Inputs:
    x (torch.Tensor): The input tensor to be reshaped.

    Returns:
    torch.Tensor: The reshaped output tensor with the specified shape.
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.view(self.shape)


class MaxChannel(nn.Module):
    """A class with a static method that applies the max operation along a specified axis of the input tensor."""

    @staticmethod
    def forward(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """Forward pass."""
        return torch.max(x, axis)[0]


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """A class that includes a layer followed by an optional activation and grouped normalization.

    Args:
        in_dim: The number of input features
        out_dim: The number of output features
        act_cls: An optional callable for the output activation of the output
        norm_cls: An optional callable for normalization on the output (before the activation)
        n_groups_dense: The number of groups for the dense layer. Default is 1.
        use_soft_init: Whether to initialize the weights of the dense layer very close to zero.

    Attributes:
        in_dim (int): The number of input features
        out_dim (int): The number of output features
        n_groups_dense (int): The number of groups for the dense layer
        use_bias (bool): A boolean value indicating whether the bias term is in the layer (and no normalization)
        layer (nn.Module): The wrapped layer of the neural network
        norm (nn.Module): A boolean value indicating whether to add grouped normalization on the output
        act (Optional[nn.Module]): The activation function applied to the output
        init (Callable[[torch.Tensor], torch.Tensor]): The initialization function for the weights of the dense layer

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        n_groups_dense: int = 1,
        use_residual: bool = False,
        use_soft_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.n_groups_dense: int = n_groups_dense
        self.use_bias: bool = True if norm_cls is None else False
        self.layer = self.get_dense_layer()
        self.norm = self.get_norm_layer(norm_cls, out_dim)
        self.use_residual: bool = use_residual
        self.act = self.get_activation(act_cls)
        self.soft_init: bool = use_soft_init
        self.init = self.get_init(self.act, use_soft_init)
        self.init(cast(torch.Tensor, self.layer.weight))
        if DEBUG_MODE:
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    @abc.abstractmethod
    def get_dense_layer(self) -> nn.Module:
        """Get the wrapped layer"""

    def forward(self, x):
        """Forward pass."""
        y = self.layer(x)

        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        if self.use_residual:
            return y + x.repeat_interleave(self.out_dim // self.in_dim + 1, 1)[:, : y.shape[1], ...]

        return y

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        """Get the activation function."""
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace):
            act.inplace = True

        return act

    @staticmethod
    def get_init(act: nn.Module | None, soft_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization strategy according to the type of activation."""

        if soft_init:
            return functools.partial(nn.init.xavier_normal_, gain=0.01)

        if act is None:
            return functools.partial(nn.init.xavier_normal_, gain=1)

        if isinstance(act, nn.ReLU):
            return functools.partial(nn.init.kaiming_uniform_, nonlinearity='relu')

        if isinstance(act, nn.LeakyReLU):
            return functools.partial(nn.init.kaiming_uniform_, a=act.negative_slope)

        if isinstance(act, nn.Hardtanh):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('tanh'))

        return lambda x: x

    @staticmethod
    def get_norm_layer(norm_cls: NormClass | None, out_dim: int) -> nn.Module | None:
        """Get the normalization layer"""
        if norm_cls is None:
            return None

        return norm_cls(out_dim)


# Input (Batch, Features)
class LinearLayer(BaseLayer):
    @override
    def get_dense_layer(self) -> nn.Module:
        if self.n_groups_dense > 1:
            raise NotImplementedError('nn.Linear has no option for groups')

        return nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)


# Input (Batch, Points, Features)
class PointsConvLayer(BaseLayer):
    @override
    def get_dense_layer(self) -> nn.Module:
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=self.use_bias, groups=self.n_groups_dense)


# Input (Batch, Points, Features, Features)
class EdgeConvLayer(BaseLayer):
    @override
    def get_dense_layer(self) -> nn.Module:
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=self.use_bias, groups=self.n_groups_dense)


class TemperatureScaledSoftmax(nn.Softmax):
    temperature: torch.Tensor

    def __init__(self, dim: int | None = None, temperature: float = 1):
        super().__init__(dim)
        self.temperature = torch.tensor(temperature, dtype=torch.float)
        return

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        return super().forward(x / self.temperature)


class TransferGrad(Function):
    """Transfers the gradient from one tensor to another."""

    @staticmethod
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass."""
        from_tensor, _to_tensor = args
        return from_tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        """Backward pass."""
        return None, grad_outputs[0].clone()

    @classmethod
    def apply(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the function to two tensors."""
        return cast(torch.Tensor, super().apply(x, y))


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """This function is used for debugging purposes during the training process.

    It checks for NaN and Inf values in the output tensor of a neural network layer.
    If such values are found, it triggers a debugger for further inspection.
    """
    if isinstance(tensor_out, tuple):
        tensor = tensor_out[0]
    else:
        tensor = tensor_out

    if torch.any(torch.isnan(tensor)):
        breakpoint()
        pdb.set_trace()
    elif torch.any(torch.isinf(tensor)):
        breakpoint()
        pdb.set_trace()

    return None


def frozen_forward(network: nn.Module, x: torch.Tensor) -> Any:
    """Temporarily disable gradients for all parameters."""
    was_requires_grad: list[bool] = []
    for p in network.parameters():
        was_requires_grad.append(p.requires_grad)
        p.requires_grad_(False)

    # Run the network
    output = network(x)

    # Restore original requires_grad flags
    for p, req in zip(network.parameters(), was_requires_grad, strict=True):
        p.requires_grad_(req)

    return output


def reset_child_params(module: nn.Module) -> None:
    """Reset all parameters of a module and its children."""
    for layer in module.children():
        if isinstance(layer, CanReset):
            layer.reset_parameters()

        reset_child_params(layer)
        return
