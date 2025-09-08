"""This module contains classes for various layers in the neural network."""

import abc
import sys
import functools
from typing import Any, Callable, TypeAlias, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

debug_mode = sys.gettrace()
if debug_mode:
    import pdb

_grad_t: TypeAlias = tuple[torch.Tensor, ...] | torch.Tensor
ActClass: TypeAlias = Callable[[], nn.Module]


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """
    This function is used for debugging purposes during the training process. It checks for NaN and Inf values in the
    output tensor of a neural network layer. If such values are found, it triggers a debugger for further inspection.

    Args:
        _not_used1:
        _not_used2:
        tensor_out: The output tensor of the neural network layer.

    """
    if isinstance(tensor_out, tuple):
        tensor = tensor_out[0]
    else:
        tensor = tensor_out
    if torch.any(torch.isnan(tensor)):
        breakpoint()
        pdb.set_trace()
    if torch.any(torch.isinf(tensor)):
        breakpoint()
        pdb.set_trace()
    return None


class View(nn.Module):
    """
    A simple module that reshapes the input tensor according to the specified shape.

    Args:
    *shape (int): A tuple of integers specifying the desired shape for the output tensor.

    Inputs:
    x (torch.Tensor): The input tensor to be reshaped.

    Returns:
    torch.Tensor: The reshaped output tensor with the specified shape.
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.view(self.shape)


class MaxChannel(nn.Module):
    """A class with a static method that applies the max operation along a specified axis of the input tensor."""

    @staticmethod
    def forward(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """Forward pass."""
        return torch.max(x, axis)[0]


# Input (Batch, Features)
class GeneralizedLinearLayer(nn.Module, metaclass=abc.ABCMeta):
    """A class that wraps a generalized linear (dense) layer class.

    Args:
        in_dim: The number of input features
        out_dim: The number of output features
        act_cls: An optional callable for the output activation of the linear layer
        batch_norm: A boolean value indicating whether to include batch normalization in the layer
        groups: The number of groups for the linear layer. Default is 1.

    Attributes:
        in_dim (int): The number of input features
        out_dim (int): The number of output features
        groups (int): The number of groups for the linear layer
        batch_norm (bool): A boolean value indicating whether to include batch normalization in the layer
        bias (bool): A boolean value indicating whether the bias term is in the layer (and not in batch normalization)
        dense (nn.Module): The wrapped layer of the neural network
        bn (nn.Module): The batch normalization layer of the neural network, if included
        act (Optional[nn.Module]): The activation function applied to the output of the linear layer or None
        init (Callable[[torch.Tensor], torch.Tensor]): The initialization function for the weights of the dense layer

    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 act_cls: Optional[ActClass] = None,
                 batch_norm: bool = True,
                 bn_momentum: Optional[float] = None,
                 groups: int = 1,
                 residual: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.groups = groups
        self.batch_norm = batch_norm
        self.bias = False if batch_norm else True
        self.dense = self.get_dense_layer()
        self.bn_momentum = 0.1 if bn_momentum is None else bn_momentum
        self.bn = self.get_bn_layer() if batch_norm else None
        self.resnet = residual
        if act_cls is None:
            self.act = None
        else:
            try:
                self.act = act_cls(inplace=True)  # type: ignore
            except TypeError:
                self.act = act_cls()
        if debug_mode:
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)
        self.init = self.get_init(self.act)
        self.init(self.dense.weight)

    @staticmethod
    def get_init(act: Optional[nn.Module]) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        This static method returns an initialization determined based on the type of activation.

        Parameters:
        act (Optional[nn.Module]): The optional activation function applied to the output of the linear layer.

        Returns:
        Callable[[torch.Tensor], torch.Tensor]: A partially initialized initialization for the dense layer's weights.
        """
        if act is None:
            return functools.partial(nn.init.xavier_normal_, gain=1)
        if isinstance(act, nn.Identity):  # this is for the final output layer
            return functools.partial(nn.init.xavier_normal_, gain=0.01)
        if isinstance(act, nn.ReLU):
            return functools.partial(nn.init.kaiming_uniform_, nonlinearity='relu')
        if isinstance(act, nn.LeakyReLU):
            return functools.partial(nn.init.kaiming_uniform_, a=act.negative_slope)
        if isinstance(act, nn.Hardtanh):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('tanh'))
        return lambda x: x

    @abc.abstractmethod
    def get_dense_layer(self) -> nn.Module:
        """Get the wrapped layer"""

    @abc.abstractmethod
    def get_bn_layer(self):
        """Get the batch normalization layer"""

    def forward(self, x):
        """Forward pass."""
        y = self.bn(self.dense(x)) if self.batch_norm else self.dense(x)
        if self.act is not None:
            y = self.act(y)
        if self.resnet:
            return y + x.repeat_interleave(self.out_dim // self.in_dim + 1, 1)[:, :y.shape[1], ...]
        return y


# Input (Batch, Features)
class LinearLayer(GeneralizedLinearLayer):

    def get_dense_layer(self) -> nn.Module:
        if self.groups > 1:
            raise NotImplementedError('nn.Linear has not option for groups')
        return nn.Linear(self.in_dim, self.out_dim, bias=self.bias)

    def get_bn_layer(self):
        return nn.BatchNorm1d(self.out_dim, momentum=self.bn_momentum)


# Input (Batch, Points, Features)
class PointsConvLayer(GeneralizedLinearLayer):

    def get_dense_layer(self) -> nn.Module:
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)

    def get_bn_layer(self):
        return nn.BatchNorm1d(self.out_dim, momentum=self.bn_momentum)


class EdgeConvLayer(GeneralizedLinearLayer):

    def get_dense_layer(self) -> nn.Module:
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)

    def get_bn_layer(self) -> nn.Module:
        return nn.BatchNorm2d(self.out_dim, momentum=self.bn_momentum)


class TemperatureScaledSoftmax(nn.Softmax):

    def __init__(self, dim: Optional[int] = None, temperature: float = 1):
        super().__init__(dim)
        self.temperature = torch.tensor(temperature, dtype=torch.float)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        return super().forward(x / self.temperature)


class TransferGrad(Function):
    """Transfers the gradient from one tensor to another."""

    @staticmethod
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass."""
        from_tensor, to_tensor = args
        return from_tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        """Backward pass."""
        return None, grad_outputs[0].clone()


def frozen_forward(network, x):
    """Temporarily disable gradients for all parameters."""
    was_requires_grad = []
    for p in network.parameters():
        was_requires_grad.append(p.requires_grad)
        p.requires_grad_(False)

    # Run the network
    output = network(x)

    # Restore original requires_grad flags
    for p, req in zip(network.parameters(), was_requires_grad):
        p.requires_grad_(req)

    return output
