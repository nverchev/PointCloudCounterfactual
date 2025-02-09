import abc
import sys
import functools
from typing import Any, Callable, TypeAlias, Optional

import torch
import torch.nn as nn
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

    Parameters:
    _not_used1 (nn.Module): The neural network module being checked. This parameter is not used in the function.
    _not_used2 (_grad_t): The input tensor to the neural network layer. This parameter is not used in the function.
    tensor_out (_grad_t): The output tensor of the neural network layer.

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
        return x.view(self.shape)


class MaxChannel(nn.Module):
    """
    A class with a static method that applies the max operation along a specified axis of the input tensor.

    Args:
    x (torch.Tensor): The input tensor from which to compute the maximum value.
    axis (int, optional): The axis along which to compute the maximum value. Default is the last dimension.

    Returns:
    torch.Tensor: A tensor containing the maximum value along the specified axis.
    """

    @staticmethod
    def forward(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.max(x, axis)[0]


# Input (Batch, Features)
class GeneralizedLinearLayer(nn.Module, metaclass=abc.ABCMeta):
    """
    A class that wraps a generalized linear (dense) layer class.

    Args:
    in_dim (int): The number of input features.
    out_dim (int): The number of output features.
    act_cls (Optional[ActClass]): An optional callable for the output activation of the linear layer.
    batch_norm (bool): A boolean value indicating whether to include batch normalization in the layer.
    groups (int): The number of groups for the linear layer. Default is 1.

    Attributes:
    in_dim (int): The number of input features.
    out_dim (int): The number of output features.
    groups (int): The number of groups for the linear layer.
    batch_norm (bool): A boolean value indicating whether to include batch normalization in the layer.
    bias (bool): A boolean value indicating whether the bias term is in the layer (and not in batch normalization).
    dense (nn.Module): The wrapped layer of the neural network.
    bn (nn.Module): The batch normalization layer of the neural network, if included.
    act (Optional[nn.Module]): The activation function applied to the output of the linear layer or None.
    init (Callable[[torch.Tensor], torch.Tensor]): The initialization function for the weights of the dense layer.

    """

    def __init__(self,
                 in_dim:
                 int, out_dim: int,
                 act_cls: Optional[ActClass] = None,
                 batch_norm: bool = True,
                 groups: int = 1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.groups = groups
        self.batch_norm = batch_norm
        self.bias = False if batch_norm else True
        self.dense = self.get_dense_layer()
        self.bn = self.get_bn_layer() if batch_norm else None
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
        x = self.bn(self.dense(x)) if self.batch_norm else self.dense(x)
        return x if self.act is None else self.act(x)


# Input (Batch, Features)
class LinearLayer(GeneralizedLinearLayer):

    def get_dense_layer(self) -> nn.Module:
        if self.groups > 1:
            raise NotImplementedError('nn.Linear has not option for groups')
        return nn.Linear(self.in_dim, self.out_dim, bias=self.bias)

    def get_bn_layer(self):
        return nn.BatchNorm1d(self.out_dim)


# Input (Batch, Points, Features)
class PointsConvLayer(GeneralizedLinearLayer):

    def get_dense_layer(self) -> nn.Module:
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)

    def get_bn_layer(self):
        return nn.BatchNorm1d(self.out_dim)


class EdgeConvLayer(GeneralizedLinearLayer):

    def get_dense_layer(self) -> nn.Module:
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)

    def get_bn_layer(self) -> nn.Module:
        return nn.BatchNorm2d(self.out_dim)


class TransferGrad(Function):
    """
    A custom autograd function that transfers the gradient from one tensor to another.

    Args:
    *args (torch.Tensor): A tuple of input tensors. The first is the source of the gradient.
    **kwargs: Additional keyword arguments are ignored.

    Returns:
    torch.Tensor: The input tensor from_tensor is returned as the output.
    """

    @staticmethod
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        from_tensor, to_tensor = args
        ctx.save_for_backward(from_tensor, to_tensor)
        return from_tensor

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        from_tensor, to_tensor = ctx.saved_tensors
        return None, grad_outputs[0].clone()
