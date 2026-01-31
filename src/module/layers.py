"""Classes for various layers in the neural network."""

import abc
import functools
import itertools
import math
import pdb
import sys

from collections.abc import Callable
from typing import Any, Protocol, cast, override, runtime_checkable, TypeVar, Generic
from collections.abc import Sequence

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


@runtime_checkable
class Weighted(Protocol):
    """Protocol for classes that have a weight attribute."""

    weight: torch.Tensor
    bias: torch.Tensor | None


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
        use_trunc_init: Whether to initialize the weights of the dense layer very close to zero.

    Attributes:
        in_dim (int): The number of input features
        out_dim (int): The number of output features
        n_groups_dense (int): The number of groups for the dense layer
        use_bias (bool): A boolean value indicating whether the bias term is in the layer (and no normalization)
        module (nn.Module): The wrapped layer of the neural network
        norm (nn.Module): A boolean value indicating whether to add grouped normalization on the output
        act (Optional[nn.Module]): The activation function applied to the output
        init_weight (Callable[[torch.Tensor], torch.Tensor]): The initialization function for the dense layer's weights
        init_bias (Callable[[torch.Tensor], torch.Tensor]): The initialization function for the dense layer's bias

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        n_groups_dense: int = 1,
        use_trunc_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.n_groups_dense: int = n_groups_dense
        self.use_bias: bool = True if norm_cls is None else False
        self.module = self.get_module()
        self.norm = self.get_norm_layer(norm_cls, out_dim)
        self.act = self.get_activation(act_cls)
        self.use_trunc_init: bool = use_trunc_init
        self.init_weight = self.get_init_weight(self.act, use_trunc_init)
        self.init_bias = self.get_init_bias(use_trunc_init)
        self.init_weight(cast(torch.Tensor, self.module.weight))
        if self.norm is None:
            self.init_bias(cast(torch.Tensor, self.module.bias))
        else:
            self.init_bias(cast(torch.Tensor, self.norm.bias))

        if DEBUG_MODE:
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Get the wrapped layer"""

    def forward(self, x):
        """Forward pass."""
        y = self.module(x)

        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

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
    def get_init_weight(act: nn.Module | None, use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization for the weight strategy according to the type of activation."""

        if use_trunc_init:
            return functools.partial(nn.init.trunc_normal_, mean=0.0, std=0.02, a=-0.04, b=0.04)

        if act is None:
            return functools.partial(nn.init.xavier_normal_, gain=1)

        if isinstance(act, nn.Hardtanh):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('tanh'))

        if isinstance(act, nn.ReLU):
            return functools.partial(nn.init.kaiming_normal_, nonlinearity='relu')

        if isinstance(act, nn.LeakyReLU):
            return functools.partial(nn.init.kaiming_normal_, a=act.negative_slope, nonlinearity='leaky_relu')

        if isinstance(act, nn.GELU):
            return functools.partial(nn.init.xavier_normal_, gain=1)

        return functools.partial(nn.init.xavier_normal_, gain=1)

    @staticmethod
    def get_init_bias(use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization strategy for the bias according to the type of activation."""
        return torch.nn.init.zeros_ if use_trunc_init else lambda x: x

    @staticmethod
    def get_norm_layer(norm_cls: NormClass | None, out_dim: int) -> nn.Module | None:
        """Get the normalization layer"""
        if norm_cls is None:
            return None

        return norm_cls(out_dim)


# Input (Batch, Features)
class LinearLayer(BaseLayer):
    @override
    def get_module(self) -> nn.Module:
        if self.n_groups_dense > 1:
            raise NotImplementedError('nn.Linear has no option for groups')

        return nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)


# Input (Batch, Points, Features)
class PointsConvLayer(BaseLayer):
    @override
    def get_module(self) -> nn.Module:
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=self.use_bias, groups=self.n_groups_dense)


# Input (Batch, Points, Features, Features)
class EdgeConvLayer(BaseLayer):
    @override
    def get_module(self) -> nn.Module:
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=self.use_bias, groups=self.n_groups_dense)


Layer = TypeVar('Layer', bound=BaseLayer)


class BaseResBlock(nn.Module, Generic[Layer], abc.ABC):
    """A block of a neural network consisting of a sequence of layers."""

    def __init__(
        self,
        dims: Sequence[int],
        act_cls: ActClass | None = None,
        norm_cls: NormClass | None = None,
        n_groups_layer: int = 1,
    ) -> None:
        super().__init__()
        self.dims: Sequence[int] = dims
        self.act_cls: ActClass | None = act_cls
        self.norm_cls: NormClass | None = norm_cls
        self.n_groups_layer: int = n_groups_layer
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        for in_dim, out_dim in itertools.pairwise(dims):
            self.layers.append(self.get_layer(in_dim, out_dim, act_cls, norm_cls, n_groups_layer))
            if in_dim == out_dim:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(self.get_projection(in_dim, out_dim))

        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer.module, Weighted):
                    layer.module.weight.mul_(1 / math.sqrt(len(dims) - 1))
                    if layer.module.bias is not None:
                        layer.module.bias.mul_(0)

        return

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = self.projections[i](x) + layer(x)

        return x

    @classmethod
    @abc.abstractmethod
    def get_layer(
        cls, in_dim: int, out_dim: int, act_cls: ActClass | None, norm_cls: NormClass | None, n_groups_layer: int
    ) -> Layer:
        """Get the layer"""

    @classmethod
    @abc.abstractmethod
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        """Get a projection layer"""


class LinearResBlock(BaseResBlock[LinearLayer]):
    """A block of a neural network consisting of a sequence of linear layers."""

    @classmethod
    @override
    def get_layer(
        cls, in_dim: int, out_dim: int, act_cls: ActClass | None, norm_cls: NormClass | None, n_groups_layer: int
    ) -> LinearLayer:
        return LinearLayer(in_dim, out_dim, act_cls, norm_cls, n_groups_layer)

    @classmethod
    @override
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        projection = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.eye_(projection.weight)
        return projection


class PointsConvResBlock(BaseResBlock[PointsConvLayer]):
    """A block of a neural network consisting of a sequence of points convolution layers."""

    @classmethod
    @override
    def get_layer(
        cls, in_dim: int, out_dim: int, act_cls: ActClass | None, norm_cls: NormClass | None, n_groups_layer: int
    ) -> PointsConvLayer:
        return PointsConvLayer(in_dim, out_dim, act_cls, norm_cls, n_groups_layer)

    @classmethod
    @override
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        projection = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        nn.init.eye_(projection.weight[:, :, 0])
        return projection


class EdgeConvResBlock(BaseResBlock[EdgeConvLayer]):
    """A block of a neural network consisting of a sequence of edge convolution layers."""

    @classmethod
    @override
    def get_layer(
        cls, in_dim: int, out_dim: int, act_cls: ActClass | None, norm_cls: NormClass | None, n_groups_layer: int
    ) -> EdgeConvLayer:
        return EdgeConvLayer(in_dim, out_dim, act_cls, norm_cls, n_groups_layer)

    @classmethod
    @override
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        projection = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        nn.init.eye_(projection.weight[:, :, 0, 0])
        return projection


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    Args:
        in_dim: input embedding dimension
        n_heads: Number of attention heads
        hidden_dim: Dimension of the hidden layer in the feedforward network
        act_cls: Activation class for feedforward network
        dropout_rate: Dropout probability
        out_dim: The number of output embeddings. Defaults to in_dim
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_dim: int,
        n_heads: int,
        hidden_dim: int,
        act_cls: ActClass,
        dropout_rate: float,
        out_dim: int | None = None,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.n_heads = n_heads
        self.dim_feedforward = hidden_dim
        self.dropout_p = dropout_rate
        self.use_residual = use_residual
        if self.out_dim != self.in_dim and self.use_residual:
            raise ValueError('Output dimension must be equal to input dimension if residual connections are used')

        self.self_attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout_rate, batch_first=True)
        self.linear1 = LinearLayer(in_dim, hidden_dim, act_cls=act_cls, use_trunc_init=True)
        self.linear2 = LinearLayer(hidden_dim, self.out_dim, use_trunc_init=True)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(self.out_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        return

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.norm1(x)
        y = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        y = self.dropout1(y)
        y = x + y
        z = self.norm2(y)
        z = self.linear2(self.dropout2(self.linear1(z)))
        if self.use_residual:
            z = y + z

        return z


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    Args:
        in_dim: input embedding dimension
        n_heads: Number of attention heads
        hidden_dim: Dimension of the hidden layer in the feedforward network
        act_cls: Activation class for feedforward network
        dropout_rate: Dropout probability
        out_dim: The number of output embeddings. Defaults to in_dim
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_dim: int,
        n_heads: int,
        hidden_dim: int,
        act_cls: ActClass,
        dropout_rate: float,
        out_dim: int | None = None,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.n_heads = n_heads
        self.dim_feedforward = hidden_dim
        self.dropout_p = dropout_rate
        self.use_residual = use_residual
        if self.out_dim != self.in_dim and self.use_residual:
            raise ValueError('Output dimension must be equal to input dimension if residual connections are used')

        self.self_attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout_rate, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout_rate, batch_first=True)
        self.linear1 = LinearLayer(in_dim, hidden_dim, act_cls=act_cls, use_trunc_init=True)
        self.linear2 = LinearLayer(hidden_dim, self.out_dim, use_trunc_init=True)
        self.memory_norm = nn.LayerNorm(in_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(self.out_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        return

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Target input tensor of shape (batch, tgt_seq_len, in_dim)
            memory: Encoder output tensor of shape (batch, src_seq_len, in_dim)
            tgt_mask: Attention mask for the target sequence (typically causal mask)
            memory_mask: Attention mask for the encoder output
            tgt_key_padding_mask: Padding mask for the target sequence
            memory_key_padding_mask: Padding mask for the encoder output

        Returns:
            Output tensor of shape (batch, tgt_seq_len, out_dim)
        """
        # Self-attention block
        x = self.norm1(x)
        y = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        y = self.dropout1(y)
        if self.use_residual:
            y = x + y

        # Cross-attention block
        memory = self.memory_norm(memory)
        y = self.norm2(y)
        z = self.cross_attn(
            y, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False
        )[0]
        z = self.dropout2(z)
        if self.use_residual:
            z = y + z

        # Feedforward block
        z = self.norm3(z)
        w = self.linear2(self.dropout3(self.linear1(z)))
        if self.use_residual:
            w = z + w

        return w


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers.

    Args:
        in_dim: Input embedding dimension
        n_heads: Number of attention heads
        feedforward_dim: Dimension of the hidden layer in the feedforward network
        act_cls: Activation class for feedforward network
        dropout_rate: Dropout probability
        n_layers: Number of encoder layers to stack
        out_dim: The number of output embeddings. Defaults to in_dim
        use_residual: Whether to use residual connections
        use_final_norm: Whether to apply normalization after all layers
    """

    def __init__(
        self,
        in_dim: int,
        n_heads: int,
        feedforward_dim: int,
        act_cls: ActClass,
        dropout_rate: float,
        n_layers: int,
        out_dim: int | None = None,
        use_residual: bool = True,
        use_final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    in_dim=in_dim,
                    n_heads=n_heads,
                    hidden_dim=feedforward_dim,
                    act_cls=act_cls,
                    dropout_rate=dropout_rate,
                    out_dim=self.out_dim,
                    use_residual=use_residual,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.out_dim) if use_final_norm else None
        return

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers.

    Args:
        in_dim: Input embedding dimension
        n_heads: Number of attention heads
        hidden_dim: Dimension of the hidden layer in the feedforward network
        act_cls: Activation class for feedforward network
        dropout_rate: Dropout probability
        n_layers: Number of decoder layers to stack
        out_dim: The number of output embeddings. Defaults to in_dim.
        use_residual: Whether to use residual connections
        use_final_norm: Whether to apply normalization after all layers
    """

    def __init__(
        self,
        in_dim: int,
        n_heads: int,
        hidden_dim: int,
        act_cls: ActClass,
        dropout_rate: float,
        n_layers: int,
        out_dim: int | None = None,
        use_residual: bool = True,
        use_final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    in_dim=in_dim,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    act_cls=act_cls,
                    dropout_rate=dropout_rate,
                    out_dim=self.out_dim,
                    use_residual=use_residual,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.out_dim) if use_final_norm else None
        return

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all decoder layers.

        Args:
            tgt: Target input tensor of shape (batch, tgt_seq_len, in_dim)
            memory: Encoder output tensor of shape (batch, src_seq_len, in_dim)
            tgt_mask: Target attention mask (typically causal mask)
            memory_mask: Encoder attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Encoder padding mask

        Returns:
            Output tensor of shape (batch, tgt_seq_len, out_dim)
        """
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            tgt = self.norm(tgt)

        return tgt


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
