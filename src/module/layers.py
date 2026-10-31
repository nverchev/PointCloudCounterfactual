"""Neural network layers and point cloud operators."""

import abc
import warnings

from typing import Any, Final, Protocol, override, runtime_checkable

import torch
import torch.nn as nn

from torch.autograd import Function

from src.config.environment import DEBUG_MODE
from src.config.torch import ActClass
from src.utils.neighbour_ops import get_graph_features, graph_filtering

__all__ = [
    'BaseLayer',
    'EdgeConvLayer',
    'EdgeFeatures',
    'GraphFilter',
    'LinearLayer',
    'Max',
    'Mean',
    'Modulate',
    'PointsConvLayer',
    'Repeat',
    'SphereProjection',
    'TransferGrad',
    'Transpose',
    'debug_check',
]

DEFAULT_BN_MOMENTUM: Final = 0.1
"""Momentum of the batch normalization when no configuration sets it."""

FINAL_LAYER_GAIN: Final = 0.01
"""Gain of the layers ending in :class:`torch.nn.Identity`, which start close to zero."""

type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor


@runtime_checkable
class HasInplace(Protocol):
    """Protocol for classes that have an inplace option."""

    inplace: bool


class Max(nn.Module):
    """Keep the largest value along a dimension.

    Attributes:
        dim: dimension to reduce.
    """

    dim: Final[int]

    def __init__(self, dim: int) -> None:
        """Initialize.

        Args:
            dim: dimension to reduce.
        """
        super().__init__()
        self.dim = dim
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the dimension."""
        return x.max(self.dim)[0]

    @override
    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Mean(nn.Module):
    """Average along a dimension.

    Attributes:
        dim: dimension to reduce.
    """

    dim: Final[int]

    def __init__(self, dim: int) -> None:
        """Initialize.

        Args:
            dim: dimension to reduce.
        """
        super().__init__()
        self.dim = dim
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the dimension."""
        return x.mean(self.dim)

    @override
    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Repeat(nn.Module):
    """Tile a tensor, as :meth:`torch.Tensor.repeat` does.

    Attributes:
        sizes: times to repeat each dimension.
    """

    sizes: Final[tuple[int, ...]]

    def __init__(self, *sizes: int) -> None:
        """Initialize.

        Args:
            sizes: times to repeat each dimension.
        """
        super().__init__()
        self.sizes = sizes
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tile the input."""
        return x.repeat(*self.sizes)

    @override
    def extra_repr(self) -> str:
        return f'sizes={self.sizes}'


class Transpose(nn.Module):
    """Swap two dimensions.

    Attributes:
        dim0: first dimension to swap.
        dim1: second dimension to swap.
    """

    dim0: Final[int]
    dim1: Final[int]

    def __init__(self, dim0: int, dim1: int) -> None:
        """Initialize.

        Args:
            dim0: first dimension to swap.
            dim1: second dimension to swap.
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swap the dimensions."""
        return x.transpose(self.dim0, self.dim1)

    @override
    def extra_repr(self) -> str:
        return f'dim0={self.dim0}, dim1={self.dim1}'


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """Dense or convolutional layer with optional batch normalization, activation and residual.

    The initialization of the weights depends on the activation that follows them.

    Attributes:
        in_dim: Input channels.
        out_dim: Output channels.
        groups: Groups of the channels.
        batch_norm: Whether batch normalization follows the layer.
        bn_momentum: Momentum of the batch normalization.
        residual: Whether the input, repeated to match the output channels, is added to the output.
        module: The dense or convolutional layer.
        norm: The batch normalization, if any.
        act: The activation, if any.
    """

    in_dim: Final[int]
    out_dim: Final[int]
    groups: Final[int]
    batch_norm: Final[bool]
    bn_momentum: Final[float]
    residual: Final[bool]
    module: nn.Module
    norm: nn.Module | None
    act: nn.Module | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None,
        batch_norm: bool = True,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        groups: int = 1,
        residual: bool = False,
    ) -> None:
        """Initialize.

        Args:
            in_dim: input channels.
            out_dim: output channels.
            act_cls: activation class, ``None`` for no activation.
            batch_norm: whether batch normalization follows the layer.
            bn_momentum: momentum of the batch normalization.
            groups: groups of the channels.
            residual: whether the input, repeated to match the output channels, is added to the output.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.groups = groups
        self.batch_norm = batch_norm
        self.bn_momentum = bn_momentum
        self.residual = residual
        self.module = self.get_module()
        self.norm = self.get_norm_layer() if batch_norm else None
        self.act = self.get_activation(act_cls)
        self._initialize_weight()
        if DEBUG_MODE:
            warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer, the normalization and the activation, then the residual."""
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        if self.residual:
            return y + x.repeat_interleave(self.out_dim // self.in_dim + 1, 1)[:, : self.out_dim, ...]

        return y

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Return the wrapped dense or convolutional layer."""

    @abc.abstractmethod
    def get_norm_layer(self) -> nn.Module:
        """Return the normalization layer."""

    def _initialize_weight(self) -> None:
        weight = self.module.get_parameter('weight')
        if self.act is None:
            nn.init.xavier_normal_(weight, gain=1)
        elif isinstance(self.act, nn.Identity):
            nn.init.xavier_normal_(weight, gain=FINAL_LAYER_GAIN)
        elif isinstance(self.act, nn.ReLU):
            nn.init.kaiming_uniform_(weight, nonlinearity='relu')
        elif isinstance(self.act, nn.LeakyReLU):
            nn.init.kaiming_uniform_(weight, a=self.act.negative_slope)
        elif isinstance(self.act, nn.Hardtanh):
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('tanh'))

        return

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        """Get the activation, in place when it supports it."""
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace):
            act.inplace = True

        return act


class LinearLayer(BaseLayer):
    """Fully connected layer."""

    @override
    def get_module(self) -> nn.Module:
        return nn.Linear(self.in_dim, self.out_dim, bias=not self.batch_norm)

    @override
    def get_norm_layer(self) -> nn.Module:
        return nn.BatchNorm1d(self.out_dim, momentum=self.bn_momentum)


class PointsConvLayer(BaseLayer):
    """Layer shared by the points ``(B, C, N)``."""

    @override
    def get_module(self) -> nn.Module:
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=not self.batch_norm, groups=self.groups)

    @override
    def get_norm_layer(self) -> nn.Module:
        return nn.BatchNorm1d(self.out_dim, momentum=self.bn_momentum)


class EdgeConvLayer(BaseLayer):
    """Layer shared by the edges ``(B, C, N, n_neighbours)``."""

    @override
    def get_module(self) -> nn.Module:
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=not self.batch_norm, groups=self.groups)

    @override
    def get_norm_layer(self) -> nn.Module:
        return nn.BatchNorm2d(self.out_dim, momentum=self.bn_momentum)


class EdgeFeatures(nn.Module):
    """Build the edge features between every point and its neighbours in the feature space.

    Attributes:
        n_neighbours: neighbours of every point.
    """

    n_neighbours: Final[int]

    def __init__(self, n_neighbours: int) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the edge features ``(B, 2C, N, n_neighbours)`` of the features ``(B, C, N)``."""
        return get_graph_features(x, self.n_neighbours)[1]

    @override
    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class GraphFilter(nn.Module):
    """Sharpen a cloud by moving every point away from its nearest neighbours.

    Attributes:
        n_neighbours: neighbours of every point, the point itself included.
    """

    n_neighbours: Final[int]

    def __init__(self, n_neighbours: int = 4) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point, the point itself included.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter the coordinates ``(B, OUT_CHAN, M)``."""
        return graph_filtering(x, self.n_neighbours)

    @override
    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class Modulate(nn.Module):
    """Condition point features on the codeword by multiplying them."""

    @override
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Multiply the features ``(B, W, M)`` by the codeword ``(B, W)``."""
        return w.unsqueeze(2) * x


class SphereProjection(nn.Module):
    """Project points onto the unit sphere, along the coordinates."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide every point ``(B, D, M)`` by its norm."""
        return x / torch.linalg.vector_norm(x, dim=1, keepdim=True)


class TransferGrad(Function):
    """Straight-through gradient estimator transferring backward gradients."""

    @staticmethod
    @override
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return the quantized tensor unchanged."""
        from_tensor, _to_tensor = args
        return from_tensor

    @staticmethod
    @override
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        """Route the gradient to the tensor before quantization."""
        return None, grad_outputs[0]


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """Validate layer outputs during debug execution."""
    tensor = tensor_out[0] if isinstance(tensor_out, tuple) else tensor_out
    if not torch.isfinite(tensor).all():
        breakpoint()

    return
