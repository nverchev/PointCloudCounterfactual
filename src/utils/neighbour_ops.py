"""Neighbourhood operations and graph filtering using PyKeOps."""

import pykeops
import torch

from pykeops.torch import LazyTensor

from src.data.structures import OUT_CHAN

pykeops.set_verbose(False)

__all__ = [
    'cpu_knn',
    'cpu_square_distance',
    'get_graph_features',
    'get_neighbours',
    'graph_filtering',
    'knn',
    'pykeops_knn',
    'pykeops_square_distance',
    'self_square_distance',
    'square_distance',
]


def square_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor | LazyTensor:
    """Compute pairwise squared distance."""
    if t1.device.type == 'cuda':
        return pykeops_square_distance(t1, t2)

    return cpu_square_distance(t1, t2)


def pykeops_square_distance(t1: torch.Tensor, t2: torch.Tensor) -> LazyTensor:
    """Compute pairwise squared distance using PyKeOps LazyTensor."""
    t1_lazy = LazyTensor(t1[:, :, None, :])
    t2_lazy = LazyTensor(t2[:, None, :, :])
    return ((t1_lazy - t2_lazy) ** 2).sum(-1)


def cpu_square_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared distance on CPU."""
    t2_t = t2.transpose(-1, -2)
    dist = -2 * torch.matmul(t1, t2_t)
    dist += torch.sum(t1**2, -1, keepdim=True)
    dist += torch.sum(t2_t**2, -2, keepdim=True)
    return dist


def self_square_distance(t1: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared distances within the same point set."""
    t2 = t1.transpose(-1, -2)
    square_component = torch.sum(t1**2, -2, keepdim=True)
    dist = torch.tensor(-2) * torch.matmul(t2, t1)
    dist += square_component
    dist += square_component.transpose(-1, -2)
    return dist


def knn(x: torch.Tensor, n_neighbours: int) -> torch.Tensor:
    """Compute nearest neighbours."""
    if x.device.type == 'cuda':
        return pykeops_knn(x, n_neighbours)

    return cpu_knn(x, n_neighbours)


def cpu_knn(x: torch.Tensor, n_neighbours: int) -> torch.Tensor:
    """Compute nearest neighbours on CPU."""
    d_ij = self_square_distance(x)
    return d_ij.topk(k=n_neighbours, largest=False, dim=-1)[1]


def pykeops_knn(x: torch.Tensor, n_neighbours: int) -> torch.Tensor:
    """Compute nearest neighbours on GPU using PyKeOps."""
    x_t = x.transpose(2, 1).contiguous()
    d_ij = pykeops_square_distance(x_t, x_t)
    return d_ij.argKmin(n_neighbours, dim=2)


def get_neighbours(
    x: torch.Tensor, n_neighbours: int, indices: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the nearest neighbours of every point.

    Args:
        x: features ``(B, C, N)``.
        n_neighbours: neighbours of every point.
        indices: neighbours to gather, computed from ``x`` when left out.

    Returns:
        The neighbour indices and the gathered neighbours ``(B, C, N, n_neighbours)``.
    """
    batch, n_feat, n_points = x.size()
    if indices is None:
        indices = knn(x, n_neighbours)

    indices_expanded = indices.contiguous().view(batch, 1, n_neighbours * n_points).expand(-1, n_feat, -1)
    neighbours = torch.gather(x, 2, indices_expanded).view(batch, n_feat, n_points, n_neighbours)
    return indices, neighbours


def get_graph_features(
    x: torch.Tensor, n_neighbours: int = 20, indices: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the edge features between every point and its neighbours.

    Args:
        x: features ``(B, C, N)``.
        n_neighbours: neighbours of every point.
        indices: neighbours to build the edges on, computed from ``x`` when
            left out.

    Returns:
        The neighbour indices and the edge features ``(B, 2C, N, n_neighbours)``.
    """
    indices, neighbours = get_neighbours(x, n_neighbours, indices)
    x_expanded = x.unsqueeze(3).expand(-1, -1, -1, n_neighbours)
    return indices, torch.cat([neighbours - x_expanded, x_expanded], dim=1).contiguous()


def graph_filtering(x: torch.Tensor, n_neighbours: int = 4) -> torch.Tensor:
    """Apply graph filtering to point coordinates."""
    neighbours = get_neighbours(x, n_neighbours=n_neighbours)[1][..., 1:]
    diff = x.unsqueeze(-1).expand(-1, -1, -1, n_neighbours - 1) - neighbours
    dist = torch.sqrt(torch.clamp((diff**2).sum(1), min=1e-12))
    sigma = torch.clamp(dist[..., 0:1].mean(1, keepdim=True), min=0.005)
    norm_dist = dist / sigma
    weights = torch.exp(-norm_dist)
    x_weight = weights.sum(2).unsqueeze(1).expand(-1, OUT_CHAN, -1)
    weighted_neighbours = weights.unsqueeze(1).expand(-1, OUT_CHAN, -1, -1) * neighbours
    return (1 + x_weight) * x - weighted_neighbours.sum(-1)
