"""Point cloud operations for nearest neighbor computations and graph feature extraction."""

from typing import cast

import pykeops  # type: ignore
import torch

from pykeops.torch import LazyTensor  # type: ignore

pykeops.set_verbose(False)


def square_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor | LazyTensor:
    """Compute the squared distance between two point clouds using PyKeOps or PyTorch backend."""
    if t1.device.type == 'cuda':
        return pykeops_square_distance(t1, t2)
    else:
        return torch_square_distance(t1, t2)


def pykeops_square_distance(t1: torch.Tensor, t2: torch.Tensor) -> LazyTensor:
    """Compute the squared distance between two point clouds using PyKeOps backend."""
    t1_lazy = LazyTensor(t1[:, :, None, :])
    t2_lazy = LazyTensor(t2[:, None, :, :])
    dist = ((t1_lazy - t2_lazy) ** 2).sum(-1)
    return dist


def pykeops_cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> LazyTensor:
    """Compute the squared distance between two point clouds using PyKeOps backend."""
    t1_lazy = LazyTensor(t1[:, :, None, :])
    t2_lazy = LazyTensor(t2[:, None, :, :])
    sim = cast(LazyTensor, (t1_lazy * t2_lazy)).sum(-1)
    return sim


def torch_square_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Compute the squared distance between two point clouds using PyTorch backend."""
    # [batch, points, features]
    t2 = t2.transpose(-1, -2)
    dist = -2 * torch.matmul(t1, t2)
    dist += torch.sum(t1**2, -1, keepdim=True)
    dist += torch.sum(t2**2, -2, keepdim=True)
    return dist


def self_square_distance(t1: torch.Tensor) -> torch.Tensor:
    """Compute the squared distance between a point cloud and itself."""
    t2 = t1.transpose(-1, -2)
    square_component = torch.sum(t1**2, -2, keepdim=True)
    dist = torch.tensor(-2) * torch.matmul(t2, t1)
    dist += square_component
    dist += square_component.transpose(-1, -2)
    return dist


def torch_knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find the k nearest neighbors using PyTorch backend."""
    d_ij = self_square_distance(x)
    return d_ij.topk(k=k, largest=False)[1]


def pykeops_knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find the k nearest neighbors using PyKeOps backend."""
    x = x.transpose(2, 1).contiguous()
    d_ij = pykeops_square_distance(x, x)
    indices = d_ij.argKmin(k, dim=2)
    return indices


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Find the k nearest neighbors for each point in a point cloud."""
    if x.device.type == 'cuda':
        return pykeops_knn(x, k)
    else:
        return torch_knn(x, k)


def get_neighbours(x: torch.Tensor, indices: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the nearest neighbors of each point in a point cloud."""
    batch, n_feat, n_points = x.size()
    if indices.numel():
        indices = indices
    else:
        indices = knn(x, n_neighbors)  # (batch_size, num_points, n_neighbors)

    indices_expanded = indices.contiguous().view(batch, 1, n_neighbors * n_points).expand(-1, n_feat, -1)
    neighbours = torch.gather(x, 2, indices_expanded).view(batch, n_feat, n_points, n_neighbors)
    return indices, neighbours


def get_local_covariance(x: torch.Tensor, indices: torch.Tensor, n_neighbors: int = 16) -> torch.Tensor:
    """Compute the local covariance matrix for each point in a point cloud."""
    neighbours = get_neighbours(x, indices, n_neighbors)[1]
    neighbours -= neighbours.mean(3, keepdim=True)
    covariances = torch.matmul(neighbours.transpose(1, 2), neighbours.permute(0, 2, 3, 1))
    x = torch.cat([x, covariances.flatten(start_dim=2).transpose(1, 2)], dim=1).contiguous()
    return x


def graph_max_pooling(x: torch.Tensor, indices: torch.Tensor, n_neighbors: int = 16) -> torch.Tensor:
    """Perform max pooling operation over the nearest neighbors."""
    neighbours = get_neighbours(x, indices, n_neighbors)[1]
    max_pooling = torch.max(neighbours, dim=-1)[0]
    return max_pooling


def get_graph_features(
    x: torch.Tensor, indices: torch.Tensor, n_neighbors: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract graph features by concatenating the difference between points and their nearest neighbors."""
    indices_out, neighbours = get_neighbours(x, indices, n_neighbors)
    x = x.unsqueeze(3).expand(-1, -1, -1, n_neighbors)
    feature = torch.cat([neighbours - x, x], dim=1).contiguous()
    # (batch_size, 2 * num_dims, num_points, n_neighbors)
    return indices_out, feature


def graph_filtering(x: torch.Tensor, n_neighbors: int = 4) -> torch.Tensor:
    """Apply a graph-based filtering operation on the point cloud."""
    neighbours = get_neighbours(x, n_neighbors=n_neighbors, indices=torch.empty(0))[1]
    neighbours = neighbours[..., 1:]  # the closest neighbor is the point itself
    diff = x.unsqueeze(-1).expand(-1, -1, -1, n_neighbors - 1) - neighbours
    dist = torch.sqrt(abs((diff**2).sum(1)))
    sigma = torch.clamp(dist[..., 0:1].mean(1, keepdim=True), min=0.005)
    norm_dist = dist / sigma
    weights = torch.exp(-norm_dist)
    x_weight = weights.sum(2).unsqueeze(1).expand(-1, 3, -1)
    weighted_neighbours = weights.unsqueeze(1).expand(-1, 3, -1, -1) * neighbours
    return (1 + x_weight) * x - weighted_neighbours.sum(-1)
