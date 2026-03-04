"""Point cloud operations for nearest neighbor computations and graph feature extraction."""

import pykeops  # type: ignore
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from pykeops.torch import LazyTensor  # type: ignore

pykeops.set_verbose(False)


def pykeops_square_distance(t1: torch.Tensor, t2: torch.Tensor) -> LazyTensor:
    """Compute the squared distance between two point clouds using PyKeOps backend."""
    t1_lazy = LazyTensor(t1[:, :, None, :])
    t2_lazy = LazyTensor(t2[:, None, :, :])
    dist = ((t1_lazy - t2_lazy) ** 2).sum(-1)
    return dist


def torch_square_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Compute the squared distance between two point clouds using PyTorch backend."""
    # [batch, points, features]
    t2 = t2.transpose(-1, -2)
    dist = -2 * torch.matmul(t1, t2)
    dist += torch.sum(t1**2, -1, keepdim=True)
    dist += torch.sum(t2**2, -2, keepdim=True)
    return dist.clamp(min=0)


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


def pykeops_1nn(x: torch.Tensor) -> torch.Tensor:
    """Find the k nearest neighbors using PyKeOps backend."""
    x = x.transpose(2, 1).contiguous()
    d_ij = pykeops_square_distance(x, x)
    indices = d_ij.argKmin(1, dim=2)
    return indices


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


def farthest_point_sample(x: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Perform farthest point sampling on a batch of point clouds.
    Args:
        x: (batch, n_feat, n_points_orig)
        n_points: number of points to sample
    Returns:
        indices: (batch, n_points)
    """
    device = x.device
    batch_size, n_feat, n_points_orig = x.shape
    centroids = torch.zeros(batch_size, n_points, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, n_points_orig, device=device) * 1e10
    farthest = torch.randint(0, n_points_orig, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    # Reshape x to (batch, n_points_orig, n_feat) for easier distance calculation
    x_trans = x.transpose(1, 2).contiguous()

    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = x_trans[batch_indices, farthest, :].view(batch_size, 1, n_feat)
        dist = torch.sum((x_trans - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def farthest_point_sample_numpy(xyz: np.ndarray, n_points: int) -> np.ndarray:
    """Farthest Point Sampling in NumPy."""
    n = xyz.shape[0]
    indices = np.zeros(n_points, dtype=np.int32)
    distances = np.ones(n) * 1e10
    farthest = np.random.randint(0, n)
    for i in range(n_points):
        indices[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return indices


def dist_squared_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient squared Euclidean distance between two sets of points."""
    return np.sum(a**2, axis=-1)[:, None] + np.sum(b**2, axis=-1)[None, :] - 2 * np.dot(a, b.T)


def match_repeated_points(source: np.ndarray, target: np.ndarray, factor: int) -> np.ndarray:
    """Repeat source points and match them to target points using linear sum assignment."""
    source_rep = np.repeat(source, factor, axis=0)
    cost = dist_squared_numpy(source_rep, target)
    _, col_idx = linear_sum_assignment(cost)
    return target[col_idx]


def get_bijective_assignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get bijective assignment between two point clouds."""
    B, N, _ = x.shape
    cost = torch_square_distance(x, y).cpu().numpy()
    batch_idx = torch.zeros((B, N), dtype=torch.long, device=x.device)
    for b in range(B):
        _, idx_b = linear_sum_assignment(cost[b])
        batch_idx[b] = torch.from_numpy(idx_b).to(x.device)

    return batch_idx


def cluster_wise_assign(x: torch.Tensor, noise: torch.Tensor, perms: torch.Tensor, k: int) -> torch.Tensor:
    """Assign noise to x cluster-wise using precomputed permutations."""
    B, N, C = x.shape
    NK, rest = divmod(N, k)
    if rest != 0:
        raise ValueError(f'N must be divisible by k, got N={N}, k={k}')

    x_c = x.view(B, NK, k, C)
    n_c = noise.view(B, NK, k, C)
    dist = torch_square_distance(x_c, n_c)
    perms = perms.to(x.device)
    P = perms.shape[0]  # P = k!
    # dist: (B, NK, k, k) -> (B, NK, P, k, k)
    dist_exp = dist.unsqueeze(2).expand(-1, -1, P, -1, -1)
    # p: (P, k) -> (B, NK, P, k, 1)
    p_exp = perms[None, None, :, :, None].expand(B, NK, -1, -1, -1).long()
    total_perm_dist = torch.gather(dist_exp, 4, p_exp).squeeze(-1).sum(dim=-1)
    best_p_idx = total_perm_dist.argmin(dim=-1)
    best_p = perms[best_p_idx]
    n_ordered = torch.gather(n_c, 2, best_p[:, :, :, None].expand(-1, -1, -1, C))
    return n_ordered.view(B, N, C)
