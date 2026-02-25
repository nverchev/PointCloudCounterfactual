"""Module containing preprocessing utilities."""

from typing import Any
import numpy as np
from numpy import typing as npt
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.data.structures import PCD


def normalize(cloud: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], float]:
    """Standard normalization to mean 0 and std 1."""
    cloud -= cloud.mean(axis=0)
    std = cloud.std()
    cloud /= std
    return cloud, std


def kmeans_downsample(cloud: npt.NDArray[Any], n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample a point cloud using k-means clustering.
    Returns:
        centroids: npt.NDArray[np.float32] - The downsampled point cloud.
        assignments: npt.NDArray[np.int32] - Index of the centroid each original point is assigned to.
    """
    if len(cloud) <= n_points:
        return cloud.astype(np.float32), np.arange(len(cloud), dtype=np.int32)

    kmeans = KMeans(n_clusters=n_points, n_init='auto', random_state=42)
    kmeans.fit(cloud)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    assignments = kmeans.labels_
    if assignments is None:
        raise ValueError('Assignments not available.')

    assignments = assignments.astype(np.int32)

    return centroids, assignments


def upsample_from_assignments(centroids: npt.NDArray[Any], assignments: npt.NDArray[np.int32]) -> npt.NDArray[Any]:
    """Creates an upsampled point cloud where each point takes the value of its assigned centroid."""
    return centroids[assignments]


def preprocess_cloud_hierarchy(pc: npt.NDArray[np.float32]) -> PCD:
    """
    Full preprocessing pipeline for a single point cloud.
    1. Normalize.
    2. Downsample to 512 + assignments.
    3. Downsample pc_512 to 128 + assignments.
    4. Create pc_512_up and pc_128_up.
    """
    norm_pc, std = normalize(pc.copy())

    # Hierarchical downsampling
    pc_512, assign_512 = kmeans_downsample(norm_pc, 512)
    pc_128, assign_128 = kmeans_downsample(pc_512, 128)

    # Upsampling
    pc_512_up = upsample_from_assignments(pc_512, assign_512)
    pc_128_up = upsample_from_assignments(pc_128, assign_128)

    return PCD(
        pcs=norm_pc,
        pcs_512=pc_512,
        pcs_128=pc_128,
        pcs_512_up=pc_512_up,
        pcs_128_up=pc_128_up,
        labels=np.empty(0),
        std=np.array([std]),
    )


def preprocess_point_clouds(
    pcs: list[npt.NDArray[np.float32]], labels: npt.NDArray[Any] | None = None, desc: str = ''
) -> PCD:
    """
    Processes a list of point clouds and returns a PCD container with stacked arrays.
    """
    sample_pc: list[PCD] = []

    for pc in tqdm(pcs, desc=desc, leave=False):
        pcd = preprocess_cloud_hierarchy(pc)
        sample_pc.append(pcd)

    return PCD(
        pcs=np.stack([pc.pcs for pc in sample_pc]).astype(np.float32),
        pcs_512=np.stack([pc.pcs_512 for pc in sample_pc]).astype(np.float32),
        pcs_128=np.stack([pc.pcs_128 for pc in sample_pc]).astype(np.float32),
        pcs_512_up=np.stack([pc.pcs_512_up for pc in sample_pc]).astype(np.float32),
        pcs_128_up=np.stack([pc.pcs_128_up for pc in sample_pc]).astype(np.float32),
        labels=labels if labels is not None else np.empty(0),
        std=np.stack([pc.std for pc in sample_pc]).astype(np.float32),
    )
