"""Module containing preprocessing utilities."""

from typing import Any

import numpy as np
from numpy import typing as npt
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.data.structures import PCD
from src.utils.neighbour_ops import match_repeated_points, dist_squared_numpy
from scipy.optimize import linear_sum_assignment


def normalize(cloud: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], float]:
    """Standard normalization to mean 0 and std 1."""
    cloud -= cloud.mean(axis=0)
    std = cloud.std()
    cloud /= std
    return cloud, std


def preprocess_equal_clusters(pc: np.ndarray) -> PCD:
    """Equal clusters preprocessing: 8192 -> 2048 -> 512 -> 128."""
    norm_pc, std = normalize(pc.copy())

    # KMeans to 2048
    if len(norm_pc) > 2048:
        km2048 = KMeans(n_clusters=2048, n_init='auto', random_state=42).fit(norm_pc)
        c2048 = km2048.cluster_centers_.astype(np.float32)
        labels2048 = km2048.labels_
    else:
        c2048 = norm_pc.astype(np.float32)
        labels2048 = np.arange(len(c2048))

    # Hierarchical levels
    km512 = KMeans(n_clusters=512, n_init='auto', random_state=42).fit(c2048)
    c512 = km512.cluster_centers_.astype(np.float32)

    km128 = KMeans(n_clusters=128, n_init='auto', random_state=42).fit(c512)
    c128 = km128.cluster_centers_.astype(np.float32)

    # Reorder to maintain hierarchy
    # 512 matched to 128 repeats
    pc_512 = match_repeated_points(c128, c512, factor=4)

    # 2048 matched to 512 repeats
    source_rep = np.repeat(pc_512, 4, axis=0)
    cost = dist_squared_numpy(source_rep, c2048)
    _, col_idx = linear_sum_assignment(cost)
    pc_2048 = c2048[col_idx]

    if len(norm_pc) > 8192:
        # Generate 8192 points after 2048 clusters have been reordered
        pc_8192 = []
        for idx_original in col_idx:
            cluster_points = norm_pc[labels2048 == idx_original]
            if len(cluster_points) == 0:
                cluster_points = c2048[idx_original : idx_original + 1]

            if len(cluster_points) >= 4:
                pc_8192.append(cluster_points[:4])
            else:
                indices = np.arange(len(cluster_points))
                rep_indices = np.tile(indices, (4 // len(cluster_points)) + 1)[:4]
                pc_8192.append(cluster_points[rep_indices])

        pc_final = np.concatenate(pc_8192, axis=0).astype(np.float32)
        pcd_2048_field = pc_2048
    else:
        pc_final = pc_2048
        pcd_2048_field = None

    return PCD(
        pcd=pc_final,
        pcd_512=pc_512,
        pcd_128=c128,
        labels=np.empty(0),
        std=np.array([std], dtype=np.float32),
        pcd_2048=pcd_2048_field,
    )


def preprocess_point_clouds(
    pcs: list[npt.NDArray[np.float32]], labels: npt.NDArray[Any] | None = None, desc: str = ''
) -> PCD:
    """
    Processes a list of point clouds and returns a PCD container with stacked arrays.
    """
    sample_pc: list[PCD] = []

    for pc in tqdm(pcs, desc=desc, leave=False):
        pcd = preprocess_equal_clusters(pc)
        sample_pc.append(pcd)

    pcd_2048_list = [pc.pcd_2048 for pc in sample_pc if pc.pcd_2048 is not None]
    pcd_2048 = np.stack(pcd_2048_list).astype(np.float32) if len(pcd_2048_list) == len(sample_pc) else None

    return PCD(
        pcd=np.stack([pc.pcd for pc in sample_pc]).astype(np.float32),
        pcd_512=np.stack([pc.pcd_512 for pc in sample_pc]).astype(np.float32),
        pcd_128=np.stack([pc.pcd_128 for pc in sample_pc]).astype(np.float32),
        labels=labels if labels is not None else np.empty(0),
        std=np.stack([pc.std for pc in sample_pc]).astype(np.float32).ravel(),
        pcd_2048=pcd_2048,
    )
