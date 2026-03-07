"""Utilities for H5 data storage and retrieval."""

import pathlib
from typing import Any
import h5py
import numpy as np
from numpy import typing as npt

from src.data.structures import PCD


def load_h5(path: pathlib.Path, wild_str: str) -> PCD:
    """Loads processed point cloud data from H5 files into a PCD container."""
    pcs_list: list[npt.NDArray[Any]] = []
    pcs_2048_list: list[npt.NDArray[Any]] = []
    pcs_512_list: list[npt.NDArray[Any]] = []
    pcs_128_list: list[npt.NDArray[Any]] = []
    labels_list: list[npt.NDArray[Any]] = []
    std_list: list[npt.NDArray[Any]] = []
    h5_files = sorted(path.glob(wild_str))
    if not h5_files:
        raise FileNotFoundError(f'No H5 files matching {wild_str} found in {path}')

    for h5_name in h5_files:
        with h5py.File(h5_name, 'r') as f:
            required = ['pcd', 'pcd_512', 'pcd_128', 'labels', 'std']
            if any(data not in f for data in required):
                raise KeyError(f'Missing data in {h5_name}. Please run preprocessing first.')

            # Load and cast to float32 immediately
            pcs_list.append(extract_cloud_from_h5(f, 'pcd').astype(np.float32))
            if 'pcd_2048' in f:
                pcs_2048_list.append(extract_cloud_from_h5(f, 'pcd_2048').astype(np.float32))

            pcs_512_list.append(extract_cloud_from_h5(f, 'pcd_512').astype(np.float32))
            pcs_128_list.append(extract_cloud_from_h5(f, 'pcd_128').astype(np.float32))
            labels_list.append(extract_labels_from_h5(f, 'labels').astype(np.int64))
            std_list.append(extract_labels_from_h5(f, 'std').astype(np.float32))

    return PCD(
        pcd=np.concatenate(pcs_list, axis=0),
        pcd_512=np.concatenate(pcs_512_list, axis=0),
        pcd_128=np.concatenate(pcs_128_list, axis=0),
        labels=np.concatenate(labels_list, axis=0).ravel(),
        std=np.concatenate(std_list, axis=0).ravel(),
        pcd_2048=np.concatenate(pcs_2048_list, axis=0) if pcs_2048_list else None,
    )


def extract_cloud_from_h5(file: h5py.File, dataset_name: str) -> npt.NDArray[Any]:
    data = file[dataset_name]
    assert isinstance(data, h5py.Dataset)
    return data[()].astype(np.float32)


def extract_labels_from_h5(file: h5py.File, dataset_name: str) -> npt.NDArray[Any]:
    data = file[dataset_name]
    assert isinstance(data, h5py.Dataset)
    return data[()].astype(np.int64)


def save_preprocessed_h5(h5_path: pathlib.Path, pcd: PCD) -> None:
    """Utility to save preprocessed point cloud data (as a PCD object) to H5."""
    if h5_path.exists():
        return

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'w') as f:
        # pcd attributes should already be numpy arrays from preprocess_point_clouds
        f.create_dataset('pcd', data=pcd.pcd)
        if pcd.pcd_2048 is not None:
            f.create_dataset('pcd_2048', data=pcd.pcd_2048)

        f.create_dataset('pcd_512', data=pcd.pcd_512)
        f.create_dataset('pcd_128', data=pcd.pcd_128)
        f.create_dataset('labels', data=pcd.labels)
        f.create_dataset('std', data=pcd.std)

    return
