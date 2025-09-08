"""A utility module providing various helper functions and classes for data processing and manipulation."""

from typing import Any

import logging
import pathlib
import zipfile
import numpy as np
import numpy.typing as npt
import torch
import requests  # type: ignore
import h5py  # type: ignore
from sklearn.neighbors import KDTree  # type: ignore

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning  # type: ignore

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except ImportError:
    InsecureRequestWarning = None  # type: ignore
    pass


class Singleton(type):
    """A metaclass that ensures only one instance of a class is created."""
    _instances = dict[type, Any]()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Allows a temporary change using the "with" clause
class UsuallyFalse:
    """A class that provides a context manager for temporarily changing a boolean value."""
    _value: bool = False

    def __bool__(self) -> bool:
        return self._value

    def __enter__(self):
        self._value = True

    def __exit__(self, *_):
        self._value = False


def download_zip(target_folder: pathlib.Path, url: str) -> None:
    """Downloads and extracts a zip file from a URL to a target folder."""
    # check the folder already exists
    if not target_folder.exists():
        r = requests.get(url, verify=False)
        zip_path = target_folder.with_suffix('.zip')
        with zip_path.open('wb') as zip_file:
            zip_file.write(r.content)
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(target_folder.parent)
    return


def index_k_neighbours(pcs: list[npt.NDArray], k: int) -> npt.NDArray:
    """Finds the k nearest neighbors for each point in a list of point clouds."""
    indices_list = []
    for pc in pcs:
        kdtree = KDTree(pc)
        indices = kdtree.query(pc, k, return_distance=False)
        indices_list.append(indices.reshape(-1, k))
    return np.stack(indices_list)


def load_h5_modelnet(path: pathlib.Path,
                     wild_str: str,
                     input_points: int,
                     k: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Loads and processes ModelNet data from H5 files, including point clouds, indices and labels."""
    pcd_list: list[npt.NDArray] = []
    indices_list: list[npt.NDArray] = []
    labels_list: list[npt.NDArray] = []
    for h5_name in path.glob(wild_str):
        with h5py.File(h5_name, 'r+') as f:
            logging.info('Load: %s', h5_name)
            # Dataset is already normalized
            pcs = f['data'][:].astype('float32')
            pcs = pcs[:, :input_points, :]
            label = f['label'][:].astype('int64')
            index_k = f'index_{k}'
            if index_k in f.keys():
                index = f[index_k][:].astype(np.short)
            else:
                index = index_k_neighbours(pcs, k).astype(np.short)
                f.create_dataset(index_k, data=index)

        pcd_list.append(pcs)
        indices_list.append(index)
        labels_list.append(label)
    pcd = np.concatenate(pcd_list, axis=0)
    indices = np.concatenate(indices_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return pcd, indices, labels.ravel()


def print_embedding_usage(one_hot_idx: torch.Tensor):
    """Prints statistics about the usage of embeddings in a one-hot encoded tensor."""
    one_hot_idx = one_hot_idx.float().mean(0)  # average dataset values
    print_statistics('Index Usage', one_hot_idx.tolist())
    print(torch.histc(one_hot_idx))
    return


def print_statistics(title: str, test_outcomes: list[float]):
    """Prints various statistical measures for a list of test outcomes."""
    if not len(test_outcomes):
        print(f'No test performed for "{title}"')
        return
    np_test_outcomes = np.array(test_outcomes)
    print(title)
    print(f'Number of tests: {len(np_test_outcomes):} Min: {np_test_outcomes.min():.4e} '
          f'Max: {np_test_outcomes.max():.4e} Mean: {np_test_outcomes.mean():.4e} Std: {np_test_outcomes.std():.4e}')
