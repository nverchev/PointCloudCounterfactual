"""Builder of the ModelNet40 clouds of 2048 points."""

import logging

from typing import ClassVar, override

import h5py
import numpy as np

from numpy import typing as npt

from src.data.databuilder import DataBuilder
from src.data.split import Partitions
from src.utils.download import download_extract_zip

__all__ = ['ModelNetDataBuilder']


class ModelNetDataBuilder(DataBuilder):
    """ModelNet40 restricted to the selected classes.

    The files hold clouds already centered and fit in the unit ball. The labels follow the order
    of the selected classes. The validation partition takes one in ``val_every`` training clouds.
    """

    folder_raw: ClassVar[str] = 'modelnet40_hdf5_2048'
    url: ClassVar[str] = 'https://gaimfs.ugent.be/Public/Dataset/modelnet40_hdf5_2048.zip'
    val_every: ClassVar[int] = 6

    _clouds: dict[Partitions, npt.NDArray[np.float32]]
    _labels: dict[Partitions, npt.NDArray[np.int64]]

    @override
    def _download(self) -> None:
        download_extract_zip(self.dir_raw, self.url)
        return

    @override
    def _get_all_names(self) -> tuple[str, ...]:
        return tuple((self.dir_raw / 'shape_names.txt').read_text().split())

    @override
    def _get_partition_data(self, partition: Partitions, /) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        return self._clouds[partition], self._labels[partition]

    @override
    def _preprocess(self) -> None:
        self._clouds, self._labels = {}, {}
        new_labels = self._get_new_labels()
        for partition in (Partitions.train_val, Partitions.test):
            clouds, labels = self._read(partition)
            keep = new_labels[labels] >= 0
            self._clouds[partition], self._labels[partition] = clouds[keep], new_labels[labels[keep]]

        self._split_train_val()
        return

    def _get_new_labels(self) -> npt.NDArray[np.int64]:
        new_labels = np.full(len(self._all_names), -1, dtype=np.int64)
        for new_label, name in enumerate(self.class_names):
            new_labels[self._all_names.index(name)] = new_label

        return new_labels

    def _read(self, partition: Partitions) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        file_prefix = 'test' if partition is Partitions.test else 'train'
        clouds: list[npt.NDArray[np.float32]] = []
        labels: list[npt.NDArray[np.int64]] = []
        for path in sorted(self.dir_raw.glob(f'{file_prefix}*.h5')):
            logging.info(f'Reading {path}.')
            with h5py.File(path, 'r', locking=False) as file:
                clouds.append(_read_dataset(file, 'data').astype(np.float32))
                labels.append(_read_dataset(file, 'label').astype(np.int64).ravel())

        if not clouds:
            msg = f'No {file_prefix} files in {self.dir_raw}.'
            raise ValueError(msg)

        return np.concatenate(clouds, axis=0), np.concatenate(labels, axis=0)

    def _split_train_val(self) -> None:
        train_idx = list(range(len(self._clouds[Partitions.train_val])))
        val_idx = [train_idx.pop(i) for i in train_idx[:: -self.val_every]]
        for partition, idx in ((Partitions.train, train_idx), (Partitions.val, val_idx)):
            self._clouds[partition] = self._clouds[Partitions.train_val][idx]
            self._labels[partition] = self._labels[Partitions.train_val][idx]

        return


def _read_dataset(file: h5py.File, key: str) -> npt.NDArray[np.generic]:
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        msg = f'{key} is not a dataset in {file.filename}.'
        raise KeyError(msg)

    return dataset[()]
