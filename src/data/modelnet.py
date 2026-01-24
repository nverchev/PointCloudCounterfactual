"""Dataset for the ModelNet40 dataset."""

import logging
import pathlib
from typing import Any, override, cast

import h5py
import numpy as np
import torch
from numpy import typing as npt
from torch.utils.data import Dataset

from src.data.augmentations import augment_clouds, jitter_cloud, normalise
from src.config.experiment import Experiment
from src.data.structures import Inputs, Targets
from src.data.protocols import PointCloudDataset, Partitions, SplitCreator
from src.utils.download import download_extract_zip


class ModelNet40Split(PointCloudDataset):
    """Class for the ModelNet40 dataset sampled with 2048 points."""

    def __init__(self, pcd: npt.NDArray[Any], labels: npt.NDArray[Any]) -> None:
        super().__init__()
        cfg_data = Experiment.get_config().data
        self.pcd = pcd.astype(np.float32)
        self.labels = labels
        self.input_points = cfg_data.n_input_points
        self.resample_target = cfg_data.resample_target
        self.sample_with_replacement = cfg_data.sample_with_replacement
        self.augment = augment_clouds()
        self.jitter = jitter_cloud()
        return

    def __len__(self) -> int:
        return self.pcd.shape[0]

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        np_cloud, np_label = self.pcd[index], self.labels[index]
        label = torch.tensor(np_label, dtype=torch.long)
        if not torch.is_inference_mode_enabled():
            index_pool = np.arange(np_cloud.shape[0])
            sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=self.sample_with_replacement)
            input_cloud = normalise(np_cloud[sampled_indices])[0]
            cloud = torch.from_numpy(input_cloud)
            cloud = self.jitter(cloud)
            if self.resample_target:
                sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=False)
                np_ref_cloud = normalise(np_cloud[sampled_indices])[0]
                ref_cloud = torch.from_numpy(np_ref_cloud[sampled_indices])
                cloud, ref_cloud, *_ = self.augment([cloud, ref_cloud])
            else:
                ref_cloud = torch.from_numpy(normalise(np_cloud[: self.input_points])[0])

        else:
            ref_cloud = cloud = torch.from_numpy(np_cloud[: self.input_points])

        return Inputs(cloud=cloud), Targets(ref_cloud=ref_cloud, label=label)


class ModelNet40Dataset(SplitCreator):
    """This class creates the splits for the ModelNet40 Dataset"""

    classes: list[str]
    data_dir: pathlib.Path
    modelnet_path: pathlib.Path
    pcd: dict[Partitions, npt.NDArray[Any]]
    indices: dict[Partitions, npt.NDArray[Any]]
    labels: dict[Partitions, npt.NDArray[Any]]

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        user_cfg = cfg.user

        with open(user_cfg.path.metadata_dir / 'modelnet_classes.txt') as f:
            self.classes = f.read().splitlines()
        selected_classes = cfg.data.dataset.settings['select_classes']
        try:
            selected_labels = [self.classes.index(selected_class) for selected_class in selected_classes]
        except ValueError as ve:
            raise ValueError(f'One of classes in {selected_classes} not in the dataset') from ve

        label_map = {old: new for new, old in enumerate(selected_labels)}

        self.data_dir = user_cfg.path.data_dir
        self.modelnet_path = self.data_dir / 'modelnet40_hdf5_2048'
        self._download()
        self.pcd, self.labels = {}, {}
        for split in [Partitions.train, Partitions.test]:
            pcd, labels = self.load_h5(
                path=self.modelnet_path,
                wild_str=f'*{split.name}*.h5',
                input_points=cfg.data.n_input_points,
            )
            selected_indices: slice | np.ndarray[Any, np.dtype[np.bool_]]
            if cfg.data.dataset.n_classes == 40:
                selected_indices = slice(None)
            else:
                selected_indices = np.isin(labels, selected_labels)

            self.pcd[split] = pcd[selected_indices]
            self.labels[split] = np.vectorize(label_map.get)(labels[selected_indices])

        return

    @override
    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        if split == Partitions.train_val:
            assert Partitions.val not in self.pcd.keys(), 'train dataset has already been split'
            split = Partitions.train
        elif split in [Partitions.train, Partitions.val] and Partitions.val not in self.pcd.keys():
            self._train_val_to_train_and_val()

        return ModelNet40Split(pcd=self.pcd[split], labels=self.labels[split])

    def _download(self) -> None:
        url = 'https://gaimfs.ugent.be/Public/Dataset/modelnet40_hdf5_2048.zip'
        return download_extract_zip(target_folder=self.modelnet_path, url=url)

    def _train_val_to_train_and_val(self, val_every: int = 6) -> None:
        train_idx = list(range(self.pcd[Partitions.train].shape[0]))
        val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
        # partition train into train and val
        for new_split, new_split_idx in ((Partitions.val, val_idx), (Partitions.train, train_idx)):
            self.pcd[new_split] = self.pcd[Partitions.train][new_split_idx]
            self.labels[new_split] = self.labels[Partitions.train][new_split_idx]

        return

    @staticmethod
    def load_h5(path: pathlib.Path, wild_str: str, input_points: int) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Loads and processes ModelNet data from H5 files, including point clouds and labels."""
        pcd_list: list[npt.NDArray[Any]] = []
        labels_list: list[npt.NDArray[Any]] = []

        for h5_name in path.glob(wild_str):
            with h5py.File(h5_name, 'r+') as f:
                logging.info('Load: %s', h5_name)

                # Use cast to resolve Dataset ambiguity
                pcs_ds = cast(h5py.Dataset, f['data'])
                pcs = pcs_ds[:].astype('float32')
                pcs = pcs[:, :input_points, :]

                label_ds = cast(h5py.Dataset, f['label'])
                label = label_ds[:].astype('int64')

            pcd_list.append(pcs)
            labels_list.append(label)

        pcd = np.concatenate(pcd_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return pcd, labels.ravel()
