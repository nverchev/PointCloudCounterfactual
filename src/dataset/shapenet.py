"""Dataset for the Shapenet dataset."""

import json
import pathlib
from typing import Any, override

import numpy as np
import torch
from numpy import typing as npt
from torch.utils.data import Dataset

from src.dataset.augmentations import augment_clouds, normalise
from src.config_options import Experiment
from src.data_types import Inputs, Targets
from src.dataset.protocols import PointCloudDataset, Partitions, SplitCreator


class ShapenetFlowSplit(PointCloudDataset):
    """Class for the Shapenet dataset sampled with 10.000 points."""

    def __init__(self, paths: list[pathlib.Path]) -> None:
        super().__init__()
        cfg_data = Experiment.get_config().data
        self.paths = paths
        self.pcd = list[npt.NDArray[Any]]()
        self.input_points = cfg_data.n_input_points
        self.resample = cfg_data.resample
        self.folder_id_list = list[str]()
        self.augment = augment_clouds()
        for path in paths:
            pc, _scale = normalise(np.load(path))
            self.pcd.append(pc.astype(np.float32))
            self.folder_id_list.append(path.parent.parent.name)

        set_id = set(self.folder_id_list)
        map_id = {folder_id: i for i, folder_id in enumerate(sorted(set_id))}
        self.labels = [map_id[folder_id] for folder_id in self.folder_id_list]

    def __len__(self) -> int:
        return len(self.pcd)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        np_cloud = self.pcd[index]
        label = torch.tensor(self.labels[index])
        index_pool = np.arange(np_cloud.shape[0])
        if self.resample:
            sampling = np.random.choice(index_pool, size=2 * self.input_points, replace=False)
            input_cloud_np = np_cloud[sampling[: self.input_points]]
            ref_cloud_np = np_cloud[sampling[self.input_points :]]
            clouds = [torch.from_numpy(input_cloud_np), torch.from_numpy(ref_cloud_np)]
            input_cloud, ref_cloud, *_ = self.augment(clouds)
        else:
            sampling = np.random.choice(index_pool, size=self.input_points, replace=False)
            input_cloud_np = np_cloud[sampling[: self.input_points]]
            clouds = [torch.from_numpy(input_cloud_np)]
            input_cloud, *_ = self.augment(clouds)
            ref_cloud = input_cloud

        return Inputs(cloud=input_cloud), Targets(ref_cloud=ref_cloud, label=label)


class ShapeNetDatasetFlow(SplitCreator):
    """This class creates the splits for the Shapenet Dataset."""

    classes: dict[str, str]
    data_dir: pathlib.Path
    shapenet_path: pathlib.Path
    paths: dict[Partitions, list[pathlib.Path]]

    def __init__(self):
        cfg = Experiment.get_config()
        user_cfg = cfg.user

        with open(user_cfg.path.metadata_dir / 'shapenet_PointFlow_classes.json') as f:
            self.classes = json.load(f)

        self.data_dir = user_cfg.path.data_dir
        self.shapenet_path = self.data_dir / 'ShapeNetCore.v2.PC15k'
        link = 'https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ'
        assert self.shapenet_path.exists, f'Download and extract dataset from here: {link}'
        folders = self.shapenet_path.glob('*')
        self.paths = dict[Partitions, list[pathlib.Path]]()

        if cfg.data.dataset.n_classes < 55:
            selected_classes = cfg.data.dataset.settings['select_classes']
            folders = [folder for folder in folders if self.classes[folder.name] in selected_classes]
            assert folders, 'class is not in dataset'
        for folder in folders:
            train_files = list((folder / 'train').glob('*'))
            val_files = list((folder / 'val').glob('*'))
            test_files = list((folder / 'test').glob('*'))
            train_val_files = train_files + val_files
            self.paths.setdefault(Partitions.train, []).extend(train_files)
            self.paths.setdefault(Partitions.train_val, []).extend(train_val_files)
            self.paths.setdefault(Partitions.val, []).extend(val_files)
            self.paths.setdefault(Partitions.test, []).extend(test_files)

    @override
    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        return ShapenetFlowSplit(self.paths[split])
