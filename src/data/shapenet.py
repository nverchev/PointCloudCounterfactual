"""Dataset for the Shapenet dataset."""

import json
import pathlib
from typing import Any, override

import numpy as np
import torch
from numpy import typing as npt
from torch.utils.data import Dataset

from src.data.augmentations import augment_clouds, normalise, jitter_cloud
from src.config.experiment import Experiment
from src.data.structures import Inputs, Targets
from src.data.protocols import PointCloudDataset, Partitions, SplitCreator
from src.utils.download import download_extract_zip


class ShapenetFlowSplit(PointCloudDataset):
    """Class for the Shapenet dataset sampled with 10.000 points."""

    def __init__(self, paths: list[pathlib.Path]) -> None:
        super().__init__()
        cfg_data = Experiment.get_config().data
        self.paths = paths
        self.pcd = list[npt.NDArray[Any]]()
        self.input_points = cfg_data.n_input_points
        self.resample = cfg_data.resample
        self.replace = cfg_data.sample_with_replacement
        self.folder_id_list = list[str]()
        self.augment = augment_clouds()
        self.jitter = jitter_cloud()
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
        np_cloud, np_label = self.pcd[index], self.labels[index]
        label = torch.tensor(np_label, dtype=torch.long)
        if not torch.is_inference_mode_enabled():
            if self.resample:
                index_pool = np.arange(np_cloud.shape[0])
                sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=self.replace)
                np_cloud = np_cloud[sampled_indices]
            else:
                np_cloud = np_cloud[: self.input_points]

            np_cloud = normalise(np_cloud)[0]
            cloud = torch.from_numpy(np_cloud)
            cloud = self.jitter(cloud)
            ref_cloud = torch.from_numpy(normalise(np_cloud[: self.input_points])[0])
            cloud, ref_cloud, *_ = self.augment([cloud, ref_cloud])

        else:
            ref_cloud = cloud = torch.from_numpy(np_cloud[: self.input_points])

        return Inputs(cloud=cloud), Targets(ref_cloud=ref_cloud, label=label)


class ShapeNetDatasetFlow(SplitCreator):
    """This class creates the splits for the Shapenet Dataset."""

    def __init__(self):
        cfg = Experiment.get_config()
        user_cfg = cfg.user

        with open(user_cfg.path.metadata_dir / 'shapenet_PointFlow_classes.json') as f:
            self.classes = json.load(f)

        self.data_dir = user_cfg.path.data_dir
        self.dir = self.data_dir / 'ShapeNetCore.v2.PC15k'
        if not self.dir.exists():
            self._download()

        self.paths = dict[Partitions, list[pathlib.Path]]()
        folders = self.dir.glob('*')
        if cfg.data.dataset.n_classes < len(self.classes):
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

        return

    def _download(self) -> None:
        url = 'https://gaimfs.ugent.be/Public/Dataset/ShapeNetCore.v2.PC15k.zip'
        return download_extract_zip(target_folder=self.dir, url=url)

    @override
    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        return ShapenetFlowSplit(self.paths[split])
