"""Module containing point cloud dataset splits and runtime augmentations."""

import enum

import torch
import numpy as np

from torch.utils.data import Dataset

from src.data.structures import Inputs, Targets, PCD
from src.data.augmentations import augment_clouds, jitter_cloud
from src.config.experiment import Experiment


class Partitions(enum.Enum):
    """Splits of the dataset."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class PointCloudSplit(Dataset[tuple[Inputs, Targets]]):
    """Base class for point cloud dataset splits with shared logic."""

    def __init__(self, pcd: PCD, class_names: list[str]) -> None:
        super().__init__()
        cfg_data = Experiment.get_config().data
        self.pcd = pcd
        self.labels = pcd.labels
        self.input_points = cfg_data.n_input_points
        self.resample = cfg_data.resample
        self.replace = cfg_data.sample_with_replacement
        self.augment = augment_clouds()
        self.jitter = jitter_cloud()
        self.class_names = class_names
        return

    def __len__(self) -> int:
        return self.pcd.pcs.shape[0]

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        np_cloud, np_label = self.pcd.pcs[index], self.labels[index]
        np_cloud_512_up = self.pcd.pcs_512_up[index]
        label = torch.tensor(np_label, dtype=torch.long)

        if not torch.is_inference_mode_enabled():
            if self.resample:
                index_pool = np.arange(np_cloud.shape[0])
                sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=self.replace)
                np_input_cloud = np_cloud[sampled_indices]
                np_input_cloud_512_up = np_cloud_512_up[sampled_indices]
            else:
                np_input_cloud = np_cloud[: self.input_points]
                np_input_cloud_512_up = np_cloud_512_up[: self.input_points]

            input_cloud = torch.from_numpy(np_input_cloud)
            input_cloud = self.jitter(input_cloud)
            ref_cloud = torch.from_numpy(np_cloud[: self.input_points])
            input_cloud_512_up = torch.from_numpy(np_input_cloud_512_up)
            input_cloud, ref_cloud, input_cloud_512_up, *_ = self.augment([input_cloud, ref_cloud, input_cloud_512_up])
        else:
            ref_cloud = input_cloud = torch.from_numpy(np_cloud[: self.input_points])
            input_cloud_512_up = torch.from_numpy(np_cloud_512_up[: self.input_points])

        cloud_512 = torch.from_numpy(self.pcd.pcs_512[index])
        cloud_128 = torch.from_numpy(self.pcd.pcs_128[index])
        cloud_128_up = torch.from_numpy(self.pcd.pcs_128_up[index])

        return Inputs(
            cloud=input_cloud,
            initial_sampling=input_cloud,
            cloud_512=cloud_512,
            cloud_128=cloud_128,
            cloud_512_up=input_cloud_512_up,
            cloud_128_up=cloud_128_up,
        ), Targets(ref_cloud=ref_cloud, label=label)
