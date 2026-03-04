"""Module containing point cloud dataset splits and runtime augmentations."""

import enum

import torch

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
        self.augment = augment_clouds()
        self.jitter = jitter_cloud()
        self.class_names = class_names
        return

    def __len__(self) -> int:
        return self.pcd.pcd.shape[0]

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        # Choose the right cloud level based on n_input_points
        if self.input_points == 2048 and self.pcd.pcd_2048 is not None:
            np_cloud = self.pcd.pcd_2048[index]
        else:
            np_cloud = self.pcd.pcd[index]

        cloud_512 = torch.from_numpy(self.pcd.pcd_512[index])
        cloud_128 = torch.from_numpy(self.pcd.pcd_128[index])
        cloud_2048 = torch.from_numpy(self.pcd.pcd_2048[index]) if self.pcd.pcd_2048 is not None else torch.empty(0)

        np_label = self.labels[index]
        label = torch.tensor(np_label, dtype=torch.long)
        if not torch.is_inference_mode_enabled():
            np_input_cloud = np_cloud[: self.input_points]
            input_cloud = torch.from_numpy(np_input_cloud)
            input_cloud = self.jitter(input_cloud)
            ref_cloud = torch.from_numpy(np_cloud[: self.input_points])
            clouds = (input_cloud, cloud_512, cloud_128, cloud_2048, ref_cloud)
            input_cloud, cloud_512, cloud_128, cloud_2048, ref_cloud = self.augment(clouds)
        else:
            ref_cloud = input_cloud = torch.from_numpy(np_cloud[: self.input_points])

        inputs = Inputs(
            cloud=input_cloud,
            cloud_512=cloud_512,
            cloud_128=cloud_128,
            cloud_2048=cloud_2048,
        )
        targets = Targets(ref_cloud=ref_cloud, label=label)
        return inputs, targets
