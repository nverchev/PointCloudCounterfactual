import json
from abc import ABCMeta, abstractmethod
import enum
import pathlib
from typing import Iterable, Callable, Any

import h5py  # type: ignore
import torch
import numpy as np
import numpy.typing as npt

import glob2  # type: ignore
from torch.utils.data import Dataset

from src.config_options import ParentExperiment, Datasets
from src.data_structures import Inputs, Targets
from src.autoencoder import VQVAE
from src.utils import download_zip, load_h5_modelnet
from src.utils import Singleton


class Partitions(enum.Enum):
    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


def normalise(cloud: npt.NDArray) -> tuple[npt.NDArray, float]:
    cloud -= cloud.mean(axis=0)
    std = np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    cloud /= std
    return cloud, std


def jitter(cloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
    jitter_noise = torch.randn(cloud.shape) * torch.tensor(sigma)
    new_cloud = cloud.clone()
    new_cloud += torch.clamp(jitter_noise, min=-clip, max=clip)
    return new_cloud


def random_rotation() -> Callable[[torch.Tensor], torch.Tensor]:
    theta = torch.tensor(2 * torch.pi) * torch.rand(1)
    s = torch.sin(theta)
    rotation_matrix = torch.eye(2) * torch.cos(theta)
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s

    def rotate(cloud: torch.Tensor) -> torch.Tensor:
        new_cloud = cloud.clone()
        new_cloud[:, [0, 2]] = cloud[:, [0, 2]].mm(rotation_matrix)
        return new_cloud

    return rotate


def random_scale_and_translate() -> Callable[[torch.Tensor], torch.Tensor]:
    scale = torch.rand(1, 3) * 5 / 6 + 2 / 3
    translate = torch.rand(1, 3) * 0.4 - 0.2

    def scale_and_translate(cloud: torch.Tensor) -> torch.Tensor:
        new_cloud = cloud.clone()
        new_cloud *= scale
        new_cloud += translate
        return new_cloud

    return scale_and_translate


def augment_clouds() -> Callable[[Iterable[torch.Tensor]], Iterable[torch.Tensor]]:
    cfg_data = ParentExperiment.get_child_config().data
    rotation = cfg_data.rotation
    translation_and_scale = cfg_data.translation

    def augment(clouds: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        if rotation:
            rotate = random_rotation()
            clouds = map(rotate, clouds)
        if translation_and_scale:
            scale_and_translate = random_scale_and_translate()
            clouds = map(scale_and_translate, clouds)
        return clouds

    return augment


class PointCloudDataset(Dataset[tuple[Inputs, Targets]], metaclass=ABCMeta):

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        ...


class Modelnet40Split(PointCloudDataset):
    def __init__(self, pcd, indices, labels) -> None:
        super().__init__()
        self.pcd = pcd.astype(np.float32)
        self.indices = indices
        self.labels = labels
        self.input_points = ParentExperiment.get_child_config().data.input_points
        self.resample = ParentExperiment.get_child_config().data.resample
        self.augment = augment_clouds()

    def __len__(self) -> int:
        return self.pcd.shape[0]

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        cloud, neighbours_indices, label = self.pcd[index], self.indices[index], self.labels[index]
        cloud = torch.from_numpy(cloud)
        neighbours_indices = torch.from_numpy(neighbours_indices).long()
        cloud, *_ = self.augment([cloud])
        return Inputs(cloud=cloud, indices=neighbours_indices), Targets(ref_cloud=cloud, label=label)


class Modelnet40Dataset(metaclass=Singleton):

    def __init__(self) -> None:
        cfg = ParentExperiment.get_child_config()
        with open(cfg.user.path.metadata_dir / 'modelnet_classes.txt', 'r') as f:
            self.classes = f.read().splitlines()

        self.data_dir = cfg.user.path.data_dir
        self.modelnet_path = self.data_dir / 'modelnet40_hdf5_2048'
        self.download()
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in [Partitions.train, Partitions.test]:
            self.pcd[split], self.indices[split], self.labels[split] = load_h5_modelnet(
                self.modelnet_path / f'*{split.name}*.h5',
                cfg.data.input_points,
                cfg.data.k,
            )
            selected_classes: list[str] = cfg.data.dataset.settings['select_classes']
            selected_indices: slice | np.ndarray[Any, np.dtype[np.bool_]]
            if selected_classes == ['All']:
                selected_indices = slice(None)
            else:
                try:
                    selected_labels = [self.classes.index(selected_class) for selected_class in selected_classes]
                except ValueError:
                    print(f'One of classes in {selected_classes} not in {split.name} dataset')
                    raise
                else:
                    selected_indices = np.isin(self.labels[split], selected_labels)

            self.pcd[split] = self.pcd[split][selected_indices]
            self.indices[split] = self.indices[split][selected_indices]
            self.labels[split] = self.labels[split][selected_indices]

    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        if split == Partitions.train_val:
            assert Partitions.val not in self.pcd.keys(), 'train dataset has already been split'
            split = Partitions.train
        elif split in [Partitions.train, Partitions.val] and Partitions.val not in self.pcd.keys():
            self.train_val_to_train_and_val()
        return Modelnet40Split(pcd=self.pcd[split], indices=self.indices[split], labels=self.labels[split])

    def download(self) -> None:
        url = 'https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1'
        return download_zip(target_folder=self.modelnet_path, url=url)

    def train_val_to_train_and_val(self, val_every: int = 6) -> None:
        train_idx = list(range(self.pcd[Partitions.train].shape[0]))
        val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
        # partition train into train and val
        for new_split, new_split_idx in ((Partitions.val, val_idx), (Partitions.train, train_idx)):
            self.pcd[new_split] = self.pcd[Partitions.train][new_split_idx]
            self.indices[new_split] = self.indices[Partitions.train][new_split_idx]
            self.labels[new_split] = self.labels[Partitions.train][new_split_idx]


class ShapenetAtlasSplit(PointCloudDataset):
    def __init__(self, paths, labels) -> None:
        super().__init__()
        cfg = ParentExperiment.get_child_config()
        self.paths = paths
        self.input_points = cfg.data.input_points
        self.resample = cfg.data.resample
        self.label_index = list(labels.values())
        self.augment = augment_clouds()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        path = self.paths[index]
        cloud = np.load(path).astype(np.float32)
        index_pool = np.arange(cloud.shape[0])
        if self.resample:
            sampling = np.random.choice(index_pool, size=2 * self.input_points, replace=True)
            norm_cloud_np, scale = normalise(cloud[sampling[:self.input_points]])
            ref_cloud_np = normalise(cloud[sampling[self.input_points:]])[0]
            clouds = [torch.from_numpy(norm_cloud_np), torch.from_numpy(ref_cloud_np)]
            input_cloud, ref_cloud, *_ = self.augment(clouds)
        else:
            sampling = np.random.choice(index_pool, size=self.input_points, replace=False)
            norm_cloud_np, scale = normalise(cloud[sampling[:self.input_points]])
            clouds = [torch.from_numpy(norm_cloud_np)]
            input_cloud, *_ = self.augment(clouds)
            ref_cloud = input_cloud

        return Inputs(cloud=input_cloud), Targets(ref_cloud=ref_cloud, scale=torch.tensor(scale))


class ShapeNetDatasetAtlas(metaclass=Singleton):

    def __init__(self) -> None:
        cfg = ParentExperiment.get_child_config()
        with open(cfg.user.path.metadata_dir / 'shapenet_AtlasNet_classes.json', 'r') as f:
            self.classes = json.load(f)
        self.data_dir = cfg.user.path.data_dir
        self.shapenet_path = self.data_dir / 'shapenet'
        if not self.shapenet_path.exists():
            self.shapenet_path.mkdir()
            self.to_numpy()
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.input_points = cfg.data.input_points
        folders = self.shapenet_path.iterdir()
        self.paths: dict[Partitions, list[pathlib.Path]] = {}
        for folder in folders:
            files = sorted(folder.iterdir())
            first_split = int(len(files) * (1 - self.val_ratio - self.test_ratio))
            second_split = int(len(files) * (1 - self.test_ratio))
            self.paths.setdefault(Partitions.train, []).extend(files[:first_split])
            self.paths.setdefault(Partitions.train_val, []).extend(files[:second_split])
            self.paths.setdefault(Partitions.val, []).extend(files[first_split:second_split])
            self.paths.setdefault(Partitions.test, []).extend(files[second_split:])

    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        return ShapenetAtlasSplit(self.paths[split], self.classes)

    def to_numpy(self) -> None:
        original_path = self.data_dir / 'customShapeNet'
        if not original_path.exists():
            'Download shapenet as in https://github.com/TheoDEPRELLE/AtlasNetV2'
        for code, label in self.classes.items():
            code_file = original_path / code / 'ply'
            files = code_file.glob('*.ply')
            shapenet_label_path = self.shapenet_path / label
            if not shapenet_label_path.exists():
                shapenet_label_path.mkdir()
            i = 0
            for file in files:
                if file.find('*.') > -1:  # Here * is not a wildcard but a character
                    continue
                np_file = shapenet_label_path / str(i)
                if np_file.exists():
                    i += 1
                    continue
                try:
                    pc = np.loadtxt(file, skiprows=12, usecols=(0, 1, 2))
                except UnicodeDecodeError:
                    print(f'File with path: \n {file} \n is corrupted.')
                else:
                    np.save(np_file, pc)
                finally:
                    i += 1


def get_dataset(partition: Partitions) -> PointCloudDataset:
    cfg = ParentExperiment.get_child_config()
    cfg.user.path.data_dir.mkdir(exist_ok=True)

    dataset_name = cfg.data.dataset.name
    dataset_dict: dict[Datasets, Any] = {Datasets.ModelNet: Modelnet40Dataset,
                                         Datasets.ShapenetAtlas: ShapeNetDatasetAtlas}
    dataset = dataset_dict[dataset_name]().split(partition)
    return dataset


class WDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            module: VQVAE,
    ) -> None:
        self.dataset = dataset
        if hasattr(dataset, '__len__'):
            self.dataset_len = dataset.__len__()
        else:
            raise ValueError("Dataset does not have '__len__' method")
        self.module = module
        self.device: torch.device = next(module.parameters()).device

    def __getitems__(self, index_list: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.module = self.module.train(not torch.is_inference_mode_enabled())
        dataset_batch = [self.dataset[i] for i in index_list]
        batched_cloud = torch.stack([data[0].cloud for data in dataset_batch]).to(self.device)
        batched_indices = torch.cat([data[0].indices for data in dataset_batch]).to(self.device)
        return self._run_module(batched_cloud, batched_indices)

    @torch.inference_mode()
    def _run_module(self,
                    batched_cloud: torch.Tensor,
                    batched_indices: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        data = self.module.encode(batched_cloud, batched_indices)
        one_hot_idx = self.module.quantise(data.w_q)[1]
        return list(zip(data.w_q, one_hot_idx))

    def __len__(self) -> int:
        return self.dataset_len
