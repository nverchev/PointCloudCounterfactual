import json
from abc import ABCMeta, abstractmethod
import enum
import pathlib
from collections.abc import Iterable, Callable
from typing import Any, Literal

import h5py  # type: ignore
import torch
import numpy as np
import numpy.typing as npt
import glob2  # type: ignore
from torch.utils.data import Dataset
from dry_torch import Model
from typing_extensions import override

from src.config_options import MainExperiment, Datasets
from src.data_structures import Inputs, Targets, W_Targets, Outputs, W_Inputs
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


def jitter_cloud() -> Callable[[torch.Tensor], torch.Tensor]:
    cfg_data = MainExperiment.get_config().data
    jitter_sigma = cfg_data.jitter_sigma
    jitter_clip = cfg_data.jitter_clip

    def _jitter(cloud: torch.Tensor) -> torch.Tensor:
        if jitter_sigma and jitter_clip:
            return jitter(cloud, jitter_sigma, jitter_clip)
        return cloud

    return _jitter


def augment_clouds() -> Callable[[Iterable[torch.Tensor]], Iterable[torch.Tensor]]:
    cfg_data = MainExperiment.get_config().data
    rotation_flag = cfg_data.rotation
    translation_and_scale_flag = cfg_data.translation

    def augment(clouds: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        if rotation_flag:
            rotate = random_rotation()
            clouds = map(rotate, clouds)
        if translation_and_scale_flag:
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
        cfg_data = MainExperiment.get_config().data
        self.pcd = pcd.astype(np.float32)
        self.indices = indices
        self.labels = labels
        self.input_points = cfg_data.n_input_points
        self.resample = cfg_data.resample
        self.augment = augment_clouds()
        self.jitter = jitter_cloud()

    def __len__(self) -> int:
        return self.pcd.shape[0]

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        np_cloud, np_neighbours_indices, label = self.pcd[index], self.indices[index], self.labels[index]

        # neighbours_indices = torch.from_numpy(np_neighbours_indices).long()
        label = torch.tensor(label, dtype=torch.long)
        if not torch.is_inference_mode_enabled():
            index_pool = np.arange(np_cloud.shape[0])
            sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=True)
            input_cloud = normalise(np_cloud)[0]
            cloud = torch.from_numpy(input_cloud)

            cloud = self.jitter(cloud)
            if self.resample:
                sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=True)
                np_ref_cloud = normalise(np_cloud)[0]
                ref_cloud = torch.from_numpy(np_ref_cloud)
                cloud, ref_cloud, *_ = self.augment([cloud, ref_cloud])
            else:
                cloud, *_ = self.augment([cloud])
                ref_cloud = cloud
        else:
            ref_cloud = cloud = torch.from_numpy(np_cloud)

        return Inputs(cloud=cloud), Targets(ref_cloud=ref_cloud, label=label)


class ShapenetFlowSplit(PointCloudDataset):
    def __init__(self, paths: list[pathlib.Path]) -> None:
        super().__init__()
        cfg_data = MainExperiment.get_config().data
        self.paths = paths
        self.pcd = list[npt.NDArray]()
        self.input_points = cfg_data.n_input_points
        self.resample = cfg_data.resample
        self.folder_id_list = list[str]()
        self.augment = augment_clouds()
        for path in paths:
            pc, scale = normalise(np.load(path))
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
            input_cloud_np = np_cloud[sampling[:self.input_points]]
            ref_cloud_np = np_cloud[sampling[self.input_points:]]
            clouds = [torch.from_numpy(input_cloud_np), torch.from_numpy(ref_cloud_np)]
            input_cloud, ref_cloud, *_ = self.augment(clouds)
        else:
            sampling = np.random.choice(index_pool, size=self.input_points, replace=False)
            input_cloud_np = np_cloud[sampling[:self.input_points]]
            clouds = [torch.from_numpy(input_cloud_np)]
            input_cloud, *_ = self.augment(clouds)
            ref_cloud = input_cloud

        return Inputs(cloud=input_cloud), Targets(ref_cloud=ref_cloud, label=label)


class Modelnet40Dataset(metaclass=Singleton):

    def __init__(self) -> None:
        cfg = MainExperiment.get_config()
        user_cfg = MainExperiment.get_config().user

        with open(user_cfg.path.metadata_dir / 'modelnet_classes.txt', 'r') as f:
            self.classes = f.read().splitlines()
        selected_classes = cfg.data.dataset.settings['select_classes']
        try:
            selected_labels = [self.classes.index(selected_class) for selected_class in selected_classes]
        except ValueError:
            raise ValueError(f'One of classes in {selected_classes} not in the dataset')

        label_map = {old: new for new, old in enumerate(selected_labels)}

        self.data_dir = user_cfg.path.data_dir
        self.modelnet_path = self.data_dir / 'modelnet40_hdf5_2048'
        self.download()
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in [Partitions.train, Partitions.test]:
            pcd, indices, labels = load_h5_modelnet(self.modelnet_path / f'*{split.name}*.h5',
                                                    cfg.data.n_input_points,
                                                    cfg.data.k)
            selected_indices: slice | np.ndarray[Any, np.dtype[np.bool_]]
            if cfg.data.dataset.n_classes == 40:
                selected_indices = slice(None)
            else:
                selected_indices = np.isin(labels, selected_labels)

            self.pcd[split] = pcd[selected_indices]
            self.indices[split] = indices[selected_indices]
            self.labels[split] = np.vectorize(label_map.get)(labels[selected_indices])

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


class ShapeNetDatasetFlow(metaclass=Singleton):

    def __init__(self):
        cfg = MainExperiment.get_config()
        user_cfg = MainExperiment.get_config().user
        with open(user_cfg.path.metadata_dir / 'shapenet_PointFlow_classes.json', 'r') as f:
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

    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        return ShapenetFlowSplit(self.paths[split])


def get_dataset(partition: Partitions) -> PointCloudDataset:
    cfg = MainExperiment.get_config()
    user_cfg = MainExperiment.get_config().user
    user_cfg.path.data_dir.mkdir(exist_ok=True)

    dataset_name = cfg.data.dataset.name
    dataset_dict: dict[Datasets, Any] = {Datasets.ModelNet: Modelnet40Dataset,
                                         Datasets.ShapenetFlow: ShapeNetDatasetFlow}
    dataset = dataset_dict[dataset_name]().split(partition)
    return dataset


class WDatasetWithProbs(Dataset[tuple[W_Inputs, W_Targets]]):

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: VQVAE,
            classifier: Model[Inputs, torch.Tensor]
    ) -> None:
        self.dataset = dataset
        if hasattr(dataset, '__len__'):
            self.dataset_len = dataset.__len__()
        else:
            raise ValueError('Dataset does not have ``__len__`` method')
        self.autoencoder = autoencoder
        self.device: torch.device = next(autoencoder.parameters()).device
        self.classifier = classifier
        self.classifier.module = classifier.module.eval()
        self.max_batch = 64

    def __getitems__(self, index_list: list[int]) -> list[tuple[W_Inputs, W_Targets]]:
        self.autoencoder = self.autoencoder.train(not torch.is_inference_mode_enabled())

        # Load all data points
        dataset_batch = [self.dataset[i] for i in index_list]
        batched_cloud = torch.stack([data[0].cloud for data in dataset_batch]).to(self.device)
        batched_indices = torch.cat([data[0].indices for data in dataset_batch]).to(self.device)
        outputs = list[tuple[W_Inputs, W_Targets]]()

        for i in range(0, len(index_list), self.max_batch):
            batch_slice = slice(i, i + self.max_batch)
            cloud_batch = batched_cloud[batch_slice]
            indices_batch = batched_indices[batch_slice]
            batch_inputs = Inputs(cloud=cloud_batch, indices=indices_batch)

            batch_logits = self._run_classifier(batch_inputs)
            batch_ae_data = self._run_autoencoder(cloud_batch, indices_batch)
            for (w_q, one_hot_idx), logits in zip(batch_ae_data, batch_logits):
                outputs.append((W_Inputs(w_q, logits), W_Targets(one_hot_idx=one_hot_idx, logits=logits)))

        return outputs

    @torch.inference_mode()
    def _run_autoencoder(self,
                         batched_cloud: torch.Tensor,
                         batched_indices: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        data = self.autoencoder.encode(batched_cloud, batched_indices)
        w, one_hot_idx = self.autoencoder.quantise(data.w_q)

        return list(zip(data.w_q, one_hot_idx))

    @torch.inference_mode()
    def _run_classifier(self, batched_cloud: Inputs) -> list[torch.Tensor]:
        probs = self.classifier(batched_cloud)
        return list(probs)

    def __len__(self) -> int:
        return self.dataset_len


class ReconstructedDataset(Dataset[tuple[Inputs, Targets]]):
    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: VQVAE,
    ) -> None:
        self.dataset = dataset
        if hasattr(dataset, '__len__'):
            self.dataset_len = dataset.__len__()
        else:
            raise ValueError('Dataset does not have ``__len__`` method')
        self.autoencoder = autoencoder.eval()
        self.device: torch.device = next(autoencoder.parameters()).device

    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        dataset_batch = [self.dataset[i] for i in index_list]
        batched_cloud = torch.stack([data[0].cloud for data in dataset_batch]).to(self.device)
        batched_indices = torch.cat([data[0].indices for data in dataset_batch]).to(self.device)
        batched_labels = torch.cat([data[1].label.unsqueeze(0) for data in dataset_batch]).to(self.device)
        new_labels = self._modify_labels(batched_labels)
        recons = self._run_autoencoder(batched_cloud, batched_indices, new_labels).recon
        batch = list[tuple[Inputs, Targets]]()
        for recon, label in zip(recons, new_labels):
            batch.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))
        return batch

    @torch.inference_mode()
    def _run_autoencoder(self,
                         batched_cloud: torch.Tensor,
                         batched_indices: torch.Tensor,
                         labels: torch.Tensor) -> Outputs:
        _unused = labels
        with self.autoencoder.double_encoding:
            data = self.autoencoder.encode(batched_cloud, batched_indices)
            return self.autoencoder.decode(data)

    def _modify_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels

    def __len__(self) -> int:
        return self.dataset_len


class CounterfactualDataset(ReconstructedDataset):
    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: VQVAE,
            target_label: Literal['original'] | int = 'original',
            target_value: float = 1.,
            num_classes: int = 2,
    ) -> None:
        super().__init__(dataset, autoencoder)
        self.num_classes = num_classes
        self.target_label = target_label
        self.target_value = target_value

    @override
    @torch.inference_mode()
    def _run_autoencoder(self,
                         batched_cloud: torch.Tensor,
                         batched_indices: torch.Tensor,
                         target_dim: torch.Tensor) -> Outputs:

        with self.autoencoder.double_encoding:
            data = self.autoencoder.encode(batched_cloud, batched_indices)
            target = torch.zeros_like(data.z_c)
            target[:, target_dim] = 1
            data.z_c = (1 - self.target_value) * data.z_c + self.target_value * target
            return self.autoencoder.decode(data)

    @override
    def _modify_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.target_label == 'original':
            target_dim = labels
        else:
            target_dim = self.target_label * torch.ones_like(labels, dtype=torch.long)
        return target_dim

    def __len__(self) -> int:
        return self.dataset_len
