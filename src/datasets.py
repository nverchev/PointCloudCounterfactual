"""Module that defines the datasets for the project."""

import abc
import json
from abc import ABCMeta, abstractmethod
import enum
import pathlib
from collections.abc import Iterable, Callable
from typing import Any, Generic, TypeVar, Generator, Literal
import itertools

import torch
import torch.distributed as dist
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
from drytorch import Model
from typing_extensions import override

from src.config_options import Experiment, Datasets
from src.data_structures import Inputs, Targets, WTargets, Outputs, WInputs
from src.autoencoder import AbstractVQVAE, VQVAE, CounterfactualVQVAE
from src.utils import download_zip, load_h5_modelnet
from src.utils import Singleton

VQ = TypeVar('VQ', bound=AbstractVQVAE)


class Partitions(enum.Enum):
    """Splits of the dataset."""
    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


def normalise(cloud: npt.NDArray) -> tuple[npt.NDArray, float]:
    """Standard normalization to unit sphere."""
    cloud -= cloud.mean(axis=0)
    std = np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    cloud /= std
    return cloud, std


def jitter(cloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
    """Add noise to points coordinates."""
    jitter_noise = torch.randn(cloud.shape) * torch.tensor(sigma)
    new_cloud = cloud.clone()
    new_cloud += torch.clamp(jitter_noise, min=-clip, max=clip)
    return new_cloud


def random_rotation() -> Callable[[torch.Tensor], torch.Tensor]:
    """Define random rotation to be applied to input and reference clouds."""
    theta = torch.tensor(2 * torch.pi) * torch.rand(1)
    s = torch.sin(theta)
    rotation_matrix = torch.eye(2) * torch.cos(theta)
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s

    def _rotate(cloud: torch.Tensor) -> torch.Tensor:
        new_cloud = cloud.clone()
        new_cloud[:, [0, 2]] = cloud[:, [0, 2]].mm(rotation_matrix)
        return new_cloud

    return _rotate


def random_scale_and_translate() -> Callable[[torch.Tensor], torch.Tensor]:
    """Define random scaling and translation to be applied to input and reference clouds."""
    scale = torch.rand(1, 3) * 5 / 6 + 2 / 3
    translate = torch.rand(1, 3) * 0.4 - 0.2

    def _scale_and_translate(cloud: torch.Tensor) -> torch.Tensor:
        new_cloud = cloud.clone()
        new_cloud *= scale
        new_cloud += translate
        return new_cloud

    return _scale_and_translate


class CloudAugmenter:
    """Picklable augmentation class for rotation, scaling, and translation."""

    def __init__(self, rotation: bool, translation_and_scale: bool):
        self.rotation = rotation
        self.translation_and_scale = translation_and_scale

    def __call__(self, clouds: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        if self.rotation:
            rotate = random_rotation()
            clouds = map(rotate, clouds)
        if self.translation_and_scale:
            scale_and_translate = random_scale_and_translate()
            clouds = map(scale_and_translate, clouds)
        return clouds


class CloudJitterer:
    """Picklable jitter class."""

    def __init__(self, jitter_sigma: float | None, jitter_clip: float | None):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

    def __call__(self, cloud: torch.Tensor) -> torch.Tensor:
        if self.jitter_sigma and self.jitter_clip:
            return jitter(cloud, self.jitter_sigma, self.jitter_clip)
        return cloud


def augment_clouds() -> CloudAugmenter:
    """Create a callable for augmentation based on configuration."""
    cfg_data = Experiment.get_config().data
    return CloudAugmenter(
        rotation=cfg_data.rotation,
        translation_and_scale=cfg_data.translation
    )


def jitter_cloud() -> CloudJitterer:
    """Create jitter callable based on configuration."""
    cfg_data = Experiment.get_config().data
    return CloudJitterer(
        jitter_sigma=cfg_data.jitter_sigma,
        jitter_clip=cfg_data.jitter_clip
    )


class PointCloudDataset(Dataset[tuple[Inputs, Targets]], metaclass=ABCMeta):

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        ...


class ModelNet40Split(PointCloudDataset):
    """Class for the ModelNet40 dataset sampled with 2048 points."""

    def __init__(self, pcd, indices, labels) -> None:
        super().__init__()
        cfg_data = Experiment.get_config().data
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
            input_cloud = normalise(np_cloud[sampled_indices])[0]
            cloud = torch.from_numpy(input_cloud)

            cloud = self.jitter(cloud)
            if self.resample:
                sampled_indices = np.random.choice(index_pool, size=self.input_points, replace=True)
                np_ref_cloud = normalise(np_cloud)[0]
                ref_cloud = torch.from_numpy(np_ref_cloud[sampled_indices])
                cloud, ref_cloud, *_ = self.augment([cloud, ref_cloud])
            else:
                cloud, *_ = self.augment([cloud])
                ref_cloud = cloud
        else:
            ref_cloud = cloud = torch.from_numpy(np_cloud)

        return Inputs(cloud=cloud), Targets(ref_cloud=ref_cloud, label=label)


class ShapenetFlowSplit(PointCloudDataset):
    """Class for the Shapenet dataset sampled with 10.000 points."""

    def __init__(self, paths: list[pathlib.Path]) -> None:
        super().__init__()
        cfg_data = Experiment.get_config().data
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


class AbstractSingleton(Singleton, abc.ABCMeta):
    """Combining abstract and singleton metaclass."""


class SplitCreator(abc.ABC, metaclass=AbstractSingleton):
    """Abstract class that creates the splits for a dataset. Instantiated only once for efficiency."""

    @abc.abstractmethod
    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        """Retrieve the split."""


class ModelNet40Dataset(SplitCreator):
    """This class creates the splits for the ModelNet40 Dataset"""

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        user_cfg = cfg.user

        with open(user_cfg.path.metadata_dir / 'modelnet_classes.txt') as f:
            self.classes = f.read().splitlines()
        selected_classes = cfg.data.dataset.settings['select_classes']
        try:
            selected_labels = [self.classes.index(selected_class) for selected_class in selected_classes]
        except ValueError:
            raise ValueError(f'One of classes in {selected_classes} not in the dataset')

        label_map = {old: new for new, old in enumerate(selected_labels)}

        self.data_dir = user_cfg.path.data_dir
        self.modelnet_path = self.data_dir / 'modelnet40_hdf5_2048'
        self._download()
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in [Partitions.train, Partitions.test]:
            pcd, indices, labels = load_h5_modelnet(path=self.modelnet_path,
                                                    wild_str=f'*{split.name}*.h5',
                                                    input_points=cfg.data.n_input_points,
                                                    k=cfg.data.k)
            selected_indices: slice | np.ndarray[Any, np.dtype[np.bool_]]
            if cfg.data.dataset.n_classes == 40:
                selected_indices = slice(None)
            else:
                selected_indices = np.isin(labels, selected_labels)

            self.pcd[split] = pcd[selected_indices]
            self.indices[split] = indices[selected_indices]
            self.labels[split] = np.vectorize(label_map.get)(labels[selected_indices])

    @override
    def split(self, split: Partitions) -> Dataset[tuple[Inputs, Targets]]:
        if split == Partitions.train_val:
            assert Partitions.val not in self.pcd.keys(), 'train dataset has already been split'
            split = Partitions.train
        elif split in [Partitions.train, Partitions.val] and Partitions.val not in self.pcd.keys():
            self._train_val_to_train_and_val()
        return ModelNet40Split(pcd=self.pcd[split], indices=self.indices[split], labels=self.labels[split])

    def _download(self) -> None:
        url = 'https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1'
        return download_zip(target_folder=self.modelnet_path, url=url)

    def _train_val_to_train_and_val(self, val_every: int = 6) -> None:
        train_idx = list(range(self.pcd[Partitions.train].shape[0]))
        val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
        # partition train into train and val
        for new_split, new_split_idx in ((Partitions.val, val_idx), (Partitions.train, train_idx)):
            self.pcd[new_split] = self.pcd[Partitions.train][new_split_idx]
            self.indices[new_split] = self.indices[Partitions.train][new_split_idx]
            self.labels[new_split] = self.labels[Partitions.train][new_split_idx]


class ShapeNetDatasetFlow(SplitCreator):
    """This class creates the splits for the Shapenet Dataset."""

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
            selected_classes = cfg.data.dataset.opt_settings['select_classes']
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


class BaseVQDataset(Dataset, Generic[VQ], metaclass=ABCMeta):
    """Base dataset for VQVAE models with common functionality."""

    max_batch = 64

    def __init__(self, dataset: Dataset[tuple[Inputs, Targets]], autoencoder: VQ) -> None:
        self.dataset = dataset
        if not hasattr(dataset, '__len__'):
            raise ValueError('Dataset does not have ``__len__`` method')
        self.dataset_len = len(dataset)
        self.autoencoder = autoencoder
        self.device: torch.device = next(autoencoder.parameters()).device

    def __len__(self) -> int:
        return self.dataset_len

    def _get_data(self, index_list: list[int]) -> Generator[tuple[Inputs, torch.Tensor], None, None]:
        """Common data loading logic."""
        dataset_batch = [self.dataset[i] for i in index_list]
        batched_cloud = torch.stack([data[0].cloud for data in dataset_batch]).to(self.device)
        batched_indices = torch.cat([data[0].indices for data in dataset_batch]).to(self.device)
        batched_labels = torch.cat([data[1].label.unsqueeze(0) for data in dataset_batch]).to(self.device)

        for i in range(0, len(index_list), self.max_batch):
            batch_slice = slice(i, i + self.max_batch)
            cloud_batch = batched_cloud[batch_slice]
            indices_batch = batched_indices[batch_slice]
            batch_inputs = Inputs(cloud=cloud_batch, indices=indices_batch)
            batch_labels = batched_labels[batch_slice]
            yield batch_inputs, batch_labels

    @abstractmethod
    def __getitems__(self, index_list: list[int]) -> list:
        """Subclasses must implement this method."""
        pass


class ClassifierMixin:
    """Mixin for datasets that need classifier functionality."""

    def __init__(self, classifier: Model[Inputs, torch.Tensor], **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    @torch.inference_mode()
    def _run_classifier(self, batch_inputs: Inputs) -> torch.Tensor:
        """Run classifier on batch inputs."""
        return self.classifier(batch_inputs)


class WDataset(BaseVQDataset[VQ], Dataset[tuple[WInputs, WTargets]]):
    """Dataset for training inner autoencoder with discrete codes."""

    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        self.autoencoder = self.autoencoder.train(not torch.is_inference_mode_enabled())
        batch_data = []

        for batch_inputs, labels in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs)
            batch_w_q, batch_w_e, batch_one_hot_idx = batch_ae_data.w_q, batch_ae_data.w_e, batch_ae_data.one_hot_idx

            for w_q, w_e, one_hot_idx in zip(batch_w_q, batch_w_e, batch_one_hot_idx):
                batch_data.append((WInputs(w_q), WTargets(w_e=w_e, one_hot_idx=one_hot_idx)))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(self, inputs: Inputs) -> Outputs:
        """Run autoencoder encoding and quantization."""
        data = self.autoencoder.encode(inputs)
        data.w_e, data.one_hot_idx = self.autoencoder.quantizer.quantize(data.w_q)
        return data


class WDatasetWithLogits(ClassifierMixin, WDataset[CounterfactualVQVAE]):
    """W Dataset with classifier logits for conditional training."""

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: CounterfactualVQVAE,
            classifier: Model[Inputs, torch.Tensor]
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier
        )

    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        self.autoencoder = self.autoencoder.train(not torch.is_inference_mode_enabled())
        batch_data = []

        for batch_inputs, labels in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs)
            batch_logits = self._run_classifier(batch_inputs)
            batch_w_q, batch_w_e, batch_one_hot_idx = batch_ae_data.w_q, batch_ae_data.w_e, batch_ae_data.one_hot_idx

            for w_q, w_e, one_hot_idx, logit in zip(batch_w_q, batch_w_e, batch_one_hot_idx, batch_logits):
                batch_data.append((
                    WInputs(w_q, logit),
                    WTargets(w_e=w_q, one_hot_idx=one_hot_idx, logits=logit)
                ))

        return batch_data


class WDatasetWithLogitsFrozen(WDatasetWithLogits):
    """W Dataset with classifier logits for conditional training."""

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: CounterfactualVQVAE,
            classifier: Model[Inputs, torch.Tensor]
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier
        )
        self.w_dataset: list[tuple[WInputs, WTargets]] = []
        for batch_idx in itertools.batched(range(len(self)), 32):
            self.w_dataset.extend(super().__getitems__(list(batch_idx)))

    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        return [self.w_dataset[i] for i in index_list]


class ReconstructedDataset(BaseVQDataset[VQVAE], Dataset[tuple[Inputs, Targets]]):
    """Dataset with reconstructed inputs after double encoding."""

    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs)
            recons = batch_ae_data.recon

            for recon, label in zip(recons, batch_labels):
                batch_data.append((
                    Inputs(cloud=recon),
                    Targets(ref_cloud=recon, label=label)
                ))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(self, inputs: Inputs) -> Outputs:
        """Run autoencoder with double encoding."""
        with self.autoencoder.double_encoding:
            return self.autoencoder(inputs)


class ReconstructedDatasetWithLogits(ClassifierMixin, ReconstructedDataset):
    """Reconstructed dataset with classifier logits."""

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: CounterfactualVQVAE,
            classifier: Model[Inputs, torch.Tensor]
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier
        )

    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        self.classifier.module = self.classifier.module.eval()
        batch_data = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_logits = self._run_classifier(batch_inputs)
            batch_ae_data = self._run_autoencoder(batch_inputs, batch_logits)
            recons = batch_ae_data.recon

            for recon, label in zip(recons, batch_labels):
                batch_data.append((
                    Inputs(cloud=recon),
                    Targets(ref_cloud=recon, label=label)
                ))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(self, inputs: Inputs, batched_logits: None | torch.Tensor = None) -> Outputs:
        """Run autoencoder with logits conditioning."""
        if batched_logits is None:
            batched_logits = self._run_classifier(inputs)

        with self.autoencoder.double_encoding:
            data = self.autoencoder.encode(inputs)
            data.probs = self.autoencoder.w_autoencoder.relaxed_softmax(batched_logits)
            return self.autoencoder.decode(data, inputs)


class CounterfactualDataset(ClassifierMixin, BaseVQDataset[CounterfactualVQVAE]):
    """Dataset for counterfactual reconstruction with target conditioning."""

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: CounterfactualVQVAE,
            classifier: Model[Inputs, torch.Tensor],
            target_label: int | Literal['original'] = 'original',
            target_value: float = 1.0,
            num_classes: int = 2,
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier
        )
        self.num_classes = num_classes
        self.target_label = target_label  # don't convert to tensor yet
        self.target_value = target_value

    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs, None, self.target_label, self.target_value)
            recons = batch_ae_data.recon

            # For targets, use original labels when target_label is 'original'
            target_for_labels = batch_labels if self.target_label == 'original' else torch.tensor(self.target_label)

            for recon, original_label in zip(recons, batch_labels):
                final_label = original_label if self.target_label == 'original' else target_for_labels
                batch_data.append((
                    Inputs(cloud=recon),
                    Targets(ref_cloud=recon, label=final_label)
                ))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(
            self,
            inputs: Inputs,
            batched_logits: None | torch.Tensor = None,
            target_dim: int | Literal['original'] = 'original',
            value: float = 1.0
    ) -> Outputs:
        """Run autoencoder with counterfactual conditioning."""
        if batched_logits is None:
            batched_logits = self._run_classifier(inputs)

        with self.autoencoder.double_encoding:
            data = self.autoencoder.encode(inputs)
            probs = self.autoencoder.w_autoencoder.relaxed_softmax(batched_logits)

            # Initialize target tensor
            target = torch.zeros_like(probs)

            if target_dim == 'original':
                # Use original probabilities (no modification)
                data.probs = probs
            else:
                # Apply counterfactual conditioning
                target[:, target_dim] = 1
                data.probs = (1 - value) * probs + value * target

            return self.autoencoder.decode(data, inputs)


class BoundaryDataset(CounterfactualDataset):
    """Dataset for boundary reconstruction with neutral conditioning."""

    def __init__(
            self,
            dataset: Dataset[tuple[Inputs, Targets]],
            autoencoder: CounterfactualVQVAE,
            classifier: Model[Inputs, torch.Tensor],
            target_label: int = 0,
            num_classes: int = 2,
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier,
            target_label=target_label,
            target_value=1 / num_classes,  # Neutral probability
            num_classes=num_classes
        )


def get_dataset(partition: Partitions) -> PointCloudDataset:
    """Getter for the dataset."""
    cfg = Experiment.get_config()
    user_cfg = Experiment.get_config().user
    user_cfg.path.data_dir.mkdir(exist_ok=True)

    dataset_name = cfg.data.dataset.name
    dataset_dict: dict[Datasets, Any] = {Datasets.ModelNet: ModelNet40Dataset,
                                         Datasets.ShapenetFlow: ShapeNetDatasetFlow}
    dataset = dataset_dict[dataset_name]().split(partition)
    return dataset

def get_datasets() -> tuple[PointCloudDataset, PointCloudDataset]:
    """Get the correct datasets for training and testing."""
    cfg = Experiment.get_config()
    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    return train_dataset, test_dataset

def get_dataset_multiprocess_safe() -> tuple[PointCloudDataset, PointCloudDataset]:
    """Get the correct datasets for training and testing, but in a multiprocess safe way."""
    cfg = Experiment.get_config()
    datasets: tuple[PointCloudDataset, PointCloudDataset] | None = None
    if cfg.user.n_parallel_training_processes:
        rank = dist.get_rank()
        for i in range(cfg.user.n_parallel_training_processes):
            if rank == i:
                datasets = get_datasets()

            dist.barrier(device_ids=[rank])
    else:
        datasets = get_datasets()

    if datasets is None:
        raise RuntimeError("Datasets could not be created.")

    return datasets
