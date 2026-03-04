"""Base class for building point cloud datasets."""

import abc
from typing import Any, ClassVar

import numpy as np

from src.config.experiment import Experiment
from src.data.split import Partitions, PointCloudSplit
from src.utils.download import download_extract_zip
from src.data.storage import load_h5
from src.data.structures import PCD


class Singleton(type):
    """A metaclass that ensures only one instance of a class is created."""

    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class PointCloudDataBuilder(metaclass=Singleton):
    """Common base class for Point Cloud datasets like ModelNet and ShapeNet."""

    folder_name_raw: str
    folder_name_processed: str
    metadata_file: str
    url: str
    metadata: Any

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        user_cfg = cfg.user
        data_cfg = cfg.data
        selected_classes = data_cfg.dataset.selected_classes
        self.metadata_dir = user_cfg.path.metadata_dir
        self.n_input_points = data_cfg.n_input_points
        self.data_dir = user_cfg.path.data_dir
        self.dir_raw = self.data_dir / self.folder_name_raw
        self.dir_processed = self.data_dir / self.folder_name_processed
        self.metadata = self._load_metadata()
        self.available_classes = self._get_available_classes()
        self._check_selected_classes(selected_classes)
        self.classes = selected_classes if selected_classes else self.available_classes
        self.class_to_label = self._get_class_to_label()
        self._download()
        self._preprocess()
        self.split_to_pcd: dict[Partitions, PCD] = {}
        return

    def split(self, split: Partitions) -> PointCloudSplit:
        if split == Partitions.train_val:
            self._load_partition(Partitions.train)
            self._load_partition(Partitions.val)
            train_pcd = self.split_to_pcd[Partitions.train]
            val_pcd = self.split_to_pcd[Partitions.val]

            pcd = PCD(
                pcd=np.concatenate([train_pcd.pcd, val_pcd.pcd]),
                pcd_512=np.concatenate([train_pcd.pcd_512, val_pcd.pcd_512]),
                pcd_128=np.concatenate([train_pcd.pcd_128, val_pcd.pcd_128]),
                labels=np.concatenate([train_pcd.labels, val_pcd.labels]),
                std=np.concatenate([train_pcd.std, val_pcd.std]),
            )
        else:
            self._load_partition(split)
            pcd = self.split_to_pcd[split]

        return PointCloudSplit(pcd=pcd, class_names=self.classes)

    def _check_selected_classes(self, selected_classes: list[str]) -> None:
        available = set(self.available_classes)
        if any(c not in available for c in selected_classes):
            raise ValueError(f'One of classes in {selected_classes} not in the dataset')

    def _download(self) -> None:
        return download_extract_zip(target_folder=self.dir_raw, url=self.url)

    def _get_class_to_label(self) -> dict[str, int]:
        return {class_name: i for i, class_name in enumerate(sorted(self.classes))}

    def _load_partition(self, partition: Partitions) -> None:
        if partition in self.split_to_pcd:
            return

        pcds: list[PCD] = []
        for class_name, label in self.class_to_label.items():
            h5_path = self.dir_processed / class_name / f'{partition.name}.h5'
            try:
                pcd = load_h5(h5_path.parent, h5_path.name)
                pcd.labels = np.full(pcd.pcd.shape[0], label, dtype=np.int64)
                pcds.append(pcd)
            except FileNotFoundError:
                if partition == Partitions.val:
                    continue
                raise

        self.split_to_pcd[partition] = PCD(
            pcd=np.concatenate([p.pcd for p in pcds]),
            pcd_512=np.concatenate([p.pcd_512 for p in pcds]),
            pcd_128=np.concatenate([p.pcd_128 for p in pcds]),
            labels=np.concatenate([p.labels for p in pcds]),
            std=np.concatenate([p.std for p in pcds]),
        )
        return

    @abc.abstractmethod
    def _load_metadata(self) -> Any: ...

    @abc.abstractmethod
    def _get_available_classes(self) -> list[str]: ...

    @abc.abstractmethod
    def _preprocess(self) -> None: ...
