"""Module containing the ModelNet40 dataset builder."""

import logging
from typing import override

import h5py
import numpy as np

from src.data.databuilder import PointCloudDataBuilder
from src.data.storage import save_preprocessed_h5, extract_cloud_from_h5, extract_labels_from_h5
from src.data.preprocess import preprocess_point_clouds


class ModelNet40Dataset(PointCloudDataBuilder):
    """This class creates the splits for the ModelNet Dataset"""

    folder_name_raw: str = 'modelnet40_hdf5_2048'
    folder_name_processed: str = 'modelnet_h5'
    metadata_file: str = 'modelnet_classes.txt'
    url: str = 'https://gaimfs.ugent.be/Public/Dataset/modelnet40_hdf5_2048.zip'
    metadata: list[str]

    @override
    def _load_metadata(self) -> list[str]:
        with open(self.metadata_dir / self.metadata_file) as f:
            return f.read().splitlines()

    @override
    def _get_available_classes(self) -> list[str]:
        return self.metadata

    @override
    def _preprocess(self) -> None:
        splits = self._collect_all_shard_data()
        splits = {key: self._filter_selected_classes(value) for key, value in splits.items()}
        self._extract_val_split(splits)
        self._process_and_save_splits(splits)
        return

    def _collect_all_shard_data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        raw_splits = {}
        for split_name in ['train', 'test']:
            wild_str = f'*{split_name}*.h5'
            h5_files = self.dir_raw.glob(wild_str)
            all_pcs = []
            all_labels = []
            for h5_file in h5_files:
                with h5py.File(h5_file, 'r') as f:
                    pcs: np.ndarray = extract_cloud_from_h5(f, 'data')
                    labels: np.ndarray = extract_labels_from_h5(f, 'label')
                    all_pcs.append(pcs)
                    all_labels.append(labels)

            pcs = np.concatenate(all_pcs, axis=0)
            labels = np.concatenate(all_labels, axis=0).ravel()
            raw_splits[split_name] = (pcs, labels)

        return raw_splits

    def _filter_selected_classes(self, data: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        class_to_orig_idx = {name: i for i, name in enumerate(self.available_classes)}
        selected_orig_labels = [class_to_orig_idx[name] for name in self.classes]
        pcs, labels = data
        mask = np.isin(labels, selected_orig_labels)
        return pcs[mask], labels[mask]

    def _extract_val_split(self, splits: dict[str, tuple[np.ndarray, np.ndarray]], val_every: int = 6) -> None:
        pcs_filt_train, labels_filt_train = splits['train']
        train_indices_range = list(range(len(pcs_filt_train)))
        val_idx = [train_indices_range.pop(i) for i in train_indices_range[::-val_every]]
        train_idx = train_indices_range
        splits['train'] = (pcs_filt_train[train_idx], labels_filt_train[train_idx])
        splits['val'] = (pcs_filt_train[val_idx], labels_filt_train[val_idx])
        return

    def _process_and_save_splits(self, final_splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        class_to_orig_idx = {name: i for i, name in enumerate(self.available_classes)}
        for class_name in self.classes:
            if (self.dir_processed / class_name).exists():
                continue

            logging.info(f'Processing point clouds for class {class_name}')
            orig_idx = class_to_orig_idx[class_name]
            target_label = self.class_to_label[class_name]
            for split_name, (pcs_full, labels_full) in final_splits.items():
                indices = np.where(labels_full == orig_idx)[0]
                h5_path = self.dir_processed / class_name / f'{split_name}.h5'
                cat_pcs = pcs_full[indices].astype(np.float32)
                cat_labels = np.full(len(indices), target_label, dtype=np.int64)
                pcd = preprocess_point_clouds(list(cat_pcs), labels=cat_labels, desc=f'{class_name} {split_name}')
                save_preprocessed_h5(h5_path, pcd)

        return
