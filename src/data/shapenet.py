"""Module containing the ShapeNet dataset builder."""

import json
import logging
import pathlib
from typing import override

import numpy as np

from src.data.databuilder import PointCloudDataBuilder
from src.data.storage import save_preprocessed_h5
from src.data.preprocess import preprocess_point_clouds


class ShapeNetFlowDataset(PointCloudDataBuilder):
    """This class creates the splits for the ShapeNet Dataset."""

    folder_name_raw: str = 'ShapeNetCore.v2.PC15k'
    folder_name_processed: str = 'shapenet_preprocessed'
    metadata_file: str = 'shapenet_PointFlow_classes.json'
    url: str = 'https://gaimfs.ugent.be/Public/Dataset/ShapeNetCore.v2.PC15k.zip'
    metadata: dict[str, str]

    @override
    def _load_metadata(self) -> dict[str, str]:
        with open(self.metadata_dir / self.metadata_file) as f:
            return json.load(f)

    @override
    def _get_available_classes(self) -> list[str]:
        return list(self.metadata.values())

    @override
    def _preprocess(self) -> None:
        for folder in self.dir_raw.iterdir():
            if folder.is_dir():
                class_name = self.metadata.get(folder.name)
                if class_name and class_name in self.classes:
                    label = self.class_to_label[class_name]
                    if not (self.dir_processed / class_name).exists():
                        logging.info(f'Processing point clouds for class {class_name}')
                        self._pre_process_class(folder, class_name, label)

        return

    def _pre_process_class(self, raw_dir: pathlib.Path, class_name: str, label: int) -> None:
        for split_name in ['train', 'val', 'test']:
            h5_path = self.dir_processed / class_name / f'{split_name}.h5'
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            split_dir = raw_dir / split_name
            files = sorted(split_dir.glob('*.npy'))
            pcd_list = []
            for path in files:
                pc = np.load(path)
                pcd_list.append(pc.astype(np.float32))

            cat_labels = np.full(len(pcd_list), label, dtype=np.int64)
            pcd = preprocess_point_clouds(pcd_list, labels=cat_labels, desc=f'{class_name} {split_name}')
            save_preprocessed_h5(h5_path, pcd)

        return
