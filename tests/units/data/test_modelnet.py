"""Tests for the ModelNet builder."""

import dataclasses
import pathlib

import numpy as np
import pytest

from src.config import Experiment
from src.data.modelnet import ModelNetDataBuilder
from src.data.split import Partitions


@pytest.fixture
def builder(mocker, example_modelnet_dir: pathlib.Path) -> ModelNetDataBuilder:
    """Provide a builder reading the fake dataset."""
    return _get_builder(mocker, example_modelnet_dir)


def _get_builder(mocker, example_modelnet_dir: pathlib.Path) -> ModelNetDataBuilder:
    mocker.patch('src.data.modelnet.download_extract_zip')
    cfg = Experiment.get_config()
    path = dataclasses.replace(cfg.user.path, data_dir=example_modelnet_dir.parent)
    user = dataclasses.replace(cfg.user, path=path)
    mocker.patch.object(Experiment, 'get_config', return_value=dataclasses.replace(cfg, user=user))
    return ModelNetDataBuilder()


class TestModelNetDataBuilder:
    """Test the selection of the classes and the partitions."""

    @pytest.mark.parametrize('overrides', [['data.dataset.selected_classes=[desk,unicorn]']])
    def test_unknown_class_is_rejected(self, mocker, example_modelnet_dir: pathlib.Path) -> None:
        """A class the dataset misses stops the builder."""
        with pytest.raises(ValueError, match='unicorn'):
            _get_builder(mocker, example_modelnet_dir)

    def test_builder_is_reused(self, mocker, example_modelnet_dir: pathlib.Path) -> None:
        """The same data configuration reads the dataset once."""
        first = _get_builder(mocker, example_modelnet_dir)

        second = _get_builder(mocker, example_modelnet_dir)

        assert first is second

    def test_labels_follow_the_selection(self, builder: ModelNetDataBuilder) -> None:
        """The labels are the positions in the selected classes, the others are dropped."""
        split = builder.get_split(Partitions.test, augment=False)

        assert builder.class_names == ('desk', 'table')
        assert len(split) == 4
        assert set(split.labels.tolist()) == {0, 1}

    def test_train_and_val_join_into_train_val(self, builder: ModelNetDataBuilder) -> None:
        """The validation partition takes one cloud in six and overlaps no training cloud."""
        train = builder.get_split(Partitions.train, augment=False)
        val = builder.get_split(Partitions.val, augment=False)
        train_val = builder.get_split(Partitions.train_val, augment=False)

        assert len(train) + len(val) == len(train_val)
        assert len(val) == int(np.ceil(len(train_val) / 6))
        assert not {cloud.tobytes() for cloud in train.clouds} & {cloud.tobytes() for cloud in val.clouds}
