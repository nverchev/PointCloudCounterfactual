"""Configuration module with shared test fixtures."""

import pathlib

from collections.abc import Generator

import h5py
import numpy as np
import pytest
import torch

from src.config import Experiment, ExperimentConfig, get_experiment_config

SMALL_MODELS = [
    'autoencoding.model.architecture.decoder.conv_h_dims=[16,8]',
    'autoencoding.model.architecture.decoder.map_h_dims=[8]',
    'autoencoding.model.architecture.decoder.n_components=2',
    'autoencoding.model.architecture.encoder.h_dims=[8,8]',
    'autoencoding.model.architecture.encoder.n_neighbours=4',
    'autoencoding.model.embedding_dim=2',
    'autoencoding.model.w_dim=16',
    'classification.model.architecture.embedding_dim=8',
    'classification.model.architecture.hidden_dims=[8,8]',
    'classification.model.architecture.mlp_dims=[8,8]',
    'classification.model.architecture.n_neighbours=4',
    'data.n_points=32',
    'user.device=cpu',
    'w_autoencoding.model.architecture.decoder_h_dims=[4]',
    'w_autoencoding.model.architecture.encoder_h_dims=[2]',
    'w_autoencoding.model.z_u_dim=2',
    'w_autoencoding.model.z_y_dim=2',
]


@pytest.fixture(scope='package')
def example_config() -> ExperimentConfig:
    """Provide the ModelNet-DT configuration with small models on the CPU."""
    return get_experiment_config(overrides=SMALL_MODELS)


@pytest.fixture
def overrides() -> list[str]:
    """Provide the overrides of the configuration; a test parametrizes it to change settings."""
    return []


@pytest.fixture(autouse=True)
def active_experiment(
    overrides: list[str], example_config: ExperimentConfig, tmp_path: pathlib.Path
) -> Generator[Experiment]:
    """Provide an active run of the configuration with the overrides, for every test."""
    config = get_experiment_config(overrides=[*SMALL_MODELS, *overrides]) if overrides else example_config
    exp = Experiment(config, name='test_exp', par_dir=tmp_path)
    with exp.create_run(record=False):
        yield exp

    return


@pytest.fixture
def example_clouds() -> torch.Tensor:
    """Provide a batch of clouds of the configured size."""
    return torch.randn(4, 32, 3)


@pytest.fixture
def example_modelnet_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provide a folder with the files of a small fake ModelNet: three classes, label by index."""
    folder = tmp_path / 'modelnet40_hdf5_2048'
    folder.mkdir()
    (folder / 'shape_names.txt').write_text('chair\ndesk\ntable\n')
    for name, n_clouds in (('train0', 12), ('train1', 12), ('test0', 6)):
        with h5py.File(folder / f'{name}.h5', 'w') as file:
            file.create_dataset('data', data=np.random.rand(n_clouds, 32, 3).astype(np.float32))
            file.create_dataset('label', data=(np.arange(n_clouds) % 3).reshape(-1, 1))

    return folder
