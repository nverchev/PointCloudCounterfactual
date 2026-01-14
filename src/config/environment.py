"""Configuration for the environment."""

import enum
import pathlib
import tomllib

from pydantic_settings import BaseSettings, SettingsConfigDict


with open(pathlib.Path(__file__).resolve().parent.parent / 'pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

VERSION = pyproject['project']['version']


class EnvSettings(BaseSettings):
    dataset_dir: pathlib.Path = pathlib.Path('./datasets')
    root_exp_dir: pathlib.Path = pathlib.Path('./experiments')
    metadata_dir: pathlib.Path = pathlib.Path('./dataset_metadata')
    model_config = SettingsConfigDict(env_file='.env')


class ConfigPath(enum.StrEnum):
    """Configuration paths relative to the project root."""

    CONFIG_ALL = 'config_all'
    TUNE_AUTOENCODER = 'tune_autoencoder'
    TUNE_W_AUTOENCODER = 'tune_w_autoencoder'

    @classmethod
    def get_folder(cls) -> str:
        """Return folder_name."""
        return 'hydra_conf'

    def get_path(self) -> pathlib.Path:
        """Return folder path."""
        return pathlib.Path(__file__).parent.parent / self.get_folder() / self

    def absolute(self) -> str:
        """Absolute path to folder"""
        return str(self.get_path().absolute().resolve())

    def relative(self) -> str:
        """Relative path to folder"""
        return f'../{self.get_folder()}/{self}'
