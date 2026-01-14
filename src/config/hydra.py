"""Hydra configuration."""

import functools
import pathlib
from typing import cast
from collections.abc import Callable

import hydra
from hydra.core import hydra_config, config_store
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

from src.config.environment import ConfigPath
from src.config.specs import ConfigAll
from src.config.experiment import update_exp_name


def get_config_all(overrides: list[str] | None = None) -> ConfigAll:
    """Get hydra configuration without starting a run."""
    GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path=ConfigPath.CONFIG_ALL.relative()):
        dict_cfg = hydra.compose(config_name='defaults', overrides=overrides)
        cfg = cast(ConfigAll, OmegaConf.to_object(dict_cfg))

        if overrides is not None:
            update_exp_name(cfg, overrides)
        return cfg


def get_current_hydra_dir() -> pathlib.Path:
    """Get the path to the current hydra run."""
    config = hydra_config.HydraConfig.get()
    return pathlib.Path(config.runtime.output_dir)


def hydra_main(func: Callable[[ConfigAll], None]) -> Callable[[], None]:
    """Start hydra run and Converts dict_cfg to ConfigAll."""

    @hydra.main(version_base=None, config_path=str(ConfigPath.CONFIG_ALL.absolute()), config_name='defaults')
    @functools.wraps(func)
    def wrapper(dict_cfg: DictConfig) -> None:
        """Convert configuration to the stored object"""
        cfg = cast(ConfigAll, OmegaConf.to_object(dict_cfg))
        overrides = HydraConfig.get().overrides.task
        update_exp_name(cfg, overrides)
        return func(cfg)

    return wrapper.__call__


cs = config_store.ConfigStore.instance()  # type: ignore
cs.store(name='config_all', node=ConfigAll)
