"""Contains the Experiment class for the project and related helper functions."""

import sys

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core.track import Tracker

from src.config.specs import ConfigAll


class Experiment(drytorch.Experiment[ConfigAll]):
    """Contains the specifications for the current experiment."""

    pass


def get_trackers(cfg: ConfigAll) -> list[Tracker]:
    """Get trackers from according to the user configuration."""
    cfg_trackers = cfg.user.trackers
    cfg_hydra = cfg.user.hydra
    hydra_utils.configure_log(cfg_hydra.job_logging)
    drytorch.init_trackers(mode='hydra')
    tracker_list: list[Tracker] = []
    if sys.gettrace():  # skip in debug mode
        return tracker_list

    if cfg_trackers.wandb:
        import wandb

        from drytorch.trackers.wandb import Wandb

        tracker_list.append(Wandb(settings=wandb.Settings(project=cfg.project)))

    if cfg_trackers.hydra:
        from drytorch.trackers.hydra import HydraLink

        tracker_list.append(HydraLink(hydra_dir=cfg.user.hydra.output_dir))

    if cfg_trackers.csv:
        from drytorch.trackers.csv import CSVDumper

        tracker_list.append(CSVDumper())

    if cfg_trackers.tensorboard:
        from drytorch.trackers.tensorboard import TensorBoard

        tracker_list.append(TensorBoard())

    if cfg_trackers.sqlalchemy:
        import sqlalchemy

        from drytorch.trackers.sqlalchemy import SQLConnection

        engine_path = cfg.user.path.version_dir / 'metrics.db'
        cfg.user.path.version_dir.mkdir(exist_ok=True)
        engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
        tracker_list.append(SQLConnection(engine=engine))

    return tracker_list


def update_exp_name(cfg: ConfigAll, overrides: list[str]) -> None:
    """Adds the overrides to the name for the experiment."""
    overrides = [
        override
        for override in overrides
        if override.split('.')[0] != 'user' and override.split('=')[0] not in ('final', 'variation')
    ]
    cfg.variation = '_'.join([cfg.variation, *overrides]).replace('/', '_')
    cfg.tags = overrides
    return
