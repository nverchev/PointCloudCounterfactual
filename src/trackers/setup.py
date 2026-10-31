"""Setup of the trackers of the experiment."""

from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core import tracking
from drytorch.trackers import logging

from src.config.environment import DEBUG_MODE
from src.config.specs import ExperimentConfig
from src.trackers.logging import ClassificationLogger

if TYPE_CHECKING:
    from drytorch.trackers.sqlalchemy import SQLConnection

__all__ = ['get_metric_database', 'get_trackers']


def get_metric_database(cfg: ExperimentConfig) -> SQLConnection | None:
    """Connect to the metric database of the configuration.

    Args:
        cfg: Root configuration object.

    Returns:
        The connection, or ``None`` if the tracker is off or SQLAlchemy is not installed.

    Side Effects:
        Creates the database in ``par_dir`` if it does not exist.
    """
    if not cfg.user.trackers.sqlalchemy:
        return None

    try:
        import sqlalchemy

        from drytorch.trackers.sqlalchemy import SQLConnection

    except ImportError as err:
        warnings.warn(f'SQLAlchemy not installed, skipping. {err}', stacklevel=2)
        return None

    engine_path = cfg.par_dir / 'metrics.db'
    cfg.par_dir.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
    return SQLConnection(engine=engine)


def get_trackers(cfg: ExperimentConfig, debug_mode: bool = DEBUG_MODE) -> list[tracking.Tracker]:
    """Instantiate the experiment trackers requested in *cfg*.

    Trackers are only activated outside debug mode.  In debug mode an empty
    list is returned so that no files or remote metrics are written.

    Args:
        cfg: Root configuration object.
        debug_mode: When ``True``, skip all tracker initialisation. Defaults to
            whether a debugger runs the process.

    Returns:
        A list of initialised :class:`drytorch.core.tracking.Tracker` objects
        ready to be passed to :meth:`drytorch.Experiment.trackers.subscribe`.
    """
    cfg_trackers = cfg.user.trackers
    hydra_utils.configure_log(cfg.user.hydra.job_logging)
    drytorch.init_trackers(mode='hydra')
    tracking.DEFAULT_TRACKERS.pop(logging.BuiltinLogger.__name__, None)
    drytorch.extend_default_trackers([ClassificationLogger()])
    tracker_list: list[tracking.Tracker] = []
    if debug_mode:
        return tracker_list

    if cfg_trackers.hydra:
        from drytorch.trackers.hydra import HydraLink

        tracker_list.append(HydraLink(hydra_dir=cfg.user.hydra.output_dir))

    metric_loader = get_metric_database(cfg)
    if metric_loader is not None:
        tracker_list.append(metric_loader)

    if cfg_trackers.plotly:
        try:
            from drytorch.trackers.plotly import PlotlyPlotter

        except ImportError as err:
            warnings.warn(f'Plotly not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(PlotlyPlotter(metric_loader=metric_loader, show=cfg.user.plot.show))

    return tracker_list
