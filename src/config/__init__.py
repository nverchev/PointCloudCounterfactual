"""Package for configuring for the experiment."""

from src.config.specs import ConfigAll
from src.config.hydra import hydra_main, get_config_all
from src.config.experiment import Experiment, get_trackers, set_tuning_logging
from src.config.environment import VERSION, ConfigPath
from src.config.torch import ActClass

__all__ = [
    'VERSION',
    'ActClass',
    'ConfigAll',
    'ConfigPath',
    'Experiment',
    'get_config_all',
    'get_trackers',
    'hydra_main',
    'set_tuning_logging',
]
