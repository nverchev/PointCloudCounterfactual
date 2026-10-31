"""Package for configuring the experiment."""

from src.config.environment import PROJECT_NAME, VERSION, ConfigPath
from src.config.experiment import Experiment
from src.config.hydra import get_experiment_config, hydra_main
from src.config.specs import (
    AutoencodingConfig,
    ClassificationConfig,
    ExperimentConfig,
    PhaseConfig,
    WAutoencodingConfig,
)
from src.config.torch import ActClass

__all__ = [
    'PROJECT_NAME',
    'VERSION',
    'ActClass',
    'AutoencodingConfig',
    'ClassificationConfig',
    'ConfigPath',
    'Experiment',
    'ExperimentConfig',
    'PhaseConfig',
    'WAutoencodingConfig',
    'get_experiment_config',
    'hydra_main',
]
