"""Tune the hyperparameters of the autoencoder"""

from typing import Callable
import pathlib

import optuna
from omegaconf import DictConfig
from optuna.visualization import plot_param_importances
import hydra
import yaml

from drytorch import init_trackers
from drytorch.contrib.optuna import suggest_overrides, get_final_value

from src.config_options import Experiment, ConfigPath
from src.config_options import get_config_all
from train_autoencoder import train_autoencoder


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration."""

    def objective(trial: optuna.Trial) -> float:
        """Set up the experiment and launches the training cycle."""
        overrides = suggest_overrides(tune_cfg, trial)
        cfg = get_config_all(overrides)
        exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
        with exp.create_run(register=False):
            train_autoencoder(trial)
        return get_final_value(trial)

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_AUTOENCODER.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    init_trackers(mode='tuning')
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=tune_cfg.tune.n_startup_trials,
                                         n_warmup_steps=tune_cfg.tune.n_warmup_steps)
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    with (ConfigPath.CONFIG_ALL.get_path() / 'defaults').with_suffix('.yaml').open() as f:
        version = f"v{yaml.safe_load(f)['version']}"

    study = optuna.create_study(study_name=tune_cfg.tune.study_name + '_'.join(['', version] + tune_cfg.overrides[1:]),
                                storage=tune_cfg.storage,
                                sampler=sampler,
                                pruner=pruner,
                                load_if_exists=True)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    plot_param_importances(study).show(renderer=tune_cfg.renderer)


if __name__ == '__main__':
    main()
