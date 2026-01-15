"""Tune the hyperparameters of the autoencoder"""

import pathlib

from collections.abc import Callable

import hydra
import optuna
import torch

from omegaconf import DictConfig
from optuna.visualization import plot_param_importances

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError

from src.config import ConfigPath, Experiment, get_config_all, set_tuning_logging
from src.utils.tuning import impute_failed_trial, impute_pruned_trial, get_study_name

from train_autoencoder import train_autoencoder


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration."""

    def objective(trial: optuna.Trial) -> float:
        """Set up the experiment and launches the training cycle."""
        overrides = suggest_overrides(tune_cfg, trial)
        cfg = get_config_all(overrides)
        exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
        with exp.create_run(record=False):
            try:
                train_autoencoder(trial=trial)
            except optuna.TrialPruned:
                return impute_pruned_trial(trial)

            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_AUTOENCODER.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    set_tuning_logging()
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=tune_cfg.tune.n_startup_trials,
        n_warmup_steps=tune_cfg.tune.n_warmup_steps,
        interval_steps=tune_cfg.tune.interval_steps,
        n_min_trials=tune_cfg.tune.n_min_trials,
    )
    sampler = optuna.samplers.GPSampler(warn_independent_sampling=False)
    study_name = get_study_name(tune_cfg.tune.study_name, tune_cfg.overrides[1:])
    study = optuna.create_study(
        study_name=study_name, storage=tune_cfg.storage, sampler=sampler, pruner=pruner, load_if_exists=True
    )
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    plot_param_importances(study).show(renderer=tune_cfg.renderer)
    return


if __name__ == '__main__':
    main()
