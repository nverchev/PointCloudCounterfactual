from typing import Callable
import pathlib

import dry_torch
from dry_torch.trackers import builtin_logger, tqdm_logger
import optuna
from omegaconf import DictConfig
from optuna.visualization import plot_param_importances
import hydra

from src.config_options import ExperimentAE, MainExperiment, ConfigPath
from src.config_options import get_config_all
from train_autoencoder import train_autoencoder


def set_objective(tune_cfg) -> Callable[[optuna.Trial], float]:
    def suggest_overrides(trial: optuna.Trial) -> list[str]:
        overrides: list[str] = [*tune_cfg.overrides]
        for attr_name, param in tune_cfg.tune.params.items():
            if param.suggest == 'suggest_list':
                new_value = []
                for i in range(trial.suggest_int(name='_'.join([attr_name, 'len']),
                                                 low=param.settings.min_length,
                                                 high=param.settings.max_length)):
                    try:
                        suggest = getattr(trial, param.settings.suggest)
                    except AttributeError as ae:
                        raise ValueError(f'Invalid search configuration: {ae}')
                    new_value.append(suggest('_'.join([attr_name, str(i)]), **param.settings.settings))
            else:
                try:
                    suggest = getattr(trial, param.suggest)
                except AttributeError as ae:
                    raise ValueError(f'Invalid search configuration: {ae}')
                new_value = suggest(attr_name, **param.settings)
            overrides.append(f'{attr_name}={new_value}')

        return overrides

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_overrides(trial)
        cfg = get_config_all(overrides)
        dry_torch.remove_all_default_trackers()
        logger = builtin_logger.BuiltinLogger()
        # do not enable training bar if redirected stdout is not able to handle multiple tqdm bars
        tqdm = tqdm_logger.TqdmLogger(enable_training_bar=True)
        builtin_logger.set_verbosity(builtin_logger.INFO_LEVELS.training)
        dry_torch.extend_default_trackers([logger, tqdm])
        exp = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
        parent = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
        parent.register_child(exp)
        with exp:
            train_autoencoder(trial)

        current_study = trial.study
        best_fn = min if current_study.direction.name == 'MINIMIZE' else max
        current_frozen_study = current_study.trials[-1]
        return best_fn(current_frozen_study.intermediate_values.values())

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_AUTOENCODER.absolute(), config_name='defaults')
def tune(tune_cfg: DictConfig):
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=tune_cfg.tune.n_startup_trials,
                                         n_warmup_steps=tune_cfg.tune.n_warmup_steps)
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    study = optuna.create_study(study_name=tune_cfg.tune.study_name,
                                storage=tune_cfg.storage,
                                sampler=sampler,
                                pruner=pruner,
                                load_if_exists=True)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    plot_param_importances(study).show(renderer=tune_cfg.renderer)


if __name__ == '__main__':
    tune()
