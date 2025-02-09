from typing import Callable, cast
import pathlib

import dry_torch
import optuna
from optuna.visualization import plot_param_importances
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.config_options import ExperimentAE, ConfigTrainAE, ParentExperiment
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
        with hydra.initialize(version_base=None, config_path="hydra_conf/autoencoder_conf/"):
            dict_cfg = hydra.compose(config_name="defaults", overrides=overrides)
            cfg = cast(ConfigTrainAE, OmegaConf.to_object(dict_cfg))
            dry_torch.remove_all_default_trackers()
            exp = ExperimentAE(cfg.exp.name, config=cfg)
            ParentExperiment(cfg.exp.main_name, par_dir=cfg.user.path.exp_par_dir).register_child(exp)
            with exp:
                train_autoencoder()
            train_autoencoder(trial)

            current_study = trial.study
            best_fn = min if current_study.direction.name == 'MINIMIZE' else max
            current_frozen_study = current_study.trials[-1]
            return best_fn(current_frozen_study.intermediate_values.values())

    return objective


@hydra.main(version_base=None, config_path="hydra_conf/autoencoder_tuning_conf/", config_name="defaults")
def tune(tune_cfg: DictConfig):
    GlobalHydra.instance().clear()
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    study = optuna.create_study(study_name=tune_cfg.tune.study_name, storage=tune_cfg.storage, load_if_exists=True)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    plot_param_importances(study).show(renderer=tune_cfg.renderer)


if __name__ == '__main__':
    tune()
