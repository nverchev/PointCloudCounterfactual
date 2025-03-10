from typing import Callable
import pathlib

import dry_torch
from dry_torch import Model
from dry_torch.trackers import builtin_logger, tqdm_logger
import optuna
from omegaconf import DictConfig
from optuna.visualization import plot_param_importances
import hydra

from src.autoencoder import VQVAE
from src.classifier import DGCNN
from src.config_options import ExperimentAE, MainExperiment, ConfigPath, ExperimentWAE, ExperimentClassifier
from src.config_options import get_config_all
from train_w_autoencoder import train_w_autoencoder


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

    dry_torch.remove_all_default_trackers()
    tqdm = tqdm_logger.TqdmLogger(enable_training_bar=True)
    logger = builtin_logger.BuiltinLogger()
    builtin_logger.set_verbosity(builtin_logger.INFO_LEVELS.training)
    dry_torch.extend_default_trackers([logger, tqdm])
    cfg_orig = get_config_all()
    parent_experiment = MainExperiment(cfg_orig.name, cfg_orig.user.path.exp_par_dir, cfg_orig)
    exp_ae = ExperimentAE(cfg_orig.autoencoder.name, config=cfg_orig.autoencoder)
    exp_classifier = ExperimentClassifier(cfg_orig.classifier.name, config=cfg_orig.classifier)
    parent_experiment.register_child(exp_classifier)
    parent_experiment.register_child(exp_ae)

    with exp_classifier:
        classifier_module = DGCNN()
        classifier = Model(classifier_module, name=cfg_orig.classifier.model.name, device=cfg_orig.user.device)
        classifier.load_state()

    # load saved model with original settings
    with exp_ae:
        module = VQVAE()
        autoencoder = Model(module, name=cfg_orig.autoencoder.model.name, device=cfg_orig.user.device)
        autoencoder.load_state()
        state_dict = {key: value for key, value in module.state_dict().items() if 'w_autoencoder' not in key}

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest_overrides(trial)
        cfg = get_config_all(overrides)
        cfg_wae = cfg.w_autoencoder
        exp_new_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
        parent_experiment.register_child(exp_new_ae)

        # uses overridden settings for the architecture of the w_autoencoder
        with exp_new_ae:
            new_module = VQVAE()
            new_autoencoder = Model(new_module, name=cfg.autoencoder.model.name, device=cfg.user.device)
            new_autoencoder.module.load_state_dict(state_dict, strict=False)

        exp_wae = ExperimentWAE(cfg_wae.name, config=cfg_wae)
        parent_experiment.register_child(exp_wae)

        with exp_wae:
            train_w_autoencoder(new_module, classifier, name=new_autoencoder.name, trial=trial)

        current_study = trial.study

        best_fn = min if current_study.direction.name == 'MINIMIZE' else max
        current_frozen_study = current_study.trials[-1]
        return best_fn(current_frozen_study.intermediate_values.values())

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_W_AUTOENCODER.absolute(), config_name='defaults')
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
