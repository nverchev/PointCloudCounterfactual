"""Tune the hyperparameters of the w-autoencoder."""

import pathlib

from collections.abc import Callable

import hydra
import optuna
import torch

from omegaconf import DictConfig
from optuna.visualization import plot_param_importances

from drytorch import Model, init_trackers
from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError
from drytorch.core.register import register_model, unregister_model

from src.module import CounterfactualVQVAE, DGCNN
from src.config import ConfigPath, Experiment, get_config_all
from src.utils.tuning import impute_failed_trial, impute_pruned_trial, get_study_name

from train_w_autoencoder import train_w_autoencoder


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration."""
    main_cfg = get_config_all()
    main_exp = Experiment(main_cfg, name=main_cfg.name, par_dir=main_cfg.user.path.version_dir, tags=main_cfg.tags)

    with main_exp.create_run(resume=True):
        classifier_module = DGCNN()
        classifier = Model(classifier_module, name=main_cfg.classifier.architecture.name, device=main_cfg.user.device)
        classifier.load_state()
        unregister_model(classifier)  # unregister from the previous experiment (classifier is not modified)

        # load saved model with original settings
        vqvae_module = CounterfactualVQVAE()
        autoencoder = Model(vqvae_module, name=main_cfg.autoencoder.architecture.name, device=main_cfg.user.device)
        autoencoder.load_state()
        state_dict = {key: value for key, value in vqvae_module.state_dict().items() if 'w_autoencoder' not in key}

    def _objective(trial: optuna.Trial) -> float:
        overrides = suggest_overrides(tune_cfg, trial)
        trial_cfg = get_config_all(overrides)
        trial_exp = Experiment(trial_cfg, name='Trial', par_dir=trial_cfg.user.path.version_dir, tags=overrides)

        # uses overridden settings for the architecture of the w_autoencoder
        with trial_exp.create_run(record=False):
            new_vqvae_module = CounterfactualVQVAE()
            # best to define a new model here otherwise the weights of the w_autoencoder will be modified
            new_autoencoder = Model(
                new_vqvae_module, name=trial_cfg.autoencoder.architecture.name, device=trial_cfg.user.device
            )
            new_autoencoder.module.load_state_dict(state_dict, strict=False)
            register_model(classifier)  # register to current experiment
            try:
                train_w_autoencoder(new_vqvae_module, classifier, name=new_autoencoder.name, trial=trial)
            except optuna.TrialPruned:
                return impute_pruned_trial(trial)

            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                unregister_model(classifier)  # classifier can be safely reused for other experiments
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return _objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_W_AUTOENCODER.absolute(), config_name='defaults')
def tune(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    init_trackers(mode='minimal')
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=tune_cfg.tune.n_startup_trials,
        n_warmup_steps=tune_cfg.tune.n_warmup_steps,
        interval_steps=tune_cfg.tune.interval_steps,
        n_min_trials=tune_cfg.tune.n_min_trials,
    )
    sampler = optuna.samplers.GPSampler(warn_independent_sampling=False)
    study_name = get_study_name(tune_cfg.study_name, tune_cfg.overrides[1:])
    study = optuna.create_study(
        study_name=study_name, storage=tune_cfg.storage, sampler=sampler, pruner=pruner, load_if_exists=True
    )
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    plot_param_importances(study).show(renderer=tune_cfg.renderer)
    return


if __name__ == '__main__':
    tune()
