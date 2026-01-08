"""Tune the hyperparameters of the w-autoencoder."""

import pathlib
from typing import Callable

import drytorch.core.exceptions
import hydra
import optuna
import yaml
from drytorch import Model, init_trackers
from drytorch.contrib.optuna import suggest_overrides, get_final_value
from drytorch.core.register import unregister_model, register_model
from omegaconf import DictConfig
from optuna.visualization import plot_param_importances

from src.autoencoder import CounterfactualVQVAE
from src.classifier import DGCNN
from src.config_options import Experiment, ConfigPath
from src.config_options import get_config_all
from train_w_autoencoder import train_w_autoencoder


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration."""
    main_cfg = get_config_all()
    main_exp = Experiment(main_cfg, name=main_cfg.name, par_dir=main_cfg.user.path.exp_par_dir, tags=main_cfg.tags)

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
        trial_exp = Experiment(trial_cfg, name='Trial', par_dir=trial_cfg.user.path.exp_par_dir, tags=overrides)

        # uses overridden settings for the architecture of the w_autoencoder
        with (trial_exp.create_run(record=False)):
            new_vqvae_module = CounterfactualVQVAE()
            # best to define a new model here otherwise the weights of the w_autoencoder will be modified
            new_autoencoder = Model(new_vqvae_module,
                                    name=trial_cfg.autoencoder.architecture.name,
                                    device=trial_cfg.user.device)
            new_autoencoder.module.load_state_dict(state_dict, strict=False)
            register_model(classifier)  # register to current experiment
            try:
                train_w_autoencoder(new_vqvae_module, classifier, name=new_autoencoder.name, trial=trial)
            except optuna.TrialPruned:
                import sklearn
                import numpy as np
                study = trial.study
                *past_trials, current_trial = study.get_trials(deepcopy=False)
                current_step = current_trial.last_step
                if current_step is None:
                    return float('nan')

                real_completed_trials = [
                    t for t in past_trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and current_step in t.intermediate_values
                    and t.value is not None
                    and not t.user_attrs.get('imputed', False)
                ]
                if len(real_completed_trials) < 10:
                    raise optuna.TrialPruned()

                past_intermediate_values: list[float] = [
                    t.intermediate_values[current_step] for t in real_completed_trials
                ]
                quadratic_predictors = np.array([[value, value ** 2] for value in past_intermediate_values])
                past_final_values: list[float] = [
                    trial.value for trial in real_completed_trials if trial.value is not None
                ]
                responses = np.array(past_final_values)
                model = sklearn.linear_model.LinearRegression()
                model.fit(quadratic_predictors, responses)
                pruned_intermediate_value = current_trial.intermediate_values[current_step]
                quadratic_trial_data = np.array([[pruned_intermediate_value, pruned_intermediate_value ** 2]])
                imputed_value = model.predict(quadratic_trial_data)[0]
                median_final_value = past_final_values[len(past_final_values) // 2]
                worst_fn = min if study.direction == optuna.study.StudyDirection.MAXIMIZE else max
                imputed_value = worst_fn(imputed_value, median_final_value)
                trial.set_user_attr('imputed', True)
                return imputed_value

            except drytorch.core.exceptions.ConvergenceError:
                trial.set_user_attr('imputed', True)  # treating the trial as complete
                study = trial.study
                *past_trials, current_trial = study.get_trials(deepcopy=False)

                real_completed_trials = [
                    t for t in past_trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                    and not t.user_attrs.get('imputed', False)
                ]
                if len(real_completed_trials) < 10:
                    raise optuna.TrialPruned()

                past_final_values = [
                    trial.value for trial in real_completed_trials if trial.value is not None
                ]
                worst_fn = min if study.direction == optuna.study.StudyDirection.MAXIMIZE else max

                return worst_fn(past_final_values)
            finally:
                unregister_model(classifier)  # classifier can be safely reused for other experiments

        return get_final_value(trial)

    return _objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_W_AUTOENCODER.absolute(), config_name='defaults')
def tune(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    init_trackers(mode='minimal')
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=tune_cfg.tune.n_startup_trials,
                                         n_warmup_steps=tune_cfg.tune.n_warmup_steps,
                                         interval_steps=tune_cfg.tune.interval_steps,
                                         n_min_trials=tune_cfg.tune.n_min_trials)
    sampler = optuna.samplers.GPSampler(warn_independent_sampling=False)
    with (ConfigPath.CONFIG_ALL.get_path() / 'defaults').with_suffix('.yaml').open() as f:
        version = f'v{yaml.safe_load(f)['version']}'

    study = optuna.create_study(study_name=tune_cfg.tune.study_name + '_'.join(['', version] + tune_cfg.overrides[1:]),
                                storage=tune_cfg.storage,
                                sampler=sampler,
                                pruner=pruner,
                                load_if_exists=True)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials)
    plot_param_importances(study).show(renderer=tune_cfg.renderer)
    return


if __name__ == '__main__':
    tune()
