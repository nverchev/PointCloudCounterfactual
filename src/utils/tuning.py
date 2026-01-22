"""Utilities for hyperparameter tuning."""

import numpy as np
import optuna
import yaml
from optuna import visualization

from src.config import VERSION, ConfigPath


def get_past_final_values(trial: optuna.Trial) -> list[float]:
    """Get final value from completed trials."""
    study = trial.study
    *past_trials, _current_trial = study.get_trials(deepcopy=False)
    real_completed_trials = [
        past_trial
        for past_trial in past_trials
        if (
            past_trial.state == optuna.trial.TrialState.COMPLETE
            and past_trial.value is not None
            and not past_trial.user_attrs.get('imputed', False)
        )
    ]
    if len(real_completed_trials) < 10:
        raise optuna.TrialPruned()

    return [trial.value for trial in real_completed_trials if trial.value is not None]


def impute_pruned_trial(trial: optuna.Trial) -> float:
    """Input pruned trial with an interpolation from past completed trials' values."""
    past_final_values = get_past_final_values(trial)
    percentile = 75 if trial.study.direction == optuna.study.StudyDirection.MINIMIZE else 25
    imputed_value = np.percentile(past_final_values, percentile)
    trial.set_user_attr('imputed', True)
    return float(imputed_value)


def impute_failed_trial(trial: optuna.Trial) -> float:
    """Input pruned trial with worse completed trials' value."""
    past_final_values = get_past_final_values(trial)
    worst_fn = min if trial.study.direction == optuna.study.StudyDirection.MAXIMIZE else max
    trial.set_user_attr('imputed', True)
    return worst_fn(past_final_values)


def visualize_study(study: optuna.Study, renderer: str) -> None:
    """Visualize the optimization study."""
    visualization.plot_optimization_history(study).show(renderer=renderer)
    visualization.plot_slice(study).show(renderer=renderer)
    visualization.plot_contour(study).show(renderer=renderer)
    visualization.plot_parallel_coordinate(study).show(renderer=renderer)
    visualization.plot_param_importances(study).show(renderer=renderer)
    return


def get_study_name(tuning_scheme: str, overrides: list[str]) -> str:
    """Get the study name from the configuration."""
    version = f'v{VERSION}'
    with (ConfigPath.CONFIGS.get_path() / 'defaults').with_suffix('.yaml').open() as f:
        loaded_cfg = yaml.safe_load(f)
        variation = loaded_cfg['variation']

    overrides_repr = (override.rsplit('.', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1] for override in overrides)
    return '_'.join([version, variation, *overrides_repr, tuning_scheme])
