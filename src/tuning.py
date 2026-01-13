"""Utilities for hyperparameter tuning."""

import numpy as np
import optuna


def get_past_final_values(trial: optuna.Trial) -> list[float]:
    """Get final value from completed trials."""
    study = trial.study
    *past_trials, _current_trial = study.get_trials(deepcopy=False)
    real_completed_trials = [
        t
        for t in past_trials
        if (
            t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and not t.user_attrs.get('imputed', False)
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
