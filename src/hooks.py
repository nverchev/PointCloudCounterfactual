import warnings

import numpy as np
import torch
from dry_torch import Diagnostic
from dry_torch.hooks import static_hook_class

from src.config_options import ExperimentAE
from src.autoencoder import VQVAE

from collections.abc import Sequence
from typing import Optional, Callable, Literal
import dry_torch.protocols as p
from dry_torch.hooks import MetricMonitor

import optuna


class DiscreteSpaceOptimizer():

    def __init__(self, diagnostic: Diagnostic) -> None:
        self.diagnostic = diagnostic
        self.cfg_ae = ExperimentAE.get_config().autoencoder

    def __call__(self) -> None:
        if not isinstance(self.diagnostic.model.module, VQVAE):
            warnings.warn('Model not supported.')
            return
        module = self.diagnostic.model.module
        self.diagnostic(store_outputs=True)
        idx = torch.vstack([output.one_hot_idx for output in self.diagnostic.outputs_list]).sum(0)
        unused_idx = torch.eq(idx, 0)
        for i in range(self.cfg_ae.w_dim // self.cfg_ae.embedding_dim):
            p = np.array(idx[i])
            p = p / p.sum()
            for j in range(self.cfg_ae.book_size):
                if unused_idx[i, j]:
                    k = np.random.choice(np.arange(self.cfg_ae.book_size), p=p)
                    used_embedding = module.codebook.data[i, k]
                    noise = self.cfg_ae.vq_noise * torch.randn_like(used_embedding)
                    module.codebook.data[i, j] = used_embedding + noise
        return


class TrialCallback:
    """
    Implements pruning logic for training models.

    Attributes:
        monitor: Monitor instance
        trial: Optuna trial.
    """

    def __init__(
            self,
            trial: optuna.Trial,
            metric: Optional[str | p.MetricCalculatorProtocol] = None,
            monitor: Optional[p.EvaluationProtocol] = None,
            min_delta: float = 1e-8,
            best_is: Literal['auto', 'higher', 'lower'] = 'auto',
            aggregate_fn: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> None:
        """
        Args:
            trial: Optuna trial
            metric: Name of metric to monitor or metric calculator instance.
                            Defaults to first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            best_is: Whether higher or lower metric values are better.
               'auto' will determine this from first measurements.
            aggregate_fn: Function to aggregate recent metric values.
                Defaults to min/max based on best_is.
        """
        window: int = 10
        if aggregate_fn is None:
            def _aggregate_fn(float_list: Sequence[float], /) -> float:
                return sum(float_list[-window:]) / window
        else:
            _aggregate_fn = aggregate_fn
        self.monitor = MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            patience=0,
            best_is=best_is,
            aggregate_fn=_aggregate_fn,
        )
        self.trial = trial
        return

    def __call__(self, instance: p.TrainerProtocol) -> None:
        """
        Evaluate whether training should be stopped early.

        Args:
            instance: Trainer instance to evaluate.
        """
        epoch = instance.model.epoch

        self.monitor.register_metric(instance)
        self.trial.report(self.monitor.current_result, epoch)
        if self.trial.should_prune():
            metric_name = self.monitor.metric_name
            instance.terminate_training(
                f'Optuna pruning while monitoring {metric_name}.'
            )
            raise optuna.TrialPruned()

        return
