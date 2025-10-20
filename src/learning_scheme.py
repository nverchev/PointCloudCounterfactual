"""Module for learning scheme configuration."""

from drytorch.lib import schedulers, gradient_ops
from drytorch import LearningScheme
from drytorch.core import protocols as p
from src.config_options import Schedulers, SchedulerConfig, Experiment, LearningConfig, GradOp


def get_scheduler(config: SchedulerConfig) -> schedulers.AbstractScheduler:
    """Returns the scheduler instance based on config."""
    if config.function == Schedulers.Constant:
        scheduler: schedulers.AbstractScheduler = schedulers.ConstantScheduler()
    elif config.function == Schedulers.Cosine:
        scheduler = schedulers.CosineScheduler(**config.settings)
    elif config.function == Schedulers.Exponential:
        scheduler = schedulers.ExponentialScheduler(**config.settings)

    scheduler = scheduler.bind(schedulers.restart(
        restart_interval=config.restart_interval, restart_fraction=config.restart_fraction)
    ).bind(schedulers.warmup(config.warmup_steps))

    return scheduler


def get_grad_op(config: LearningConfig) -> None | p.GradientOpProtocol:
    """Returns the gradient clipping instance based on config."""
    if config.gradient_op == GradOp.GradNormalizer:
        return gradient_ops.GradNormalizer()
    elif config.gradient_op == GradOp.GradZScoreNormalizer:
        return gradient_ops.GradZScoreNormalizer()
    elif config.gradient_op == GradOp.GradValueClipper:
        return gradient_ops.GradValueClipper()
    elif config.gradient_op == GradOp.GradNormClipper:
        return gradient_ops.GradNormClipper()
    elif config.gradient_op == GradOp.HistClipping:
        return gradient_ops.HistClipping()
    elif config.gradient_op == GradOp.ParamHistClipping:
        return gradient_ops.ParamHistClipping()
    else:
        return None


def get_learning_scheme() -> LearningScheme:
    """Returns configured the learning scheme."""
    config = Experiment.get_config().lens.train.learn
    return LearningScheme(optimizer_cls=config.optimizer_cls,
                          base_lr=config.learning_rate,
                          scheduler=get_scheduler(config.scheduler),
                          optimizer_defaults=config.opt_settings,
                          gradient_op=get_grad_op(config)
                          )
