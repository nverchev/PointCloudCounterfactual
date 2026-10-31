"""Module for the learning schema configuration."""

from drytorch import LearningSchema
from drytorch.core import protocols as p
from drytorch.lib import gradient_ops, schedulers

from src.config.options import ClipCriterion, GradOp, Schedulers
from src.config.specs import PhaseConfig, LearningConfig, SchedulerConfig

__all__ = [
    'get_grad_op',
    'get_learning_schema',
    'get_scheduler',
]


def get_scheduler(config: SchedulerConfig, /) -> schedulers.AbstractScheduler:
    """Return the scheduler instance based on config.

    Args:
        config: scheduler configuration parameters.

    Returns:
        instantiated learning rate scheduler.

    Raises:
        ValueError: if scheduler function is not supported.
    """
    if config.function == Schedulers.CONSTANT:
        scheduler: schedulers.AbstractScheduler = schedulers.ConstantScheduler()
    elif config.function == Schedulers.COSINE:
        scheduler = schedulers.CosineScheduler(**config.settings)
    elif config.function == Schedulers.EXPONENTIAL:
        scheduler = schedulers.ExponentialScheduler(**config.settings)
    else:
        raise ValueError(f'Scheduler {config.function} not supported.')

    if config.restart_interval > 0:
        restart = schedulers.restart(restart_interval=config.restart_interval, restart_fraction=config.restart_fraction)
        scheduler = scheduler.compose(restart)

    if config.warmup_steps > 0:
        warmup = schedulers.warmup(config.warmup_steps)
        scheduler = scheduler.compose(warmup)

    return scheduler


def get_grad_op(config: LearningConfig, /) -> p.GradientOpProtocol:
    """Return the gradient clipping instance based on config.

    Args:
        config: learning configuration parameters.

    Returns:
        instantiated gradient operation.
    """
    if config.grad_op == GradOp.PARAM_NORMALIZER:
        return gradient_ops.ParamNormalizer()

    if config.grad_op == GradOp.Z_SCORE_NORMALIZER:
        return gradient_ops.ZScoreNormalizer()

    if config.grad_op == GradOp.VALUE_CLIPPER:
        return gradient_ops.ValueClipper()

    if config.grad_op == GradOp.NORM_CLIPPER:
        return gradient_ops.NormClipper()

    if config.grad_op == GradOp.HIST_CLIPPER:
        if config.clip_criterion == ClipCriterion.Z_STAT:
            return gradient_ops.HistClipper(criterion=gradient_ops.ZStatCriterion())

        if config.clip_criterion == ClipCriterion.EMA:
            return gradient_ops.HistClipper(criterion=gradient_ops.EMACriterion())

    if config.grad_op == GradOp.PARAM_HIST_CLIPPER:
        if config.clip_criterion == ClipCriterion.Z_STAT:
            return gradient_ops.ParamHistClipper(criterion=gradient_ops.ZStatCriterion())

        if config.clip_criterion == ClipCriterion.EMA:
            return gradient_ops.ParamHistClipper(criterion=gradient_ops.EMACriterion())

    return gradient_ops.NoOp()


def get_learning_schema(exp_cfg: PhaseConfig, /) -> LearningSchema:
    """Return the configured learning schema.

    Args:
        exp_cfg: experiment configuration object.

    Returns:
        configured learning schema instance.
    """
    config = exp_cfg.train.learn
    return LearningSchema(
        base_lr=config.lr,
        gradient_op=get_grad_op(config),
        optimizer_cls=config.optimizer,
        optimizer_defaults=config.opt_settings,
        scheduler=get_scheduler(config.scheduler),
    )
