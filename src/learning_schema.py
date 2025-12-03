"""Module for learning scheme configuration."""

from drytorch.lib import schedulers, gradient_ops
from drytorch import LearningSchema
from drytorch.core import protocols as p
from src.config_options import Schedulers, SchedulerConfig, Experiment, LearningConfig, GradOp, ClipCriterion


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


def get_grad_op(config: LearningConfig) -> p.GradientOpProtocol:
    """Returns the gradient clipping instance based on config."""
    if config.grad_op == GradOp.GradParamNormalizer:
        return gradient_ops.GradParamNormalizer()
    elif config.grad_op == GradOp.GradZScoreNormalizer:
        return gradient_ops.GradZScoreNormalizer()
    elif config.grad_op == GradOp.GradValueClipper:
        return gradient_ops.GradValueClipper()
    elif config.grad_op == GradOp.GradNormClipper:
        return gradient_ops.GradNormClipper()
    elif config.grad_op == GradOp.HistClipper:
        if config.clip_criterion == ClipCriterion.ZStat:
            return gradient_ops.HistClipper(criterion=gradient_ops.ZStatCriterion())
        elif config.clip_criterion == ClipCriterion.EMA:
            return gradient_ops.HistClipper(criterion=gradient_ops.EMACriterion())
    elif config.grad_op == GradOp.ParamHistClipper:
        if config.clip_criterion == ClipCriterion.ZStat:
            return gradient_ops.ParamHistClipper(criterion=gradient_ops.ZStatCriterion())
        elif config.clip_criterion == ClipCriterion.EMA:
            return gradient_ops.ParamHistClipper(criterion=gradient_ops.EMACriterion())

    return gradient_ops.NoOp()


def get_learning_schema() -> LearningSchema:
    """Returns configured the learning scheme."""
    config = Experiment.get_config().lens.train.learn
    return LearningSchema(optimizer_cls=config.optimizer_cls,
                          base_lr=config.learning_rate,
                          scheduler=get_scheduler(config.scheduler),
                          optimizer_defaults=config.opt_settings,
                          gradient_op=get_grad_op(config)
                          )
