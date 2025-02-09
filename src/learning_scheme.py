from dry_torch import schedulers
from dry_torch import LearningScheme

from src.config_options import ParentExperiment, Schedulers, SchedulerConfig, LearningConfig


def get_scheduler(config: SchedulerConfig) -> schedulers.AbstractScheduler:
    if config.function == Schedulers.Constant:
        return schedulers.ConstantScheduler()
    if config.function == Schedulers.Cosine:
        decay_steps = config.settings['decay_steps']
        min_decay = config.settings['min_decay']
        return schedulers.CosineScheduler(decay_steps=decay_steps, min_decay=min_decay)
    if config.function == Schedulers.Exponential:
        exp_decay = config.settings['exp_decay']
        min_decay = config.settings['min_decay']
        return schedulers.ExponentialScheduler(exp_decay=exp_decay, min_decay=min_decay)
    else:
        raise ValueError(f'Unsupported scheduler function: {config.function}')


def get_learning_scheme() -> LearningScheme:
    config = ParentExperiment.get_child_config().train.learn
    return LearningScheme(
        optimizer_cls=config.optimizer_cls,
        base_lr=config.learning_rate,
        scheduler=get_scheduler(config.scheduler),
        optimizer_defaults=config.settings,
    )

