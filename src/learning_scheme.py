from dry_torch import schedulers
from dry_torch import LearningScheme

from src.config_options import MainExperiment, Schedulers, SchedulerConfig


def get_scheduler(config: SchedulerConfig) -> schedulers.AbstractScheduler:
    if config.function == Schedulers.Constant:
        return schedulers.ConstantScheduler()
    if config.function == Schedulers.Cosine:
        return schedulers.CosineScheduler(**config.settings)
    if config.function == Schedulers.Exponential:
        return schedulers.ExponentialScheduler(**config.settings)
    else:
        raise ValueError(f'Unsupported scheduler function: {config.function}')


def get_learning_scheme() -> LearningScheme:
    config = MainExperiment.get_child_config().train.learn
    scheduler = schedulers.WarmupScheduler(config.warm_up, get_scheduler(config.scheduler))

    return LearningScheme(
        optimizer_cls=config.optimizer_cls,
        base_lr=config.learning_rate,
        scheduler=scheduler,
        optimizer_defaults=config.settings,
    )
