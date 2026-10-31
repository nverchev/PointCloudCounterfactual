"""Train the classifier under audit.

Usage::

    uv run train_classifier.py --config-name modelnet_dt final=true
"""

from drytorch import Trainer

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import get_splits
from src.lib import get_cloud_loader, get_learning_schema
from src.lib.hooks import register_checkpointing, register_early_stopping
from src.lib.models import get_classifier_model
from src.lib.objectives import get_classification_loss
from src.trackers import get_trackers


def train_classifier() -> None:
    """Train the classifier and save it."""
    cfg = Experiment.get_config()
    cfg_phase = cfg.classification
    cfg_train = cfg_phase.train
    train_split, eval_split = get_splits()
    builder = Trainer.builder(
        get_classifier_model(),
        learning_schema=get_learning_schema(cfg_phase),
        loader=get_cloud_loader(train_split, batch_size=cfg_train.batch_size),
        loss=get_classification_loss(),
    )
    if not cfg.final:
        builder.add_validation(get_cloud_loader(eval_split, batch_size=cfg_train.batch_size))
        if cfg_train.early_stopping.active:
            cfg_early = cfg_train.early_stopping
            register_early_stopping(builder, patience=cfg_early.patience, window=cfg_early.window)

    register_checkpointing(builder, checkpoint_every=cfg.user.checkpoint_every)
    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    trainer.save_checkpoint()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the classifier."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    resume = cfg.user.load != 0 or cfg.user.run_id is not None
    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_classifier()

    return


if __name__ == '__main__':
    main()
