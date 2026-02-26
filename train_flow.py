"""Train the flow matching model."""

from typing import TYPE_CHECKING, Any
from drytorch import Test, Trainer

from src.module import get_flow_module
from src.train.models import EMAModel
from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.train import get_flow_loss, get_learning_schema
from src.train.hooks import (
    register_checkpointing,
    register_early_stopping,
    register_pruning,
)
from src.train.loaders import get_loaders
from src.utils.parallel import DistributedWorker


if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_flow(trial: Trial | None = None) -> None:
    """Set up the experiment and launch the flow matching training."""
    cfg = Experiment.get_config()
    cfg_flow = cfg.flow
    cfg_user = cfg.user
    flow = get_flow_module()

    model = EMAModel(flow, name=cfg_flow.model.name, device=cfg_user.device)

    train_loader, test_loader = get_loaders(
        batch_size=cfg_flow.train.batch_size_per_device, n_workers=cfg_user.n_workers
    )

    loss = get_flow_loss()
    learning_schema = get_learning_schema(cfg.flow)
    trainer = Trainer(model, loader=train_loader, loss=loss, learning_schema=learning_schema)
    test = Test(model, loader=test_loader, metric=loss)

    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if not cfg.final:
        trainer.add_validation(test_loader)

    if not cfg.final and cfg.flow.train.early_stopping.active:
        cfg_early = cfg.flow.train.early_stopping
        register_early_stopping(trainer, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(trainer, trial)
    else:
        register_checkpointing(trainer, cfg_user.checkpoint_every)

    trainer.train_until(cfg_flow.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    test()
    return


def setup_and_train(cfg: AllConfig) -> None:
    """Set up experiment and start training cycle."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        # We don't include the classifier as an argument for now as per user request
        train_flow()

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Main entry point for module that creates subprocesses in parallel mode."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)


if __name__ == '__main__':
    main()
