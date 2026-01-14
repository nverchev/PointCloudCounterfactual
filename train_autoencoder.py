"""Train the outer encoder to learn a discrete representation."""

import pathlib

from typing import TYPE_CHECKING, Any

from drytorch import DataLoader, Diagnostic, Model, Test, Trainer
from drytorch.core.exceptions import TrackerNotActiveError
from drytorch.lib.hooks import EarlyStoppingCallback, Hook, StaticHook, call_every, saving_hook
from drytorch.utils.average import get_moving_average, get_trailing_mean

from src.module import AbstractVQVAE, get_autoencoder
from src.config import ConfigAll, Experiment, get_current_hydra_dir, get_trackers, hydra_main
from src.dataset import get_dataset_multiprocess_safe
from src.train.hooks import DiscreteSpaceOptimizer
from src.train.learning_schema import get_learning_schema
from src.train.metrics_and_losses import get_autoencoder_loss, get_emd_loss, get_recon_loss
from src.utils.parallel import DistributedWorker


if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_autoencoder(trial: Trial | None = None) -> None:
    """Set up the experiment and launch the autoencoder training."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    cfg_early = cfg_ae.train.early_stopping
    ae = get_autoencoder()
    model = Model(ae, name=cfg_ae.architecture.name, device=cfg_user.device)
    train_dataset, test_dataset = get_dataset_multiprocess_safe()  # test is validation unless final=True
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=cfg_ae.train.batch_size_per_device, n_workers=cfg_user.n_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg_ae.train.batch_size_per_device, n_workers=cfg_user.n_workers
    )
    with cfg.focus(cfg.autoencoder):
        learning_schema = get_learning_schema()

    loss = get_autoencoder_loss()
    trainer = Trainer(model, loader=train_loader, loss=loss, learning_schema=learning_schema)
    diagnostic = Diagnostic(model, loader=train_loader, objective=loss)
    test_all_metrics = Test(model, loader=test_loader, metric=loss | get_emd_loss())
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if isinstance(ae, AbstractVQVAE):
        rearrange_hook = StaticHook(DiscreteSpaceOptimizer(diagnostic)).bind(call_every(cfg_ae.diagnose_every))
        trainer.post_epoch_hooks.register(rearrange_hook)

    if not cfg.final:
        trainer.add_validation(test_loader)  # when not final, this uses the validation dataset

    try:
        from src.train.hooks import TensorBoardLogReconstruction

        restart_interval = cfg.autoencoder.train.learn.scheduler.restart_interval
        trainer.post_epoch_hooks.register(
            Hook(TensorBoardLogReconstruction(train_dataset)).bind(call_every(restart_interval))
        )
    except TrackerNotActiveError:  # tracker is not subscribed
        pass
    except (ImportError, ModuleNotFoundError):  # library is not installed
        pass

    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(
            EarlyStoppingCallback(
                metric=get_recon_loss(), filter_fn=get_trailing_mean(cfg_early.window), patience=cfg_early.patience
            )
        )

    if trial is None:
        if checkpoint_every := cfg_user.checkpoint_every:
            trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    else:
        from drytorch.contrib.optuna import TrialCallback

        prune_hook = TrialCallback(trial, metric=get_recon_loss(), filter_fn=get_moving_average())
        trainer.post_epoch_hooks.register(prune_hook)

    trainer.train_until(cfg_ae.train.n_epochs)
    trainer.save_checkpoint()
    test_all_metrics()
    return


def setup_and_train(cfg: ConfigAll, hydra_dir: pathlib.Path) -> None:
    """Set up experiment and start training cycle."""
    trackers = get_trackers(cfg, hydra_dir)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        train_autoencoder()

    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Main entry point for module that creates subprocesses in parallel mode."""
    n_processes = cfg.user.n_subprocesses
    hydra_dir = get_current_hydra_dir()
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).spawn(cfg, hydra_dir)
    else:
        setup_and_train(cfg, hydra_dir)


if __name__ == '__main__':
    main()
