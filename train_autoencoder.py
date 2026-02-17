"""Train the outer encoder to learn a discrete representation."""

from typing import TYPE_CHECKING, Any

from drytorch import DataLoader, Model, Test, Trainer
from drytorch.core.exceptions import TrackerNotUsedError
from drytorch.lib.hooks import EarlyStoppingCallback, Hook, call_every, saving_hook
from drytorch.utils.average import get_trailing_mean, get_moving_average

from src.data import get_datasets
from src.data.processed import EvaluatedDataset
from src.module import get_autoencoder, get_classifier, CounterfactualVAE, BaseClassifier
from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.train import get_autoencoder_loss, get_learning_schema
from src.train.models import ModelEpoch
from src.utils.parallel import DistributedWorker


if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_autoencoder(classifier: BaseClassifier | None = None, trial: Trial | None = None) -> None:
    """Set up the experiment and launch the autoencoder training."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    ae = get_autoencoder()
    model = ModelEpoch(ae, name=cfg_ae.model.name, device=cfg_user.device)
    if isinstance(ae, CounterfactualVAE) and classifier is not None:
        train_loader, test_loader = _get_evaluated_loaders(classifier)
    else:
        train_loader, test_loader = _get_loaders()  # test_loader loads the validation dataset unless final=True

    loss = get_autoencoder_loss()
    learning_schema = get_learning_schema(cfg.autoencoder)
    trainer = Trainer(model, loader=train_loader, loss=loss, learning_schema=learning_schema)
    test = Test(model, loader=test_loader, metric=loss)
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if not cfg.final:
        trainer.add_validation(test_loader)  # loads the validation dataset

    _register_reconstruction_hook(trainer)
    if not cfg.final and cfg.autoencoder.train.early_stopping.active:
        _register_early_stopping(trainer)

    if trial is not None:
        _register_pruning(trainer, trial)
    else:
        _register_checkpointing(trainer)

    trainer.train_until(cfg_ae.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    test()
    return


def _get_evaluated_loaders(classifier: BaseClassifier) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders for training and testing with classifier evaluation."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    train_dataset, test_dataset = get_datasets()  # test is validation unless final=True
    processed_train_dataset = EvaluatedDataset(train_dataset, classifier)
    processed_test_dataset = EvaluatedDataset(test_dataset, classifier)
    train_loader = DataLoader(
        dataset=processed_train_dataset,
        batch_size=cfg_ae.train.batch_size_per_device,
        n_workers=cfg_user.n_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        dataset=processed_test_dataset,
        batch_size=cfg_ae.train.batch_size_per_device,
        n_workers=cfg_user.n_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def _get_loaders() -> tuple[DataLoader, DataLoader]:
    """Get dataloaders for training and testing."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    train_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=cfg_ae.train.batch_size_per_device, n_workers=cfg_user.n_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg_ae.train.batch_size_per_device, n_workers=cfg_user.n_workers
    )
    return train_loader, test_loader


def _register_checkpointing(trainer: Trainer) -> None:
    """Register the checkpointing hook."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    if checkpoint_every := cfg_user.checkpoint_every:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))

    return


def _register_early_stopping(trainer: Trainer) -> None:
    """Register the early stopping hook."""
    cfg = Experiment.get_config()
    cfg_early = cfg.autoencoder.train.early_stopping
    trainer.post_epoch_hooks.register(
        EarlyStoppingCallback(
            metric=trainer.objective, filter_fn=get_trailing_mean(cfg_early.window), patience=cfg_early.patience
        )
    )
    return


def _register_pruning(trainer: Trainer, trial: Trial) -> None:
    """Register the pruning hook."""
    from drytorch.contrib.optuna import TrialCallback

    prune_hook = TrialCallback(trial, metric=trainer.objective, filter_fn=get_moving_average())
    trainer.post_epoch_hooks.register(prune_hook)
    return


def _register_reconstruction_hook(trainer: Trainer) -> None:
    """Register the reconstruction hook."""
    cfg = Experiment.get_config()
    try:
        from src.train.hooks import TensorBoardLogReconstruction

        restart_interval = cfg.autoencoder.train.learn.scheduler.restart_interval
        trainer.post_epoch_hooks.register(
            Hook(TensorBoardLogReconstruction(trainer.loader.dataset)).bind(call_every(restart_interval))
        )
    except TrackerNotUsedError:  # tracker is not subscribed
        pass
    except (ImportError, ModuleNotFoundError):  # library is not installed
        pass

    return


def setup_and_train(cfg: AllConfig) -> None:
    """Set up experiment and start training cycle."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        cfg_classifier = cfg.classifier
        cfg_user = cfg.user
        classifier = get_classifier()
        classifier_model = Model(classifier, name=cfg_classifier.model.name, device=cfg_user.device)
        classifier_model.load_state(-1)
        train_autoencoder(classifier=classifier)

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
