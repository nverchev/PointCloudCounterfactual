"""Train the outer encoder to learn a discrete representation."""

from typing import TYPE_CHECKING, Any
from drytorch import Test, Trainer, Model

from src.module import get_autoencoder, get_classifier, CounterfactualVAE, BaseClassifier
from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.train import get_autoencoder_loss, get_learning_schema
from src.train.hooks import (
    register_checkpointing,
    register_early_stopping,
    register_pruning,
    register_reconstruction_hook,
)
from src.train.loaders import get_loaders, get_evaluated_loaders
from src.train.models import ModelEpoch, EMAModelEpoch
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
    if trial is None:
        model = EMAModelEpoch(ae, name=cfg_ae.model.name, device=cfg_user.device)
    else:
        model = ModelEpoch(ae, name=cfg_ae.model.name, device=cfg_user.device)  # normal model for tuning

    # test_loader loads the validation dataset unless the final flag is set
    if isinstance(ae, CounterfactualVAE) and classifier is not None:
        train_loader, test_loader = get_evaluated_loaders(
            classifier, batch_size=cfg_ae.train.batch_size_per_device, n_workers=cfg_user.n_workers
        )
    else:
        train_loader, test_loader = get_loaders(
            batch_size=cfg_ae.train.batch_size_per_device, n_workers=cfg_user.n_workers
        )

    loss = get_autoencoder_loss()
    learning_schema = get_learning_schema(cfg.autoencoder)
    trainer = Trainer(model, loader=train_loader, loss=loss, learning_schema=learning_schema)
    test = Test(model, loader=test_loader, metric=loss)
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if not cfg.final:
        trainer.add_validation(test_loader)  # loads the validation dataset

    register_reconstruction_hook(trainer, cfg.autoencoder.train.learn.scheduler.restart_interval)
    if not cfg.final and cfg.autoencoder.train.early_stopping.active:
        cfg_early = cfg.autoencoder.train.early_stopping
        register_early_stopping(trainer, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(trainer, trial)
    else:
        register_checkpointing(trainer, cfg_user.checkpoint_every)

    trainer.train_until(cfg_ae.train.n_epochs)
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
