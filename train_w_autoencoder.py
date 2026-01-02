""" Train the w-autoencoder."""

from __future__ import annotations

import pathlib
from typing import Optional, TYPE_CHECKING

import torch

from drytorch import DataLoader, Model, Test, Trainer
from drytorch.lib.hooks import EarlyStoppingCallback
from drytorch.utils.average import get_moving_average, get_trailing_mean


from drytorch.contrib.optuna import TrialCallback
from src.classifier import DGCNN
from src.data_structures import Inputs
from src.metrics_and_losses import get_w_encoder_loss
from src.config_options import Experiment, ConfigAll, get_current_hydra_dir, get_trackers
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions, WDatasetWithLogits, WDatasetWithLogitsFrozen
from src.learning_schema import get_learning_schema
from src.models import ModelEpoch
from src.autoencoder import CounterfactualVQVAE
from src.parallel import DistributedWorker
# from src.visualisation import show_latent

if TYPE_CHECKING:
    import optuna


def train_w_autoencoder(vqvae: CounterfactualVQVAE,
                        classifier: Model[Inputs, torch.Tensor],
                        name: str,
                        trial: Optional[optuna.Trial] = None) -> None:
    """Train the w-autoencoder."""
    cfg = Experiment.get_config()
    cfg_w_ae = cfg.w_autoencoder
    cfg_user = cfg.user
    module = vqvae.w_autoencoder
    for param in module.parameters():
        param.requires_grad = True

    w_encoder_model = ModelEpoch(module, name=f'{name:s}.WAutoEncoder')
    if not cfg_user.load_checkpoint:
        module.recursive_reset_parameters()

    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    train_w_dataset = WDatasetWithLogitsFrozen(train_dataset, vqvae, classifier)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    test_w_dataset = WDatasetWithLogitsFrozen(test_dataset, vqvae, classifier)
    test_loader = DataLoader(dataset=test_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)
    loss_calc = get_w_encoder_loss()
    with cfg.focus(cfg.w_autoencoder):
        learning_schema = get_learning_schema()
    train_loader = DataLoader(dataset=train_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)

    trainer = Trainer(w_encoder_model,
                      loader=train_loader,
                      loss=loss_calc,
                      learning_schema=learning_schema)

    test_encoding = Test(w_encoder_model, loader=test_loader, metric=loss_calc)

    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        val_w_dataset = WDatasetWithLogitsFrozen(val_dataset, vqvae, classifier)
        val_loader = DataLoader(dataset=val_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)
        trainer.add_validation(val_loader)

    cfg_early = cfg_w_ae.train.early_stopping
    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc,
                                                                filter_fn=get_trailing_mean(cfg_early.window),
                                                                patience=cfg_early.patience))
    if trial is not None:
        prune_hook = TrialCallback(trial,
                                   filter_fn=get_moving_average(),
                                   metric=loss_calc)
        trainer.post_epoch_hooks.register(prune_hook)

    trainer.train_until(cfg_w_ae.train.n_epochs)
    test_encoding()
    return


def setup_and_train(cfg: ConfigAll, hydra_dir: pathlib.Path) -> None:
    """Set up the experiment, load the classifier and the autoencoder, and train the w-autoencoder."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    for tracker in get_trackers(cfg, hydra_dir):
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        classifier_module = DGCNN()
        classifier = Model(classifier_module, name=cfg.classifier.architecture.name, device=cfg.user.device)
        classifier.load_state()
        module = CounterfactualVQVAE()
        autoencoder = Model(module, name=cfg.autoencoder.architecture.name, device=cfg.user.device)
        autoencoder.checkpoint.load()
        train_w_autoencoder(module, classifier, name=autoencoder.name)
        autoencoder.save_state()
    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Main entry point for module that creates subprocesses in parallel mode."""
    n_processes = cfg.user.n_parallel_training_processes
    hydra_dir = get_current_hydra_dir()
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).process(cfg, hydra_dir)
    else:
        setup_and_train(cfg, hydra_dir)


if __name__ == "__main__":
    main()
