""" Train the w-autoencoder."""

from __future__ import annotations

import pathlib
from typing import Optional, TYPE_CHECKING, Any

import torch

from drytorch import DataLoader, Model, Test, Trainer
from drytorch.lib.hooks import EarlyStoppingCallback
from drytorch.utils.average import get_moving_average, get_trailing_mean


from src.classifier import DGCNN
from src.data_structures import Inputs
from src.metrics_and_losses import get_w_encoder_loss
from src.config_options import Experiment, ConfigAll, get_current_hydra_dir, get_trackers
from src.config_options import hydra_main
from src.datasets import get_dataset_multiprocess_safe, WDatasetWithLogits
from src.learning_schema import get_learning_schema
from src.models import ModelEpoch
from src.autoencoder import CounterfactualVQVAE
from src.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_w_autoencoder(vqvae: CounterfactualVQVAE,
                        classifier: Model[Inputs, torch.Tensor],
                        name: str,
                        trial: Trial | None = None) -> None:
    """Train the w-autoencoder."""
    cfg = Experiment.get_config()
    cfg_w_ae = cfg.w_autoencoder
    cfg_user = cfg.user
    module = vqvae.w_autoencoder
    module.update_codebook(vqvae.codebook)
    for param in module.parameters():
        param.requires_grad = True

    w_encoder_model = ModelEpoch(module, name=f'{name:s}.WAutoEncoder')
    if not cfg_user.load_checkpoint:
        module.recursive_reset_parameters()

    train_dataset, test_dataset = get_dataset_multiprocess_safe()  # test is validation unless final=True

    train_w_dataset = WDatasetWithLogits(train_dataset, vqvae, classifier)
    test_w_dataset = WDatasetWithLogits(test_dataset, vqvae, classifier)
    test_loader = DataLoader(dataset=test_w_dataset, batch_size=cfg_w_ae.train.batch_size_per_device, pin_memory=False)
    loss_calc = get_w_encoder_loss()

    with cfg.focus(cfg.w_autoencoder):
        learning_schema = get_learning_schema()

    train_loader = DataLoader(dataset=train_w_dataset, batch_size=cfg_w_ae.train.batch_size_per_device, pin_memory=False)
    trainer = Trainer(w_encoder_model,
                      loader=train_loader,
                      loss=loss_calc,
                      learning_schema=learning_schema)
    test_encoding = Test(w_encoder_model, loader=test_loader, metric=loss_calc)

    if not cfg.final:
        trainer.add_validation(test_loader)

    cfg_early = cfg_w_ae.train.early_stopping
    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc,
                                                                filter_fn=get_trailing_mean(cfg_early.window),
                                                                patience=cfg_early.patience))
    if trial is not None:
        from drytorch.contrib.optuna import TrialCallback

        prune_hook = TrialCallback(trial,
                                   filter_fn=get_moving_average(),
                                   metric=loss_calc)
        trainer.post_epoch_hooks.register(prune_hook)

    if cfg_user.load_checkpoint >= 0:
        trainer.train_until(cfg_w_ae.train.n_epochs)

    test_encoding()
    return


def setup_and_train(cfg: ConfigAll, hydra_dir: pathlib.Path) -> None:
    """Set up the experiment, load the classifier and the autoencoder, and train the w-autoencoder."""
    trackers = get_trackers(cfg, hydra_dir)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        classifier_module = DGCNN()
        classifier = Model(
            classifier_module,
            name=cfg.classifier.architecture.name,
            device=cfg.user.device,
            should_compile=False,
            should_distribute=False)
        classifier.load_state()
        module = CounterfactualVQVAE()
        autoencoder = Model(module,
                            name=cfg.autoencoder.architecture.name,
                            device=cfg.user.device,
                            should_compile=False,
                            should_distribute=False)
        autoencoder.checkpoint.load()
        train_w_autoencoder(module, classifier, name=autoencoder.name)
        autoencoder.save_state()
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


if __name__ == "__main__":
    main()
