"""Train the w-autoencoder."""

from typing import TYPE_CHECKING, Any

import torch
from drytorch import DataLoader, Model, Test, Trainer
from drytorch.lib.hooks import EarlyStoppingCallback
from drytorch.utils.average import get_moving_average, get_trailing_mean

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.data import Inputs, get_datasets
from src.data.processed import WDatasetWithLogitsFrozen
from src.module import CounterfactualVQVAE, get_classifier
from src.train import get_learning_schema, get_w_autoencoder_loss
from src.train.models import ModelEpoch
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_w_autoencoder(
    vqvae: CounterfactualVQVAE, classifier: Model[Inputs, torch.Tensor], trial: Trial | None = None
) -> None:
    """Train the w-autoencoder."""
    cfg = Experiment.get_config()
    cfg_w_ae = cfg.w_autoencoder
    cfg_user = cfg.user
    module = vqvae.w_autoencoder
    w_encoder_model = ModelEpoch(module, name=cfg_w_ae.model.name)
    if not cfg_user.load_checkpoint:
        module.recursive_reset_parameters()

    module.update_codebook(vqvae.codebook.detach().clone())
    for param in module.parameters():
        param.requires_grad = True

    train_dataset, test_dataset = get_datasets()  # test is validation unless final=True
    train_w_dataset = WDatasetWithLogitsFrozen(train_dataset, vqvae, classifier)
    test_w_dataset = WDatasetWithLogitsFrozen(test_dataset, vqvae, classifier)
    test_loader = DataLoader(dataset=test_w_dataset, batch_size=cfg_w_ae.train.batch_size_per_device, pin_memory=False)
    loss_calc = get_w_autoencoder_loss()
    learning_schema = get_learning_schema(cfg.w_autoencoder)
    train_loader = DataLoader(
        dataset=train_w_dataset, batch_size=cfg_w_ae.train.batch_size_per_device, pin_memory=False
    )
    trainer = Trainer(w_encoder_model, loader=train_loader, loss=loss_calc, learning_schema=learning_schema)
    test_encoding = Test(w_encoder_model, loader=test_loader, metric=loss_calc)
    if not cfg.final:
        trainer.add_validation(test_loader)

    cfg_early = cfg_w_ae.train.early_stopping
    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(
            EarlyStoppingCallback(
                metric=loss_calc, filter_fn=get_trailing_mean(cfg_early.window), patience=cfg_early.patience
            )
        )
    if trial is not None:
        from drytorch.contrib.optuna import TrialCallback

        prune_hook = TrialCallback(trial, filter_fn=get_moving_average(), metric=loss_calc)
        trainer.post_epoch_hooks.register(prune_hook)

    if cfg_user.load_checkpoint >= 0:
        trainer.train_until(cfg_w_ae.train.n_epochs)

    test_encoding()
    return


def setup_and_train(cfg: AllConfig) -> None:
    """Set up the experiment, load the classifier and the autoencoder, and train the w-autoencoder."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        classifier_module = get_classifier()
        classifier = Model(
            classifier_module,
            name=cfg.classifier.model.name,
            device=cfg.user.device,
            should_compile=False,
            should_distribute=False,
        )
        classifier.load_state()
        module = CounterfactualVQVAE()
        autoencoder = Model(
            module,
            name=cfg.autoencoder.model.name,
            device=cfg.user.device,
            should_compile=False,
            should_distribute=False,
        )
        autoencoder.checkpoint.load()
        train_w_autoencoder(module, classifier)
        autoencoder.save_state()
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
