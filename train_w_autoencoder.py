""" Train the w-autoencoder."""

import sys
from typing import Optional

import optuna
import sqlalchemy
import torch
import wandb

from drytorch import DataLoader, Model, Test, Trainer
from drytorch.lib.hooks import EarlyStoppingCallback
from drytorch.trackers.sqlalchemy import SQLConnection
from drytorch.trackers.tensorboard import TensorBoard
from drytorch.utils.average import get_trailing_mean
from drytorch.trackers.csv import CSVDumper
from drytorch.trackers.hydra import HydraLink
from drytorch.trackers.wandb import Wandb

from drytorch.contrib.optuna import TrialCallback
from src.classifier import DGCNN
from src.data_structures import Inputs
from src.metrics_and_losses import get_w_encoder_loss, get_recon_loss
from src.config_options import Experiment, ConfigAll
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions, WDatasetWithLogits
from src.learning_scheme import get_learning_scheme
from src.autoencoder import CounterfactualVQVAE
# from src.visualisation import show_latent


def train_w_autoencoder(vqvae: CounterfactualVQVAE,
                        classifier: Model[Inputs, torch.Tensor],
                        name: str,
                        trial: Optional[optuna.Trial] = None) -> None:
    """Train the w-autoencoder."""
    cfg = Experiment.get_config()
    cfg_w_ae = cfg.w_autoencoder
    cfg_user = cfg.user
    module = vqvae.w_autoencoder
    w_encoder_model = Model(module, name=f'{name:s}.WAutoEncoder')

    if not cfg_user.load_checkpoint:
        module.recursive_reset_parameters()
    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    train_w_dataset = WDatasetWithLogits(train_dataset, vqvae, classifier)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    test_w_dataset = WDatasetWithLogits(test_dataset, vqvae, classifier)
    test_loader = DataLoader(dataset=test_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)

    loss_calc = get_w_encoder_loss()
    with cfg.focus(cfg.w_autoencoder):
        learning_scheme = get_learning_scheme()
    train_loader = DataLoader(dataset=train_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)

    trainer = Trainer(w_encoder_model,
                      loader=train_loader,
                      loss=loss_calc,
                      learning_scheme=learning_scheme)

    test_encoding = Test(w_encoder_model, loader=test_loader, metric=loss_calc)

    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        val_w_dataset = WDatasetWithLogits(val_dataset, vqvae, classifier)
        val_loader = DataLoader(dataset=val_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)
        trainer.add_validation(val_loader)

    cfg_early = cfg_w_ae.train.early_stopping
    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc,
                                                                filter_fn=get_trailing_mean(cfg_early.window),
                                                                patience=cfg_early.patience))
    if trial is not None:
        prune_hook = TrialCallback(trial,
                                   filter_fn=get_trailing_mean(cfg_early.window),
                                   metric=loss_calc)
        trainer.post_epoch_hooks.register(prune_hook)

    # trainer.post_epoch_hooks.register(lambda trainer: print(trainer.model.module.temperature))
    trainer.train_until(cfg_w_ae.train.n_epochs)
    test_encoding()
    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up the experiment, load the classifier and the autoencoder, and train the w-autoencoder."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    if not sys.gettrace():
        exp.trackers.register(HydraLink())
        exp.trackers.register(CSVDumper())
        # exp.trackers.register(Wandb(settings=wandb.Settings(project=cfg.project)))
        exp.trackers.register(TensorBoard())
        engine_path = cfg.user.path.exp_par_dir / 'metrics.db'
        cfg.user.path.exp_par_dir.mkdir(exist_ok=True)
        engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
        # exp.trackers.register(SQLConnection(engine=engine))
    with exp.create_run(resume=True):
        classifier_module = DGCNN()
        classifier = Model(classifier_module, name=cfg.classifier.architecture.name, device=cfg.user.device)
        classifier.load_state()
        module = CounterfactualVQVAE()
        autoencoder = Model(module, name=cfg.autoencoder.architecture.name, device=cfg.user.device)
        autoencoder.checkpoint.load()
        train_w_autoencoder(module, classifier, name=autoencoder.name)
        autoencoder.save_state()
        test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
        test = Test(autoencoder,
                    name='DoubleEncoding',
                    loader=DataLoader(dataset=test_dataset, batch_size=cfg.autoencoder.train.batch_size),
                    metric=get_recon_loss())
        # with module.double_encoding:
        #     test()
    return


if __name__ == "__main__":
    main()
