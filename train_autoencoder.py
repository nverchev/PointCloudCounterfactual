"""Train the outer encoder to learn a discrete representation."""

import sys
from typing import Optional

import drytorch.core.exceptions
import optuna
import sqlalchemy
import wandb

from drytorch import DataLoader, Diagnostic, Model, Test, Trainer
from drytorch.contrib.optuna import TrialCallback
from drytorch.lib.hooks import Hook, EarlyStoppingCallback, StaticHook, call_every, saving_hook
from drytorch.trackers.csv import CSVDumper
from drytorch.trackers.hydra import HydraLink
from drytorch.trackers.sqlalchemy import SQLConnection
from drytorch.trackers.tensorboard import TensorBoard
from drytorch.trackers.wandb import Wandb
from drytorch.utils.average import get_moving_average, get_trailing_mean

from src.metrics_and_losses import get_autoencoder_loss, get_recon_loss, get_emd_loss
from src.config_options import Experiment, ConfigAll
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions
from src.hooks import DiscreteSpaceOptimizer, WandbLogReconstruction
from src.learning_scheme import get_learning_scheme
from src.autoencoder import get_autoencoder, AbstractVQVAE


def train_autoencoder(trial: Optional[optuna.Trial] = None) -> None:
    """Set up the experiment and launch the autoencoder training."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    ae = get_autoencoder()
    model = Model(ae, name=cfg_ae.architecture.name, device=cfg_user.device)

    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    with cfg.focus(cfg.autoencoder):
        learning_scheme = get_learning_scheme()
    loss = get_autoencoder_loss()
    trainer = Trainer(model,
                      loader=DataLoader(dataset=train_dataset, batch_size=cfg_ae.train.batch_size),
                      loss=loss,
                      learning_scheme=learning_scheme)
    diagnostic = Diagnostic(model,
                            loader=DataLoader(dataset=train_dataset, batch_size=cfg_ae.train.batch_size),
                            objective=loss
                            )

    test_all_metrics = Test(model,
                            loader=DataLoader(dataset=test_dataset, batch_size=cfg_ae.train.batch_size),
                            metric=loss | get_emd_loss())
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if isinstance(ae, AbstractVQVAE):
        rearrange_hook = StaticHook(DiscreteSpaceOptimizer(diagnostic)).bind(call_every(cfg_ae.diagnose_every))
        trainer.post_epoch_hooks.register(rearrange_hook)

    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        trainer.add_validation(DataLoader(dataset=val_dataset, batch_size=cfg_ae.train.batch_size))

    cfg_early = cfg_ae.train.early_stopping
    try:
        trainer.post_epoch_hooks.register(Hook(WandbLogReconstruction(train_dataset)))
    except drytorch.core.exceptions.DryTorchError:
        pass

    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=get_recon_loss(),
                                                                filter_fn=get_trailing_mean(cfg_early.window),
                                                                patience=cfg_early.patience))

    if trial is None:
        if checkpoint_every := cfg_user.checkpoint_every:
            trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    else:
        prune_hook = TrialCallback(trial,
                                   metric=get_recon_loss(),
                                   filter_fn=get_moving_average())

        trainer.post_epoch_hooks.register(prune_hook)

    trainer.train_until(cfg_ae.train.n_epochs)
    trainer.save_checkpoint()
    test_all_metrics()
    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up experiment and start training cycle."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    if not sys.gettrace():
        exp.trackers.register(Wandb(settings=wandb.Settings(project=cfg.project)))
        exp.trackers.register(HydraLink())
        exp.trackers.register(CSVDumper())
        exp.trackers.register(TensorBoard())
        engine_path = cfg.user.path.exp_par_dir / 'metrics.db'
        cfg.user.path.exp_par_dir.mkdir(exist_ok=True)
        engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
        exp.trackers.register(SQLConnection(engine=engine))
    with exp.create_run(resume=True):
        train_autoencoder()
    return


if __name__ == "__main__":
    main()
