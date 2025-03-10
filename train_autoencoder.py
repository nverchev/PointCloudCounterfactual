from typing import Optional

import optuna
from dry_torch import DataLoader, Diagnostic, Model, Test, Trainer
from dry_torch.hooks import EarlyStoppingCallback, StaticHook, call_every, saving_hook, mean_aggregation
from dry_torch.trackers.hydra_link import HydraLink

from src.metrics_and_losses import get_autoencoder_loss, get_recon_loss, get_emd_loss
from src.config_options import ExperimentAE, MainExperiment, ConfigAll
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions
from src.hooks import DiscreteSpaceOptimizer, TrialCallback
from src.learning_scheme import get_learning_scheme
from src.autoencoder import get_module, VQVAE


def train_autoencoder(trial: Optional[optuna.Trial] = None) -> None:
    cfg = MainExperiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    module = get_module()
    model = Model(module, name=cfg_ae.model.name, device=cfg_user.device)

    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    learning_scheme = get_learning_scheme()
    loss = get_autoencoder_loss()
    trainer = Trainer(model,
                      loader=DataLoader(dataset=train_dataset, batch_size=cfg_ae.train.batch_size),
                      loss=loss,
                      learning_scheme=learning_scheme)
    diagnostic = Diagnostic(model,
                            loader=DataLoader(dataset=train_dataset, batch_size=cfg_ae.train.batch_size),
                            metric=loss)

    test_all_metrics = Test(model,
                            loader=DataLoader(dataset=test_dataset, batch_size=cfg_ae.train.batch_size),
                            metric=loss | get_emd_loss())
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if isinstance(module, VQVAE):
        rearrange_hook = StaticHook(DiscreteSpaceOptimizer(diagnostic)).bind(call_every(cfg_ae.diagnose_every))
        trainer.post_epoch_hooks.register(rearrange_hook)

    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        trainer.add_validation(DataLoader(dataset=val_dataset, batch_size=cfg_ae.train.batch_size))

    cfg_early = cfg_ae.train.early_stopping
    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=get_recon_loss(),
                                                                aggregate_fn=mean_aggregation(cfg_early.window),
                                                                patience=cfg_early.patience))

    if trial is None:
        if checkpoint_every := cfg_user.checkpoint_every:
            trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    else:
        prune_hook = TrialCallback(trial,
                                   metric=get_recon_loss(),
                                   aggregate_fn=mean_aggregation(window=cfg_early.window))

        trainer.post_epoch_hooks.register(prune_hook)

    trainer.train_until(cfg_ae.train.epochs)
    DiscreteSpaceOptimizer(diagnostic)()
    trainer.save_checkpoint()
    test_all_metrics()
    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    exp = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    exp.trackers.register(HydraLink())
    parent = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    parent.register_child(exp)
    with exp:
        train_autoencoder()
    return


if __name__ == "__main__":
    main()
