from typing import Optional, cast

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
from dry_torch import DataLoader, Diagnostic, Model, Test, Trainer
from dry_torch.hooks import EarlyStoppingCallback, StaticHook, call_every, saving_hook
from dry_torch.trackers.hydra_link import HydraLink

from src.calculating import get_autoencoder_loss, get_recon_loss, get_emd_loss
from src.config_options import ConfigTrainAE, ExperimentAE, ParentExperiment
from src.datasets import get_dataset, Partitions
from src.hooks import DiscreteSpaceOptimizer, TrialCallback
from src.learning_scheme import get_learning_scheme
from src.autoencoder import get_module, VQVAE

cs = hydra.core.config_store.ConfigStore.instance()  # type: ignore
cs.store(name='config_ae', node=ConfigTrainAE)


def train_autoencoder(trial: Optional[optuna.Trial] = None) -> None:
    cfg = ExperimentAE.get_config()
    module = get_module()
    model = Model(module, name=cfg.autoencoder.name, device=cfg.user.device)

    train_dataset = get_dataset(Partitions.train_val if cfg.exp.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.exp.final else Partitions.val)
    learning_scheme = get_learning_scheme()
    trainer = Trainer(model,
                      loader=DataLoader(dataset=train_dataset, batch_size=cfg.train.batch_size),
                      loss=get_autoencoder_loss(),
                      learning_scheme=learning_scheme)
    diagnostic = Diagnostic(model,
                            loader=DataLoader(dataset=train_dataset, batch_size=cfg.train.batch_size),
                            metric=get_autoencoder_loss())

    test_all_metrics = Test(model,
                            loader=DataLoader(dataset=test_dataset, batch_size=cfg.train.batch_size),
                            metric=get_autoencoder_loss() | get_emd_loss())
    if cfg.train.load_checkpoint:
        trainer.load_checkpoint(cfg.train.load_checkpoint)

    if isinstance(module, VQVAE):
        rearrange_hook = StaticHook(DiscreteSpaceOptimizer(diagnostic)).bind(call_every(cfg.train.diagnose_every))
        trainer.post_epoch_hooks.register(rearrange_hook)

    if not cfg.exp.final:
        val_dataset = get_dataset(Partitions.val)
        trainer.add_validation(DataLoader(dataset=val_dataset, batch_size=cfg.train.batch_size))
    if not cfg.exp.final and cfg.train.early_stopping is not None:
        cfg_early = cfg.train.early_stopping
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=get_recon_loss(),
                                                                patience=cfg_early.patience))

    if trial is None:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(cfg.user.checkpoint_every)))
    else:
        prune_hook = TrialCallback(trial)
        trainer.post_epoch_hooks.register(prune_hook)

    trainer.train_until(cfg.train.epochs)
    test_all_metrics()


@hydra.main(version_base=None, config_path="hydra_conf/autoencoder_conf/", config_name="defaults")
def main(dict_cfg: DictConfig) -> None:
    cfg = cast(ConfigTrainAE, OmegaConf.to_object(dict_cfg))
    exp = ExperimentAE(cfg.exp.name,  config=cfg)
    exp.trackers.register(HydraLink())
    ParentExperiment(cfg.exp.main_name, par_dir=cfg.user.path.exp_par_dir).register_child(exp)
    with exp:
        train_autoencoder()
    return


if __name__ == "__main__":
    main()
