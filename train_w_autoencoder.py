from typing import cast

import hydra
from omegaconf import OmegaConf
from dry_torch.trackers.hydra_link import HydraLink

from dry_torch.hooks import call_every, StaticHook
from omegaconf import DictConfig
from dry_torch import DataLoader, Model, Test, Trainer, Validation

from src.calculating import get_w_encoder_loss, get_chamfer_loss, get_emd_loss
from src.config_options import ExperimentWAE
from src.datasets import get_dataset, Partitions, WDataset
from src.learning_scheme import get_learning_scheme
from src.autoencoder import VQVAE
from src.config_options import ParentExperiment, ConfigTrainWAE


def train_w_autoencoder() -> None:
    cfg = ExperimentWAE.get_config()
    module = VQVAE()
    model = Model(module, name=cfg.autoencoder.name, device=cfg.user.device)
    model.load_state(-1)
    w_encoder_model = Model(module.w_encoder, name=cfg.autoencoder.name + '.WEncoder')

    if not cfg.train.load_checkpoint:
        module.w_encoder.recursive_reset_parameters()
    train_dataset = get_dataset(Partitions.train_val if cfg.exp.final else Partitions.train)
    train_w_dataset = WDataset(train_dataset, module)

    test_dataset = get_dataset(Partitions.test if cfg.exp.final else Partitions.val)
    test_w_dataset = WDataset(test_dataset, module)
    test_loader = DataLoader(dataset=test_w_dataset, batch_size=cfg.train.batch_size)
    test_loader.set_pin_memory(False)

    loss_calc = get_w_encoder_loss()
    learning_scheme = get_learning_scheme()
    train_loader = DataLoader(dataset=train_w_dataset, batch_size=cfg.train.batch_size)
    train_loader.set_pin_memory(False)

    trainer = Trainer(w_encoder_model,
                      loader=train_loader,
                      loss=loss_calc,
                      learning_scheme=learning_scheme)

    test_encoding = Test(w_encoder_model, loader=test_loader, metric=loss_calc)
    test_all_metrics = Test(model,
                            loader=DataLoader(dataset=test_dataset, batch_size=cfg.train.batch_size),
                            metric=get_chamfer_loss() | get_emd_loss())

    if not cfg.exp.final:
        val_dataset = get_dataset(Partitions.val)
        val_w_dataset = WDataset(val_dataset, module)
        val_loader = DataLoader(dataset=val_w_dataset, batch_size=cfg.train.batch_size)
        val_loader.set_pin_memory(False)
        validate = Validation(w_encoder_model,
                              loader=val_loader,
                              metric=loss_calc)
        trainer.post_epoch_hooks.register(StaticHook(validate).bind(call_every(cfg.user.validate)))

    if cfg.user.plot.training:
        pass
        # metric_names = ['KLD', 'Accuracy']
        # plotting_hook = call_every(
        #     cfg.user.plot.refresh, plot_learning_curves(w_encoder_model, metric_names, 'W Encoding')
        # )
        # trainer.post_epoch_hooks.register(plotting_hook)

    if not cfg.train.load_checkpoint:
        trainer.train_until(cfg.train.epochs)
        model.save_state()

    test_encoding()
    with module.double_encoding:
        test_all_metrics(store_outputs=True)
    # mu = torch.vstack([output.mu for output in test_all_metrics.outputs_list])
    # mu_np = mu.numpy()
    # module.cw_encoder.update_pseudo_latent()
    # pseudo_mu = module.cw_encoder.pseudo_mu.detach().cpu().numpy()
    # show_latent(mu_np, pseudo_mu, model.name)
    return


@hydra.main(version_base=None, config_path="hydra_conf/w_autoencoder_conf/", config_name="defaults")
def main(dict_cfg: DictConfig) -> None:
    cfg = cast(ConfigTrainWAE, OmegaConf.to_object(dict_cfg))
    exp = ExperimentWAE(cfg.exp.name, config=cfg)
    exp.trackers.register(HydraLink())
    ParentExperiment(cfg.exp.main_name, par_dir=cfg.user.path.exp_par_dir).register_child(exp)
    with exp:
        train_w_autoencoder()
    return


if __name__ == "__main__":
    main()
