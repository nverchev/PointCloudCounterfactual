from typing import Optional

import optuna
from dry_torch.trackers.hydra_link import HydraLink
import torch

from dry_torch import DataLoader, Model, Test, Trainer
from dry_torch.hooks import EarlyStoppingCallback, mean_aggregation

from src.classifier import DGCNN
from src.data_structures import Inputs
from src.hooks import TrialCallback
from src.metrics_and_losses import get_w_encoder_loss, get_recon_loss
from src.config_options import ExperimentWAE, ExperimentAE, MainExperiment, ExperimentClassifier, ConfigAll
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions, WDatasetWithProbs
from src.learning_scheme import get_learning_scheme
from src.autoencoder import VQVAE
from src.visualisation import show_latent


def train_w_autoencoder(vqvae: VQVAE,
                        classifier: Model[Inputs, torch.Tensor],
                        name: str,
                        trial: Optional[optuna.Trial] = None) -> None:
    cfg = MainExperiment.get_config()
    cfg_w_ae = cfg.w_autoencoder
    cfg_user = cfg.user
    module = vqvae.w_autoencoder
    w_encoder_model = Model(module, name=f'{name:s}.WAutoEncoder')

    if not cfg_user.load_checkpoint:
        module.recursive_reset_parameters()
    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    train_w_dataset = WDatasetWithProbs(train_dataset, vqvae, classifier)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    test_w_dataset = WDatasetWithProbs(test_dataset, vqvae, classifier)
    test_loader = DataLoader(dataset=test_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)

    loss_calc = get_w_encoder_loss()
    learning_scheme = get_learning_scheme()
    train_loader = DataLoader(dataset=train_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)

    trainer = Trainer(w_encoder_model,
                      loader=train_loader,
                      loss=loss_calc,
                      learning_scheme=learning_scheme)

    test_encoding = Test(w_encoder_model, loader=test_loader, metric=loss_calc)

    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        val_w_dataset = WDatasetWithProbs(val_dataset, vqvae, classifier)
        val_loader = DataLoader(dataset=val_w_dataset, batch_size=cfg_w_ae.train.batch_size, pin_memory=False)
        trainer.add_validation(val_loader)

    cfg_early = cfg_w_ae.train.early_stopping
    if not cfg.final and cfg_early.active:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc,
                                                                aggregate_fn=mean_aggregation(window=cfg_early.window),
                                                                patience=cfg_early.patience))
    if trial is not None:
        prune_hook = TrialCallback(trial,
                                   aggregate_fn=mean_aggregation(window=cfg_early.window),
                                   metric=loss_calc)
        trainer.post_epoch_hooks.register(prune_hook)

    # trainer.post_epoch_hooks.register(lambda trainer: print(trainer.model.module.temperature))


    trainer.train_until(cfg_w_ae.train.epochs)

    test_encoding()
    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    parent_experiment = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    exp_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    exp_wae = ExperimentWAE(cfg.w_autoencoder.name, config=cfg.w_autoencoder)
    exp_classifier = ExperimentClassifier(cfg.classifier.name, config=cfg.classifier)
    exp_wae.trackers.register(HydraLink())
    parent_experiment.register_child(exp_wae)
    parent_experiment.register_child(exp_ae)
    parent_experiment.register_child(exp_classifier)

    with exp_classifier:
        classifier_module = DGCNN()
        classifier = Model(classifier_module, name=cfg.classifier.model.name, device=cfg.user.device)
        classifier.load_state()
    with exp_ae:
        module = VQVAE()
        autoencoder = Model(module, name=cfg.autoencoder.model.name, device=cfg.user.device)
        paths = autoencoder._model_state_io.paths
        autoencoder.epoch = autoencoder._model_state_io._get_last_saved_epoch()
        state_dict = torch.load(paths.state_path, map_location=autoencoder.device)
        state_dict = {key: value for key, value in state_dict.items() if 'w_autoencoder' not in key}
        autoencoder.module.load_state_dict(state_dict, strict=False)
    with exp_wae:
        train_w_autoencoder(module, classifier, name=autoencoder.name)
    with exp_ae:
        autoencoder.save_state()
        test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
        test = Test(autoencoder,
                    name='DoubleEncoding',
                    loader=DataLoader(dataset=test_dataset, batch_size=cfg.autoencoder.train.batch_size),
                    metric=get_recon_loss())
        with module.double_encoding:
            test()
    return


if __name__ == "__main__":
    main()
