from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf
from dry_torch import DataLoader, Model, Test, Trainer
from dry_torch.hooks import EarlyStoppingCallback
from dry_torch.trackers.hydra_link import HydraLink

from src.calculating import get_classification_loss, get_cross_entropy_loss
from src.config_options import ParentExperiment, ExperimentClassifier, ConfigTrainClassifier
from src.datasets import get_dataset, Partitions
from src.classifier import DGCNN
from src.learning_scheme import get_learning_scheme


def train_classifier() -> None:
    cfg = ExperimentClassifier.get_config()
    module = DGCNN()
    model = Model(module, name=cfg.classifier.name, device=cfg.user.device)

    train_dataset = get_dataset(Partitions.train_val if cfg.exp.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.exp.final else Partitions.val)
    loss_calc = get_classification_loss()
    learning_scheme = get_learning_scheme()
    batch_size = cfg.train.batch_size
    trainer = Trainer(model,
                      loader=DataLoader(dataset=train_dataset, batch_size=batch_size),
                      loss=loss_calc,
                      learning_scheme=learning_scheme)

    test_all_metrics = Test(model,
                            loader=DataLoader(dataset=test_dataset, batch_size=batch_size),
                            metric=get_classification_loss())
    if cfg.train.load_checkpoint:
        trainer.load_checkpoint(cfg.train.load_checkpoint)

    if not cfg.exp.final:
        val_dataset = get_dataset(Partitions.val)
        trainer.add_validation(DataLoader(dataset=val_dataset, batch_size=batch_size))
    if not cfg.exp.final and cfg.train.early_stopping is not None:
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=get_cross_entropy_loss()))

    trainer.train_until(cfg.train.epochs)
    trainer.save_checkpoint()
    test_all_metrics()


@hydra.main(version_base=None, config_path="hydra_conf/classifier_conf/", config_name="defaults")
def main(dict_cfg: DictConfig) -> None:
    cfg = cast(ConfigTrainClassifier, OmegaConf.to_object(dict_cfg))
    exp = ExperimentClassifier(cfg.exp.name, config=cfg)
    exp.trackers.register(HydraLink())
    ParentExperiment(cfg.exp.main_name, par_dir=cfg.user.path.exp_par_dir).register_child(exp)
    with exp:
        train_classifier()
    return


if __name__ == "__main__":
    main()
