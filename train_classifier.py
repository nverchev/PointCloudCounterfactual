import torch
from dry_torch import DataLoader, Model, Test, Trainer
from dry_torch.hooks import EarlyStoppingCallback, call_every, saving_hook, mean_aggregation
from dry_torch.trackers.hydra_link import HydraLink
from torchmetrics.functional import confusion_matrix

from src.metrics_and_losses import get_classification_loss, get_cross_entropy_loss
from src.config_options import MainExperiment, ExperimentClassifier, ConfigAll
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions
from src.classifier import DGCNN
from src.learning_scheme import get_learning_scheme


def train_classifier() -> None:
    cfg = MainExperiment.get_config()
    cfg_class = cfg.classifier
    cfg_user = cfg.user
    module = DGCNN()
    model = Model(module, name=cfg_class.model.name, device=cfg_user.device)

    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    loss_calc = get_classification_loss()
    learning_scheme = get_learning_scheme()
    batch_size = cfg_class.train.batch_size
    trainer = Trainer(model,
                      loader=DataLoader(dataset=train_dataset, batch_size=batch_size),
                      loss=loss_calc,
                      learning_scheme=learning_scheme)

    test_all_metrics = Test(model,
                            loader=DataLoader(dataset=test_dataset, batch_size=batch_size),
                            metric=loss_calc)
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        trainer.add_validation(DataLoader(dataset=val_dataset, batch_size=batch_size))
    if not cfg.final and cfg_class.train.early_stopping.active:
        window = cfg_class.train.early_stopping.window
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc,
                                                                aggregate_fn=mean_aggregation(window)))
    if checkpoint_every := cfg_user.checkpoint_every:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    trainer.train_until(cfg_class.train.epochs)
    trainer.save_checkpoint()
    test_all_metrics(store_outputs=True)
    outputs_probs = torch.cat(test_all_metrics.outputs_list)
    predictions = outputs_probs.argmax(dim=1)
    with torch.inference_mode():
        labels = torch.cat([data[1].label for data in test_all_metrics.loader])
    print(confusion_matrix(preds=predictions, target=labels, task='multiclass', num_classes=outputs_probs.shape[1]))


@hydra_main
def main(cfg: ConfigAll) -> None:
    exp = ExperimentClassifier(cfg.classifier.name, config=cfg.classifier)
    exp.trackers.register(HydraLink())
    parent = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    parent.register_child(exp)
    with exp:
        train_classifier()
    return


if __name__ == "__main__":
    main()
