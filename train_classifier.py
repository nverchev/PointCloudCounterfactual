"""Module for training and testing a DGCNN classifier on point cloud data."""
import sys

import torch
import sqlalchemy
import wandb

from drytorch.core. exceptions import TrackerNotActiveError
from drytorch import DataLoader, Model, Test, Trainer
from drytorch.lib.hooks import EarlyStoppingCallback, call_every, saving_hook
from drytorch.trackers.wandb import Wandb
from drytorch.trackers.sqlalchemy import SQLConnection
from drytorch.utils.average import get_trailing_mean
from drytorch.trackers.hydra import HydraLink
from drytorch.trackers.tensorboard import TensorBoard
from drytorch.trackers.csv import CSVDumper
from torchmetrics import ConfusionMatrix

from src.metrics_and_losses import get_classification_loss
from src.config_options import Experiment, ConfigAll
from src.config_options import hydra_main
from src.datasets import get_dataset, Partitions
from src.classifier import DGCNN
from src.learning_scheme import get_learning_scheme
from src.visualisation import plot_confusion_matrix_heatmap


def train_classifier() -> None:
    """Train a DGCNN classifier and test its performance."""
    cfg = Experiment.get_config()
    cfg_class = cfg.classifier
    cfg_user = cfg.user

    module = DGCNN()
    model = Model(module, name=cfg_class.architecture.name, device=cfg_user.device)

    train_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.train)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg_class.train.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_class.train.batch_size)
    loss_calc = get_classification_loss()
    with cfg.focus(cfg.classifier):
        learning_scheme = get_learning_scheme()
    batch_size = cfg_class.train.batch_size
    trainer = Trainer(model, loader=train_loader, loss=loss_calc, learning_scheme=learning_scheme)
    final_test = Test(model, loader=test_loader, metric=loss_calc)
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)
    if not cfg.final:
        val_dataset = get_dataset(Partitions.val)
        trainer.add_validation(DataLoader(dataset=val_dataset, batch_size=batch_size))
    if not cfg.final and cfg_class.train.early_stopping.active:
        window = cfg_class.train.early_stopping.window
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc,
                                                                filter_fn=get_trailing_mean(window)))
    if checkpoint_every := cfg_user.checkpoint_every:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    trainer.train_until(cfg_class.train.n_epochs)
    trainer.save_checkpoint()
    final_test(store_outputs=True)
    outputs_probs = torch.cat(final_test.outputs_list)
    predictions = outputs_probs.argmax(dim=1)
    labels = torch.cat([data[1].label for data in test_loader.get_loader(inference=True)])
    misclassified_indices = [x.item() for x in (predictions != labels).nonzero().squeeze(1)]
    max_indices_to_log = 100
    misclassified_indices_str = str(misclassified_indices[:max_indices_to_log])
    if len(misclassified_indices) > max_indices_to_log:
        misclassified_indices_str += f" ... (and {len(misclassified_indices) - max_indices_to_log} more)"
    cf_matrix = ConfusionMatrix(task="multiclass", num_classes=cfg.data.dataset.n_classes)
    cf_matrix_tensor = cf_matrix(predictions, labels)  # This will be a torch.Tensor
    cf_matrix_numpy = cf_matrix_tensor.cpu().numpy()
    class_names = cfg.data.dataset.settings['select_classes']
    fig_cm = plot_confusion_matrix_heatmap(
        cf_matrix_numpy,
        class_names,
        title='Model Confusion Matrix'
    )
    try:
        tensorboard_tracker = TensorBoard.get_current()
    except TrackerNotActiveError:
        pass
    else:
        writer = tensorboard_tracker.writer
        writer.add_figure(f"{model.name}/{final_test.name}-Confusion Matrix", fig_cm)
        writer.add_text(
            "{model.name}/{final_test.name}- Misclassified Indices",
            f"Total misclassified samples: {len(misclassified_indices)}\nIndices: {misclassified_indices_str}",
            global_step=model.epoch
        )
        writer.close()


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up the experiment and launch the classifier training."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    resume = cfg.user.load_checkpoint != 0
    if not sys.gettrace():
        exp.trackers.register(Wandb(settings=wandb.Settings(project=cfg.project)))
        exp.trackers.register(HydraLink())
        exp.trackers.register(CSVDumper())
        exp.trackers.register(TensorBoard())
        engine_path = cfg.user.path.exp_par_dir / 'metrics.db'
        cfg.user.path.exp_par_dir.mkdir(exist_ok=True)
        engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
        exp.trackers.register(SQLConnection(engine=engine))
    with exp.create_run(resume=resume):
        train_classifier()
    return


if __name__ == "__main__":
    main()
