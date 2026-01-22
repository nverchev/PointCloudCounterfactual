"""Module for training and testing a DGCNN classifier on point cloud data."""

import torch
import torch.distributed as dist

from torchmetrics import ConfusionMatrix

from drytorch import DataLoader, Model, Test, Trainer
from drytorch.core.exceptions import TrackerNotUsedError
from drytorch.lib.hooks import EarlyStoppingCallback, call_every, saving_hook
from drytorch.utils.average import get_trailing_mean

from src.module import get_classifier
from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.data import get_datasets
from src.train import get_classification_loss, get_learning_schema
from src.utils.parallel import DistributedWorker
from src.utils.visualization import plot_confusion_matrix_heatmap


def train_classifier() -> None:
    """Train a DGCNN classifier and test its performance."""
    cfg = Experiment.get_config()
    cfg_class = cfg.classifier
    cfg_user = cfg.user
    module = get_classifier()
    model = Model(module, name=cfg_class.model.name, device=cfg_user.device)
    train_dataset, test_dataset = get_datasets()  # test is validation unless final=True
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg_class.train.batch_size_per_device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg_class.train.batch_size_per_device)
    loss_calc = get_classification_loss()
    learning_schema = get_learning_schema(cfg.classifier)
    trainer = Trainer(model, loader=train_loader, loss=loss_calc, learning_schema=learning_schema)
    final_test = Test(model, loader=test_loader, metric=loss_calc)
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if not cfg.final:
        trainer.add_validation(test_loader)

    if not cfg.final and cfg_class.train.early_stopping.active:
        window = cfg_class.train.early_stopping.window
        trainer.post_epoch_hooks.register(EarlyStoppingCallback(metric=loss_calc, filter_fn=get_trailing_mean(window)))
    if checkpoint_every := cfg_user.checkpoint_every:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))

    trainer.train_until(cfg_class.train.n_epochs)
    trainer.save_checkpoint()
    final_test(store_outputs=True)
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    outputs_probs = torch.cat(final_test.outputs_list)
    predictions = outputs_probs.argmax(dim=1)
    labels = torch.cat([data[1].label for data in test_loader.get_loader(inference=True)])
    misclassified_indices = [x.item() for x in (predictions != labels).nonzero().squeeze(1)]
    max_indices_to_log = 100
    misclassified_indices_str = str(misclassified_indices[:max_indices_to_log])
    if len(misclassified_indices) > max_indices_to_log:
        misclassified_indices_str += f' ... (and {len(misclassified_indices) - max_indices_to_log} more)'

    cf_matrix = ConfusionMatrix(task='multiclass', num_classes=cfg.data.dataset.n_classes)
    cf_matrix_tensor = cf_matrix(predictions, labels)  # This will be a torch.Tensor
    cf_matrix_numpy = cf_matrix_tensor.cpu().numpy()
    names = cfg.data.dataset.settings['select_classes']
    fig_cm = plot_confusion_matrix_heatmap(cf_matrix_numpy, names, title='Model Confusion Matrix')
    try:
        from drytorch.trackers.tensorboard import TensorBoard

        tensorboard_tracker = TensorBoard.get_current()
    except TrackerNotUsedError:
        pass
    except (ModuleNotFoundError, ImportError):
        print(f'Confusion Matrix for classes {names}')
        print(cf_matrix_numpy)
        print(f'Misclassified indices: {misclassified_indices_str}')
    else:
        writer = tensorboard_tracker.writer
        if fig_cm is not None:
            writer.add_figure(f'{model.name}/{final_test.name}-Confusion Matrix', fig_cm)

        writer.add_text(
            '{model.name}/{final_test.name}- Misclassified Indices',
            f'Total misclassified samples: {len(misclassified_indices)}\nIndices: {misclassified_indices_str}',
            global_step=model.epoch,
        )
        writer.close()

    return


def setup_and_train(cfg: AllConfig) -> None:
    """Set up the experiment and launch the classifier training."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    resume = cfg.user.load_checkpoint != 0
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume):
        train_classifier()

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
