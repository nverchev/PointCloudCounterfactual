from drytorch import Model, Trainer

from src.data.dataset import get_dataset
from src.data.split import Partitions
from src.module import get_classifier
from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.train import get_classification_loss, get_learning_schema
from src.train.evaluations import ClassificationEvaluation
from src.train.hooks import register_checkpointing, register_early_stopping
from src.train.loaders import get_loaders
from src.utils.parallel import DistributedWorker


def train_classifier() -> None:
    """Train a DGCNN classifier and test its performance."""
    cfg = Experiment.get_config()
    cfg_class = cfg.classifier
    cfg_user = cfg.user
    module = get_classifier()
    model = Model(module, name=cfg_class.model.name, device=cfg_user.device)
    train_loader, test_loader = get_loaders(
        batch_size=cfg_class.train.batch_size_per_device, n_workers=cfg_user.n_workers
    )
    loss_calc = get_classification_loss()
    learning_schema = get_learning_schema(cfg.classifier)
    trainer = Trainer(model, loader=train_loader, loss=loss_calc, learning_schema=learning_schema)
    final_test = ClassificationEvaluation(
        model,
        loader=test_loader,
        metric=loss_calc,
        n_classes=cfg.data.dataset.n_classes,
        class_names=get_dataset(Partitions.test).class_names,
    )
    if cfg_user.load_checkpoint:
        trainer.load_checkpoint(cfg_user.load_checkpoint)

    if not cfg.final:
        trainer.add_validation(test_loader)
        if cfg_class.train.early_stopping.active:
            cfg_early = cfg_class.train.early_stopping
            register_early_stopping(trainer, window=cfg_early.window, patience=cfg_early.patience)

    register_checkpointing(trainer, cfg_user.checkpoint_every)
    trainer.train_until(cfg_class.train.n_epochs)
    trainer.save_checkpoint()
    final_test(store_outputs=True)
    final_test.log_results(trainer)
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
