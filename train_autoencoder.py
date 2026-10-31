"""Train the outer VQ-VAE, which the classifier does not condition.

Usage::

    uv run train_autoencoder.py --config-name modelnet_dt final=true
"""

from drytorch import Trainer
from drytorch.lib import evaluations

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import get_databuilder, get_partitions
from src.lib import get_cloud_loader, get_learning_schema
from src.lib.hooks import register_checkpointing, register_codebook_revival, register_early_stopping
from src.lib.models import get_vqvae_model
from src.lib.objectives import get_autoencoder_loss
from src.trackers import get_trackers


def train_autoencoder() -> None:
    """Train the VQ-VAE and save it."""
    cfg = Experiment.get_config()
    cfg_phase = cfg.autoencoding
    cfg_train = cfg_phase.train
    batch_size = cfg_train.batch_size
    databuilder = get_databuilder()
    train_partition, eval_partition = get_partitions()
    model = get_vqvae_model()
    loss = get_autoencoder_loss()
    train_loader = get_cloud_loader(databuilder.get_split(train_partition, augment=True), batch_size=batch_size)
    builder = Trainer.builder(
        model,
        learning_schema=get_learning_schema(cfg_phase),
        loader=train_loader,
        loss=loss,
    )
    if not cfg.final:
        eval_split = databuilder.get_split(eval_partition, augment=False)
        builder.add_validation(get_cloud_loader(eval_split, batch_size=batch_size))
        if cfg_train.early_stopping.active:
            cfg_early = cfg_train.early_stopping
            register_early_stopping(builder, patience=cfg_early.patience, window=cfg_early.window)

    diagnostic_split = databuilder.get_split(train_partition, augment=False)
    diagnostic_loader = get_cloud_loader(diagnostic_split, batch_size=batch_size)
    register_codebook_revival(
        builder,
        diagnostic=evaluations.Evaluation(model, loader=diagnostic_loader, metric=loss, name='Diagnostic'),
        module=model.module,
        noise=cfg_train.revive_noise,
        revive_every=cfg_train.revive_every,
        final_epoch=cfg_train.n_epochs,
    )
    register_checkpointing(builder, checkpoint_every=cfg.user.checkpoint_every)
    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    trainer.save_checkpoint()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the VQ-VAE."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    # the classifier lives in the run this phase continues
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_autoencoder()

    return


if __name__ == '__main__':
    main()
