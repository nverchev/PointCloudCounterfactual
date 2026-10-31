"""Train the inner VAE on the codes of the outer VQ-VAE, conditioned on the classifier.

Usage::

    uv run train_w_autoencoder.py --config-name modelnet_dt final=true
"""

from drytorch import Trainer

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import WDataset, get_splits
from src.lib import get_learning_schema, get_w_loader
from src.lib.hooks import register_checkpointing, register_early_stopping
from src.lib.models import get_w_model, load_classifier_model, load_vqvae_model
from src.lib.objectives import get_w_loss
from src.trackers import get_trackers


def train_w_autoencoder() -> None:
    """Train the inner VAE and save it, leaving the VQ-VAE and the classifier untouched."""
    cfg = Experiment.get_config()
    cfg_phase = cfg.w_autoencoding
    cfg_train = cfg_phase.train
    classifier = load_classifier_model()
    vqvae = load_vqvae_model()
    train_split, eval_split = get_splits()
    builder = Trainer.builder(
        get_w_model(),
        learning_schema=get_learning_schema(cfg_phase),
        loader=get_w_loader(WDataset(train_split, vqvae, classifier)),
        loss=get_w_loss(vqvae.module),
    )
    if not cfg.final:
        builder.add_validation(get_w_loader(WDataset(eval_split, vqvae, classifier)))
        if cfg_train.early_stopping.active:
            cfg_early = cfg_train.early_stopping
            register_early_stopping(builder, patience=cfg_early.patience, window=cfg_early.window)

    register_checkpointing(builder, checkpoint_every=cfg.user.checkpoint_every)
    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    trainer.save_checkpoint()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and train the inner VAE."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    # the weights of the earlier phases live in the run this one continues
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_w_autoencoder()

    return


if __name__ == '__main__':
    main()
