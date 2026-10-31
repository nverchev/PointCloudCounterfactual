"""Test the reconstructions of the outer VQ-VAE.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_autoencoder.py --config-name modelnet_dt final=true
"""

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import get_eval_split
from src.lib import get_cloud_loader
from src.lib.models import load_vqvae_model
from src.lib.objectives import get_reconstruction_metrics
from src.trackers import get_trackers


def evaluate_autoencoder() -> None:
    """Test the reconstructions of the trained VQ-VAE."""
    batch_size = Experiment.get_config().autoencoding.train.batch_size
    loader = get_cloud_loader(get_eval_split(), batch_size=batch_size)
    Test(load_vqvae_model(), loader=loader, metric=get_reconstruction_metrics(), name='Reconstruction')()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the VQ-VAE."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_autoencoder()

    return


if __name__ == '__main__':
    main()
