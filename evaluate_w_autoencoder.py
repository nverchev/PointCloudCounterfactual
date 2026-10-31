"""Test the inner VAE and the double reconstructions through it.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_w_autoencoder.py --config-name modelnet_dt final=true
"""

import torch

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import EvaluatedSplit, WDataset, get_eval_split
from src.lib import get_cloud_loader, get_w_loader
from src.lib.models import get_counterfactual_model, load_classifier_model, load_vqvae_model, load_w_model
from src.lib.objectives import get_reconstruction_metrics, get_w_loss
from src.trackers import get_trackers


def evaluate_w_autoencoder() -> None:
    """Test the loss of the inner VAE, then the clouds decoded from the codes it chooses."""
    cfg = Experiment.get_config()
    classifier = load_classifier_model()
    vqvae = load_vqvae_model()
    w_model = load_w_model()
    eval_split = get_eval_split()
    w_loader = get_w_loader(WDataset(eval_split, vqvae, classifier))
    Test(w_model, loader=w_loader, metric=get_w_loss(vqvae.module), name='InnerVAE')()
    batch_size = cfg.evaluation.batch_size
    conditions = classifier.evaluate_split(torch.from_numpy(eval_split.clouds), batch_size=batch_size)
    loader = get_cloud_loader(EvaluatedSplit(eval_split, conditions), batch_size=batch_size)
    model = get_counterfactual_model(vqvae, w_model)
    Test(model, loader=loader, metric=get_reconstruction_metrics(), name='DoubleReconstruction')()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the inner VAE."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_w_autoencoder()

    return


if __name__ == '__main__':
    main()
