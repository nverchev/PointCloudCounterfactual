"""Test the classifier under audit and publish its confusion matrix.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_classifier.py --config-name modelnet_dt final=true
"""

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import get_eval_split
from src.lib import get_cloud_loader
from src.lib.events import ConfusionMatrixEvent
from src.lib.models import load_classifier_model
from src.lib.objectives import AggregatedAccuracy
from src.trackers import get_trackers


def evaluate_classifier() -> None:
    """Test the trained classifier and publish its confusion matrix."""
    cfg = Experiment.get_config()
    model = load_classifier_model()
    accuracy = AggregatedAccuracy(cfg.data.dataset.n_classes)
    loader = get_cloud_loader(get_eval_split(), batch_size=cfg.classification.train.batch_size)
    test = Test(model, loader=loader, metric=accuracy, name='Accuracy')
    test()
    ConfusionMatrixEvent(
        source_name=test.name, model_name=model.name, epoch=model.epoch, matrix=accuracy.confusion_matrix
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the classifier."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_classifier()

    return


if __name__ == '__main__':
    main()
