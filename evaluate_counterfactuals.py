"""Measure the counterfactuals: the tables of the thesis and of the article.

For each counterfactual strength, a test gives the ``Success`` of the classifier on counterfactuals,
with the label set to the target class:

- ``Success_<strength>``: every cloud towards every class, the one already predicted included;
- ``Flip_<strength>``: every cloud towards every class other than the one predicted;
- ``Misclassified_<strength>``: the misclassified clouds towards their correct class.

``Original`` gives the accuracy of the classifier, ``Reconstruction`` the accuracy on the double
reconstructions, and ``NoSemantics`` the accuracy on counterfactuals targeting the same probability
for all the classes, which remove the semantics of the classification. The last two are scored on
the true labels.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_counterfactuals.py --config-name modelnet_dt final=true
"""

import torch

from drytorch import Test
from drytorch.lib import objectives

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import GeneratedSplit, Targets, get_eval_split
from src.lib import get_cloud_loader
from src.lib.events import ConfusionMatrixEvent
from src.lib.models import ClassifierModel, CounterfactualModel, load_classifier_model, load_counterfactual_model
from src.lib.objectives import AggregatedAccuracy, get_accuracy, get_success
from src.trackers import get_trackers
from src.utils.counterfactuals import interpolate_evaluation


def evaluate_counterfactuals() -> None:
    """Test the classifier on the original clouds, on their reconstructions and on the counterfactuals."""
    cfg = Experiment.get_config()
    n_classes = cfg.data.dataset.n_classes
    batch_size = cfg.evaluation.batch_size
    classifier = load_classifier_model()
    model = load_counterfactual_model()
    eval_split = get_eval_split()
    clouds = torch.from_numpy(eval_split.clouds)
    labels = torch.from_numpy(eval_split.labels)
    original = AggregatedAccuracy(n_classes)
    test = Test(
        classifier, loader=get_cloud_loader(eval_split, batch_size=batch_size), metric=original, name='Original'
    )
    test()
    ConfusionMatrixEvent(
        source_name=test.name, model_name=classifier.name, epoch=classifier.epoch, matrix=original.confusion_matrix
    )
    evaluations = classifier.evaluate_split(clouds, batch_size=batch_size)
    predictions = classifier.predict_split(clouds, batch_size=batch_size)
    _test_generated('Reconstruction', classifier, model, clouds, evaluations, labels, get_accuracy())
    uniform = torch.full_like(evaluations, 1 / n_classes)
    _test_generated('NoSemantics', classifier, model, clouds, uniform, labels, get_accuracy())
    n_clouds = len(clouds)
    all_targets = torch.arange(n_classes).repeat_interleave(n_clouds)
    all_clouds = clouds.repeat(n_classes, 1, 1)
    all_evaluations = evaluations.repeat(n_classes, 1)
    all_predictions = predictions.repeat(n_classes)
    wrong = predictions != labels
    for strength in cfg.evaluation.strengths:
        conditions = interpolate_evaluation(all_evaluations, all_targets, strength=strength)
        _test_success(f'Success_{strength:g}', classifier, model, all_clouds, conditions, all_targets)
        flips = all_targets != all_predictions
        _test_success(f'Flip_{strength:g}', classifier, model, all_clouds[flips], conditions[flips], all_targets[flips])
        if wrong.any():
            conditions = interpolate_evaluation(evaluations[wrong], labels[wrong], strength=strength)
            _test_success(f'Misclassified_{strength:g}', classifier, model, clouds[wrong], conditions, labels[wrong])

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the counterfactuals."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_counterfactuals()

    return


def _test_generated(
    name: str,
    classifier: ClassifierModel,
    model: CounterfactualModel,
    clouds: torch.Tensor,
    conditions: torch.Tensor,
    labels: torch.Tensor,
    metric: objectives.Metric[torch.Tensor, Targets],
) -> None:
    batch_size = Experiment.get_config().evaluation.batch_size
    split = GeneratedSplit(model.generate(clouds, conditions, batch_size=batch_size), labels)
    Test(classifier, loader=get_cloud_loader(split, batch_size=batch_size), metric=metric, name=name)()
    return


def _test_success(
    name: str,
    classifier: ClassifierModel,
    model: CounterfactualModel,
    clouds: torch.Tensor,
    conditions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    _test_generated(name, classifier, model, clouds, conditions, targets, get_success())
    return


if __name__ == '__main__':
    main()
