"""Render clouds of the evaluation partition with their counterfactuals, in a folder for each sample.

For each cloud, the script renders the original, its double reconstruction, and its counterfactuals
at each strength, towards every other class. Without configured indices, it renders the
misclassified clouds, towards their correct class. The probabilities the classifier assigns are
logged next to the name of each image.

Usage::

    uv run visualize_counterfactuals.py --config-name modelnet_dt final=true
    uv run visualize_counterfactuals.py --config-name modelnet_bbcv final=true "user.plot.sample_indices=[0,5,7]"
"""

import logging

import torch

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.data import CloudSplit, get_databuilder, get_eval_split, get_partitions
from src.lib.models import ClassifierModel, load_classifier_model, load_counterfactual_model
from src.trackers import get_trackers
from src.utils.counterfactuals import interpolate_evaluation
from src.utils.visualization import render_cloud


def visualize_counterfactuals() -> None:
    """Render the configured clouds, or the misclassified ones.

    Side Effects:
        Saves the images in the ``images`` folder of the experiment.
    """
    cfg = Experiment.get_config()
    class_names = get_databuilder().class_names
    batch_size = cfg.evaluation.batch_size
    classifier = load_classifier_model()
    model = load_counterfactual_model()
    eval_split = get_eval_split()
    indices = list(cfg.user.plot.sample_indices) or _get_misclassified(classifier, eval_split)
    if not indices:
        return

    _, (inputs, targets) = loading.take_from_dataset(eval_split, device=torch.device('cpu'), indices=indices)
    evaluations = classifier.evaluate_split(inputs.cloud, batch_size=batch_size)
    save_dir = cfg.par_dir / 'images' / cfg.name / get_partitions()[1].name
    for position, index in enumerate(indices):
        cloud = inputs.cloud[position : position + 1]
        evaluation = evaluations[position : position + 1]
        label = int(targets.label[position])
        logging.info(f'Cloud {index}, labelled {class_names[label]}.')
        clouds = {
            'original': cloud,
            'reconstruction': model.generate(cloud, evaluation, batch_size=1),
        }
        target_classes = range(len(class_names)) if cfg.user.plot.sample_indices else [label]
        for target in target_classes:
            for strength in cfg.evaluation.strengths:
                condition = interpolate_evaluation(evaluation, torch.tensor([target]), strength=strength)
                name = f'to_{class_names[target]}_{strength:g}'
                clouds[name] = model.generate(cloud, condition, batch_size=1)

        for name, generated in clouds.items():
            logging.info(f'{name}: {_describe(classifier, generated, class_names)}')
            render_cloud(generated[0], name=name, save_dir=save_dir / f'sample_{index}', show=cfg.user.plot.show)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and render the counterfactuals."""
    get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        visualize_counterfactuals()

    return


def _get_misclassified(classifier: ClassifierModel, split: CloudSplit) -> list[int]:
    batch_size = Experiment.get_config().evaluation.batch_size
    predictions = classifier.predict_split(torch.from_numpy(split.clouds), batch_size=batch_size)
    return (predictions != torch.from_numpy(split.labels)).nonzero().flatten().tolist()


def _describe(classifier: ClassifierModel, cloud: torch.Tensor, class_names: tuple[str, ...]) -> str:
    probabilities = torch.softmax(classifier.logits(cloud.to(classifier.device)), dim=1)[0].tolist()
    return ', '.join(f'{name} {probability:.2f}' for name, probability in zip(class_names, probabilities, strict=True))


if __name__ == '__main__':
    main()
