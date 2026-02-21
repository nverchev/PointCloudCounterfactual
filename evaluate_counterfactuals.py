"""Evaluate counterfactual generation performance."""

import logging
from typing import cast

import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from drytorch import DataLoader, Model, Test
from drytorch.core import protocols as p
from drytorch.lib.objectives import Metric, compute_metrics

from src.module import BaseClassifier, CounterfactualVAE, get_classifier
from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.data.processed import CounterfactualDataset, ReconstructedEvaluatedDataset
from src.data import Inputs, Targets, Partitions, get_dataset
from src.train.metrics_and_losses import get_classification_loss


def get_label_distribution(test_loader: p.LoaderProtocol[tuple[Inputs, Targets]], class_names: list[str]) -> Tensor:
    """Extract labels from the test loader and compute distribution."""
    with torch.inference_mode():  # using inference mode to prevent augmentation
        labels = torch.cat([batch_data[1].label for batch_data in test_loader])

    counts = [(labels == i).sum().item() for i in range(len(class_names))]
    dist_str = ' | '.join(f'{name}: {int(count)}' for name, count in zip(class_names, counts, strict=True))
    logging.info('Label distribution: %s', dist_str)
    return labels


def evaluate_original_performance(
    classifier_model: p.ModelProtocol[Inputs, Tensor],
    test_loader: DataLoader[tuple[Inputs, Targets]],
) -> Test[Inputs, Targets, Tensor]:
    """Evaluate classifier performance on original test data."""
    loss = get_classification_loss()
    test_original = Test(classifier_model, name='ClassificationOriginal', loader=test_loader, metric=loss)
    test_original(store_outputs=True)
    return test_original


def evaluate_reconstructed_performance(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: CounterfactualVAE,
    classifier: BaseClassifier,
    batch_size: int,
) -> None:
    """Evaluate classifier performance on reconstructed test data."""
    reconstructed_dataset = ReconstructedEvaluatedDataset(
        dataset=test_dataset,
        autoencoder=vae,
        classifier=classifier,
    )
    reconstructed_loader = DataLoader(dataset=reconstructed_dataset, batch_size=batch_size, pin_memory=False)
    loss = get_classification_loss()
    test_reconstructed = Test(
        classifier_model, name='ClassificationReconstructed', loader=reconstructed_loader, metric=loss
    )
    test_reconstructed()
    return


def evaluate_counterfactual_performance(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: CounterfactualVAE,
    classifier: BaseClassifier,
    class_names: list[str],
    batch_size: int,
    target_value: float,
) -> None:
    """Evaluate counterfactual generation performance across all classes."""
    metrics: list[Metric[Tensor, Targets]] = []
    loss = get_classification_loss()

    for j, class_name in enumerate(class_names):
        counterfactual_dataset = CounterfactualDataset(
            test_dataset, vae, classifier, target_dim=j, target_value=target_value
        )
        counterfactual_loader = DataLoader(dataset=counterfactual_dataset, batch_size=batch_size, pin_memory=False)
        test = Test(classifier_model, name=f'Counterfeit_to_{class_name}', loader=counterfactual_loader, metric=loss)
        test()
        metrics.append(cast(Metric[Tensor, Targets], test.objective))

    overall_metric = compute_overall_metric(metrics)
    if overall_metric:
        logging.info('Overall counterfeit success:')
        log_metrics(overall_metric)

    return


def evaluate_misclassified_samples(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: CounterfactualVAE,
    classifier: BaseClassifier,
    labels: Tensor,
    predictions: Tensor,
    batch_size: int,
) -> None:
    """Evaluate reconstruction performance on misclassified samples."""
    misclassified_bool = predictions != labels
    misclassified_dataset = Subset(test_dataset, indices=list(map(int, misclassified_bool.nonzero())))
    misclassified_reconstruction = ReconstructedEvaluatedDataset(
        dataset=misclassified_dataset, autoencoder=vae, classifier=classifier
    )
    misclassified_reconstruction_loader = DataLoader(
        misclassified_reconstruction, batch_size=batch_size, pin_memory=False
    )
    loss = get_classification_loss()
    test_misclassified = Test(
        classifier_model, name='MisclassifiedReconstructed', loader=misclassified_reconstruction_loader, metric=loss
    )
    test_misclassified()
    return


def evaluate_class_specific_counterfactuals(
    classifier_model: Model[Inputs, Tensor],
    test_dataset: Dataset[tuple[Inputs, Targets]],
    vae: CounterfactualVAE,
    classifier: BaseClassifier,
    labels: Tensor,
    predictions: Tensor,
    class_names: list[str],
    batch_size: int,
    target_value: float,
) -> None:
    """Evaluate counterfactual performance for specific class transitions."""
    metrics: list[Metric[Tensor, Targets]] = []
    loss = get_classification_loss()

    for i, name_i in enumerate(class_names):
        for j, name_j in enumerate(class_names):
            if i == j:
                continue

            i_instead_of_j = torch.logical_and(predictions == i, labels == j)
            if not torch.any(i_instead_of_j):
                continue

            counterfactual_indices = list(i_instead_of_j.nonzero())
            i_instead_of_j_dataset = Subset(test_dataset, indices=list(map(int, counterfactual_indices)))
            counterfactual_dataset = CounterfactualDataset(
                i_instead_of_j_dataset,
                vae,
                classifier=classifier,
                target_dim=j,
                target_value=target_value,
            )
            counterfactual_loader = DataLoader(dataset=counterfactual_dataset, batch_size=batch_size, pin_memory=False)
            test = Test(classifier_model, name=f'{name_i}_to_{name_j}', loader=counterfactual_loader, metric=loss)
            test(store_outputs=True)
            metrics.append(cast(Metric[Tensor, Targets], test.objective))

    overall_metric = compute_overall_metric(metrics)
    if overall_metric:
        logging.info('Overall misclassified counterfeit success:')
        log_metrics(overall_metric)

    return


def log_metrics(metrics: p.ObjectiveProtocol[Tensor, Targets], prefix: str = '') -> None:
    """Log metrics."""
    results = compute_metrics(metrics)
    metrics_str = ' | '.join(f'{k}: {v:.3f}' for k, v in results.items())
    logging.info('%s%s', prefix, metrics_str)

    return


def compute_overall_metric(metrics: list[Metric[Tensor, Targets]]) -> Metric[Tensor, Targets]:
    """Compute overall metric by merging individual metrics."""
    if not metrics:
        raise ValueError('No metrics to merge.')

    overall_metric = metrics[0].copy()
    for metric in metrics[1:]:
        overall_metric.merge_state(metric)

    return overall_metric


@torch.inference_mode()
def evaluate_counterfactuals(
    classifier_model: Model[Inputs, Tensor], vae: CounterfactualVAE, classifier: BaseClassifier
) -> None:
    """Evaluate the counterfactuals according to different metrics."""
    cfg = Experiment.get_config()
    batch_size = cfg.classifier.train.batch_size_per_device
    counterfactual_value = cfg.autoencoder.objective.counterfactual_value

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    class_names = test_dataset.classes
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    test_labels = get_label_distribution(test_loader, class_names)
    test_original = evaluate_original_performance(classifier_model, test_loader)
    evaluate_reconstructed_performance(classifier_model, test_dataset, vae, classifier, batch_size)
    evaluate_counterfactual_performance(
        classifier_model, test_dataset, vae, classifier, class_names, batch_size, counterfactual_value
    )
    outputs_logits = torch.cat(test_original.outputs_list)
    predictions = outputs_logits.argmax(dim=1)
    evaluate_misclassified_samples(
        classifier_model, test_dataset, vae, classifier, test_labels, predictions, batch_size
    )
    evaluate_class_specific_counterfactuals(
        classifier_model,
        test_dataset,
        vae,
        classifier,
        test_labels,
        predictions,
        class_names,
        batch_size,
        counterfactual_value,
    )


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the counterfactual evaluation."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True):
        classifier = get_classifier()
        classifier_model = Model(classifier, name=cfg.classifier.model.name, device=cfg.user.device)
        classifier_model.load_state()
        vae = CounterfactualVAE()
        autoencoder_model = Model(vae, name=cfg.autoencoder.model.name, device=cfg.user.device)
        autoencoder_model.load_state()
        evaluate_counterfactuals(classifier_model, vae, classifier)

    return


if __name__ == '__main__':
    main()
