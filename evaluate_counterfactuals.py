"""Evaluate counterfactuals."""

from typing import cast

import torch
from torch.utils.data import Subset

from drytorch import DataLoader, Model, Test
from drytorch.core import protocols as p
from drytorch.lib.objectives import Metric, repr_metrics

from src.autoencoder import CounterfactualVQVAE
from src.classifier import DGCNN
from src.config_options import Experiment, hydra_main, ConfigAll
from src.data_structures import Inputs
from src.datasets import get_dataset, Partitions, CounterfactualDataset, ReconstructedDatasetWithLogits
from src.metrics_and_losses import get_classification_loss


def get_label_distribution(test_loader: p.LoaderProtocol, num_classes: int) -> torch.Tensor:
    """Extract labels from the test loader and compute distribution."""
    with torch.inference_mode():  # using inference mode to prevent augmentation
        labels = torch.cat([data[1].label for data in test_loader])

    distribution = {'count_' + str(i): int((labels == i).sum().item()) for i in range(num_classes)}
    print('label distribution:', distribution)
    return labels


def evaluate_original_performance(
        classifier: p.ModelProtocol, test_dataset: torch.utils.data.Dataset, batch_size: int
) -> Test:
    """Evaluate classifier performance on original test data."""
    loss = get_classification_loss()
    test_original = Test(
        classifier,
        name='ClassificationOriginal',
        loader=DataLoader(dataset=test_dataset, batch_size=batch_size),
        metric=loss
    )
    test_original(store_outputs=True)
    return test_original


def evaluate_reconstructed_performance(
        classifier: Model[Inputs, torch.Tensor],
        test_dataset: torch.utils.data.Dataset,
        vqvae: CounterfactualVQVAE,
        batch_size: int) -> None:
    """Evaluate classifier performance on reconstructed test data."""
    reconstructed_dataset = ReconstructedDatasetWithLogits(
        dataset=test_dataset,
        autoencoder=vqvae,
        classifier=classifier,
    )
    reconstructed_loader = DataLoader(
        dataset=reconstructed_dataset,
        batch_size=batch_size,
        pin_memory=False
    )

    loss = get_classification_loss()
    test_reconstructed = Test(
        classifier,
        name='ClassificationReconstructed',
        loader=reconstructed_loader,
        metric=loss
    )
    test_reconstructed()
    return


def evaluate_counterfactual_performance(
        classifier: Model[Inputs, torch.Tensor],
        test_dataset: torch.utils.data.Dataset,
        vqvae: CounterfactualVQVAE,
        num_classes: int,
        batch_size: int,
        target_value: float) -> None:
    """Evaluate counterfactual generation performance across all classes."""
    metrics = []
    loss = get_classification_loss()

    for j in range(num_classes):
        counterfactual_dataset = CounterfactualDataset(
            test_dataset,
            vqvae,
            classifier,
            num_classes=num_classes,
            target_label=j,
            target_value=target_value
        )
        counterfactual_loader = DataLoader(
            dataset=counterfactual_dataset,
            batch_size=batch_size,
            pin_memory=False
        )

        test = Test(
            classifier,
            name=f'Counterfeit_to_{j}',
            loader=counterfactual_loader,
            metric=loss
        )
        test()
        metrics.append(cast(Metric, test.objective))

    overall_metric = compute_overall_metric(metrics)
    if overall_metric:
        print('Overall counterfeit success: ')
        print_metrics(overall_metric)

    return


def evaluate_misclassified_samples(
        classifier: Model[Inputs, torch.Tensor],
        test_dataset: torch.utils.data.Dataset,
        vqvae: CounterfactualVQVAE,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        batch_size) -> Test:
    """Evaluate reconstruction performance on misclassified samples."""
    misclassified_bool = predictions != labels
    misclassified_dataset = Subset(test_dataset, indices=list(misclassified_bool.nonzero()))
    misclassified_reconstruction = ReconstructedDatasetWithLogits(
        dataset=misclassified_dataset,
        autoencoder=vqvae,
        classifier=classifier
    )
    misclassified_reconstruction_loader = DataLoader(
        misclassified_reconstruction,
        batch_size=batch_size,
        pin_memory=False
    )

    loss = get_classification_loss()
    test_misclassified = Test(
        classifier,
        name='MisclassifiedReconstructed',
        loader=misclassified_reconstruction_loader,
        metric=loss
    )
    test_misclassified()
    return test_misclassified


def evaluate_class_specific_counterfactuals(
        classifier: Model[Inputs, torch.Tensor],
        test_dataset: torch.utils.data.Dataset,
        vqvae: CounterfactualVQVAE,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        num_classes: int,
        batch_size: int,
        target_value: float
) -> None:
    """Evaluate counterfactual performance for specific class transitions."""
    metrics = []
    loss = get_classification_loss()

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue

            i_instead_of_j = torch.logical_and(predictions == i, labels == j)
            if not torch.any(i_instead_of_j):
                continue

            counterfactual_indices = list(i_instead_of_j.nonzero())
            i_instead_of_j_dataset = Subset(test_dataset, indices=counterfactual_indices)

            counterfactual_dataset = CounterfactualDataset(
                i_instead_of_j_dataset,
                vqvae,
                classifier=classifier,
                num_classes=num_classes,
                target_label=j,
                target_value=target_value
            )
            counterfactual_loader = DataLoader(
                dataset=counterfactual_dataset,
                batch_size=batch_size,
                pin_memory=False
            )

            test = Test(
                classifier,
                name=f'{i}_to_{j}',
                loader=counterfactual_loader,
                metric=loss
            )
            test(store_outputs=True)
            metrics.append(cast(Metric, test.objective))

    overall_metric = compute_overall_metric(metrics)
    if overall_metric:
        print('Overall misclassified counterfeit success:')
        print_metrics(overall_metric)

    return


def print_metrics(metrics: p.ObjectiveProtocol) -> None:
    """Print metrics."""
    for metric_name, metric_value in repr_metrics(metrics).items():
        print(f'{metric_name}: {round(metric_value, 3)}')


def compute_overall_metric(metrics: list[Metric] | None = None) -> Metric | None:
    """Compute overall metric by merging individual metrics."""
    if not metrics:
        return None

    overall_metric = metrics[0].copy()
    for metric in metrics[1:]:
        overall_metric.merge_state(metric)
    return overall_metric


def evaluate_counterfactuals(classifier: Model[Inputs, torch.Tensor], vqvae: CounterfactualVQVAE) -> None:
    """Evaluate the counterfactuals according to different metrics."""
    cfg = Experiment.get_config()
    num_classes = cfg.data.dataset.n_classes
    batch_size = cfg.classifier.train.batch_size
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)

    test_original = evaluate_original_performance(classifier, test_dataset, batch_size)
    labels = get_label_distribution(test_original.loader, num_classes)

    evaluate_reconstructed_performance(classifier, test_dataset, vqvae, batch_size)

    evaluate_counterfactual_performance(
        classifier, test_dataset, vqvae, num_classes, batch_size, cfg.user.counterfactual_value
    )

    outputs_logits = torch.cat(test_original.outputs_list)
    predictions = outputs_logits.argmax(dim=1)

    evaluate_misclassified_samples(classifier, test_dataset, vqvae, labels, predictions, batch_size)

    evaluate_class_specific_counterfactuals(
        classifier, test_dataset, vqvae, labels, predictions,
        num_classes, batch_size, cfg.user.counterfactual_value
    )


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up the experiment and launch the counterfactual evaluation."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        dgcnn_module = DGCNN()
        classifier = Model(dgcnn_module, name=cfg.classifier.architecture.name, device=cfg.user.device)
        classifier.load_state()
        vqvae = CounterfactualVQVAE()
        autoencoder = Model(vqvae, name=cfg.autoencoder.architecture.name, device=cfg.user.device)
        autoencoder.load_state()
        evaluate_counterfactuals(classifier, vqvae)


if __name__ == "__main__":
    main()
