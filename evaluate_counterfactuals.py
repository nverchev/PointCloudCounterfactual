from typing import cast

from dry_torch import DataLoader, Model, Test
from dry_torch.metrics import Metric, repr_metrics
import torch

from torch.utils.data import Subset

from src.autoencoder import VQVAE
from src.metrics_and_losses import get_classification_loss
from src.config_options import MainExperiment, ExperimentClassifier, ExperimentAE, hydra_main, ConfigAll
from src.datasets import get_dataset, Partitions, CounterfactualDataset, ReconstructedDataset
from src.classifier import DGCNN


def evaluate_counterfactuals(vqvae: VQVAE) -> None:
    cfg = MainExperiment.get_config()
    cfg_class = cfg.classifier
    cfg_user = cfg.user
    dgcnn = DGCNN()
    num_classes = dgcnn.num_classes
    classifier = Model(dgcnn, name=cfg_class.model.name, device=cfg_user.device)
    classifier.load_state()
    batch_size = cfg_class.train.batch_size
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)

    loss = get_classification_loss()
    test_original = Test(classifier,
                         name='ClassificationOriginal',
                         loader=DataLoader(dataset=test_dataset, batch_size=batch_size),
                         metric=loss)
    test_original(store_outputs=True)
    with torch.inference_mode():
        labels = torch.cat([data[1].label for data in test_original.loader])

    print('label distribution:', {'count_' + str(i): (labels == i).sum().item() for i in range(num_classes)})
    reconstructed_dataset = ReconstructedDataset(dataset=test_dataset, autoencoder=vqvae, classifier=classifier)
    reconstructed_loader = DataLoader(dataset=reconstructed_dataset,
                                      batch_size=batch_size,
                                      pin_memory=False)
    test_reconstructed = Test(classifier,
                              name='ClassificationReconstructed',
                              loader=reconstructed_loader,
                              metric=loss)
    test_reconstructed()

    metrics = list[Metric]()
    for j in range(num_classes):
        counterfactual_dataset = CounterfactualDataset(test_dataset,
                                                       vqvae,
                                                       classifier=classifier,
                                                       num_classes=dgcnn.num_classes,
                                                       target_label=j,
                                                       target_value=cfg.user.counterfactual_value)
        counterfactual_loader = DataLoader(dataset=counterfactual_dataset,
                                           batch_size=batch_size,
                                           pin_memory=False)
        test = Test(classifier,
                    name=f'Counterfeit_to_{j}',
                    loader=counterfactual_loader,
                    metric=loss)
        test()
        metrics.append(cast(Metric, test.objective))

    overall_metric = metrics[0].copy()
    for metric in metrics[1:]:
        overall_metric.merge_state(metric)
    print('Overall counterfeit success: ', repr_metrics(overall_metric))

    outputs_logits = torch.cat(test_original.outputs_list)
    predictions = outputs_logits.argmax(dim=1)

    misclassified_bool = predictions != labels
    misclassified_dataset = Subset(test_dataset, indices=list(misclassified_bool.nonzero()))

    misclassified_reconstruction = ReconstructedDataset(dataset=misclassified_dataset,
                                                        autoencoder=vqvae,
                                                        classifier=classifier)
    misclassified_reconstruction_loader = DataLoader(misclassified_reconstruction,
                                                     batch_size=batch_size,
                                                     pin_memory=False)
    test_misclassified = Test(classifier,
                              name='MisclassifiedReconstructed',
                              loader=misclassified_reconstruction_loader,
                              metric=loss)
    test_misclassified()

    metrics = list[Metric]()
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            else:
                i_instead_of_j = torch.logical_and(predictions == i, labels == j)
                if torch.any(i_instead_of_j):
                    counterfactual_indices = list(i_instead_of_j.nonzero())
                    i_instead_of_j_dataset = Subset(test_dataset, indices=counterfactual_indices)
                    counterfactual_dataset = CounterfactualDataset(i_instead_of_j_dataset,
                                                                   vqvae,
                                                                   classifier=classifier,
                                                                   num_classes=dgcnn.num_classes,
                                                                   target_label='original',
                                                                   target_value=cfg.user.counterfactual_value)
                    counterfactual_loader = DataLoader(dataset=counterfactual_dataset,
                                                       batch_size=batch_size,
                                                       pin_memory=False)
                    test = Test(classifier,
                                name=f'{i}_to_{j}',
                                loader=counterfactual_loader,
                                metric=loss)
                    test(store_outputs=True)

                    metrics.append(cast(Metric, test.objective))

                    overall_metric = metrics[0].copy()
                    for metric in metrics[1:]:
                        overall_metric.merge_state(metric)
                    print('Overall misclassified counterfeit success: ', repr_metrics(overall_metric))
    return


@hydra_main
def main(cfg: ConfigAll) -> None:
    parent_experiment = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    exp_classifier = ExperimentClassifier(cfg.classifier.name, config=cfg.classifier)
    exp_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    parent_experiment.register_child(exp_classifier)
    parent_experiment.register_child(exp_ae)
    with exp_ae:
        vqvae = VQVAE()
        autoencoder = Model(vqvae, name=cfg.autoencoder.model.name, device=cfg.user.device)
        autoencoder.load_state()
    with exp_classifier:
        evaluate_counterfactuals(vqvae)
    return


if __name__ == "__main__":
    main()
