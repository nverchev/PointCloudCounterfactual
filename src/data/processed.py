"""Classes with Dataset that have been preprocessed for use with VAE models."""

import abc

from abc import abstractmethod
from collections.abc import Generator, Sized
from typing import Any, Generic, TypeVar, override

import torch

from torch.utils.data import Dataset

from drytorch import Model

from src.config import Experiment
from src.module.autoencoders import VAE, CounterfactualVAE, BaseVAE
from src.data.structures import Inputs, Outputs, Targets
from src.data.protocols import AbstractSingleton

V = TypeVar('V', bound=BaseVAE)


class ProcessedDataset(Generic[V], abc.ABC, metaclass=AbstractSingleton):
    """Base dataset for VAE models with common functionality."""

    max_batch: int = 64
    dataset: Dataset[tuple[Inputs, Targets]]
    dataset_len: int
    autoencoder: V

    def __init__(self, dataset: Dataset[tuple[Inputs, Targets]], autoencoder: V) -> None:
        self.dataset = dataset
        assert isinstance(dataset, Sized)
        self.dataset_len = len(dataset)
        self.autoencoder = autoencoder
        self.device: torch.device = next(autoencoder.parameters()).device
        return

    def __len__(self) -> int:
        return self.dataset_len

    def _get_data(self, index_list: list[int]) -> Generator[tuple[Inputs, torch.Tensor]]:
        """Common data loading logic."""
        with torch.inference_mode():
            batch_data = [self.dataset[i] for i in index_list]

        batched_cloud = torch.stack([data[0].cloud for data in batch_data]).to(self.device)
        batched_labels = torch.cat([data[1].label.unsqueeze(0) for data in batch_data]).to(self.device)

        # Also batch logits if they exist in Inputs, though ClassifiedDataset handles them usually
        # But this base method recreates Inputs from cloud stack

        for i in range(0, len(index_list), self.max_batch):
            batch_slice = slice(i, i + self.max_batch)
            cloud_batch = batched_cloud[batch_slice]

            # Create fresh Inputs batch
            batch_inputs = Inputs(cloud=cloud_batch)
            batch_labels = batched_labels[batch_slice]
            yield batch_inputs, batch_labels

    @abstractmethod
    def __getitems__(self, index_list: list[int]) -> list[Any]:
        """Subclasses must implement this method."""


class ClassifiedDataset(ProcessedDataset[V], Dataset[tuple[Inputs, Targets]]):
    """Dataset that evaluates inputs with a classifier and attaches logits."""

    classifier: Model[Inputs, torch.Tensor]

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: V,  # Kept for API consistency if needed, or could be Optional
        classifier: Model[Inputs, torch.Tensor],
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder)
        self.classifier = classifier
        return

    @torch.inference_mode()
    def _run_classifier(self, batch_inputs: Inputs) -> torch.Tensor:
        """Run classifier on batch inputs."""
        self.classifier.module.eval()
        return self.classifier(batch_inputs)

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_data: list[tuple[Inputs, Targets]] = []
        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_logits = self._run_classifier(batch_inputs)

            # Split batch back into individual items
            for i in range(len(batch_inputs.cloud)):
                cloud = batch_inputs.cloud[i]
                logit = batch_logits[i]
                label = batch_labels[i]

                # Re-wrap in Inputs/Targets with logits
                inp = Inputs(cloud=cloud, logits=logit)
                tgt = Targets(ref_cloud=cloud, label=label)  # Assuming ref_cloud is same as input
                batch_data.append((inp, tgt))

        return batch_data


class DoubleReconstructedDataset(ProcessedDataset[VAE], Dataset[tuple[Inputs, Targets]]):
    """Dataset with reconstructed inputs after double encoding."""

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data: list[tuple[Inputs, Targets]] = []
        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_ae_out = self._reconstruct(batch_inputs)
            recons = batch_ae_out.recon
            for recon, label in zip(recons, batch_labels, strict=True):
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))

        return batch_data

    @torch.inference_mode()
    def _reconstruct(self, inputs: Inputs) -> Outputs:
        """Run autoencoder reconstruction."""
        # Using forward (encode -> decode)
        return self.autoencoder(inputs)


class CounterfactualDataset(ProcessedDataset[CounterfactualVAE], Dataset[tuple[Inputs, Targets]]):
    """Dataset for counterfactual reconstruction with target conditioning."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVAE,
        classifier: Model[Inputs, torch.Tensor],
        target_dim: int,
        target_value: float = 1.0,
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder)
        cfg = Experiment.get_config()
        self.n_classes: int = cfg.data.dataset.n_classes
        self.target_dim: int = target_dim
        self.target_value: float = target_value
        self.classifier: Model[Inputs, torch.Tensor] = classifier
        return

    @torch.inference_mode()
    def _run_classifier(self, batch_inputs: Inputs) -> torch.Tensor:
        """Run classifier on batch inputs."""
        self.classifier.module.eval()
        return self.classifier(batch_inputs)

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data: list[tuple[Inputs, Targets]] = []
        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_logits = self._run_classifier(batch_inputs)
            batch_ae_out = self.counterfactual_data(batch_inputs, batch_logits, self.target_dim, self.target_value)
            recons = batch_ae_out.recon
            for recon, label in zip(recons, batch_labels, strict=True):
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))

        return batch_data

    @torch.inference_mode()
    def counterfactual_data(
        self,
        inputs: Inputs,
        batched_logits: torch.Tensor,
        target_dim: int,
        target_value: float = 1.0,
    ) -> Outputs:
        """Run autoencoder with counterfactual conditioning."""
        return self.autoencoder.generate_counterfactual(inputs, batched_logits, target_dim, target_value)


class BoundaryDataset(CounterfactualDataset):
    """Dataset for boundary reconstruction with neutral conditioning."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVAE,
        classifier: Model[Inputs, torch.Tensor],
        target_dim: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier,
            target_dim=target_dim,
            target_value=0,  # Neutral probability
        )
        return
