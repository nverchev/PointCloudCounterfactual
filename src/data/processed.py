"""Classes with Dataset that have been preprocessed for use with VQVAE models."""

import abc
import itertools

from abc import abstractmethod
from collections.abc import Generator, Sized
from typing import Any, Generic, TypeVar, override

import torch

from torch.utils.data import Dataset

from drytorch import Model

from src.config import Experiment
from src.module.autoencoders import BaseVQVAE, CounterfactualVQVAE, VQVAE
from src.module.w_autoencoders import BaseWAutoEncoder
from src.data.structures import Inputs, Outputs, Targets, WInputs, WTargets
from src.data.protocols import AbstractSingleton

VQ = TypeVar('VQ', bound=BaseVQVAE[BaseWAutoEncoder])


class ProcessedDataset(Generic[VQ], abc.ABC, metaclass=AbstractSingleton):
    """Base dataset for VQVAE models with common functionality."""

    max_batch: int = 64
    dataset: Dataset[tuple[Inputs, Targets]]
    dataset_len: int
    autoencoder: VQ

    def __init__(self, dataset: Dataset[tuple[Inputs, Targets]], autoencoder: VQ) -> None:
        self.dataset = dataset
        assert isinstance(dataset, Sized)
        self.dataset_len = len(dataset)
        self.autoencoder = autoencoder
        self.device: torch.device = next(autoencoder.parameters()).device

    def __len__(self) -> int:
        return self.dataset_len

    def _get_data(self, index_list: list[int]) -> Generator[tuple[Inputs, torch.Tensor]]:
        """Common data loading logic."""
        dataset_batch = [self.dataset[i] for i in index_list]
        batched_cloud = torch.stack([data[0].cloud for data in dataset_batch]).to(self.device)
        batched_indices = torch.cat([data[0].indices for data in dataset_batch]).to(self.device)
        batched_labels = torch.cat([data[1].label.unsqueeze(0) for data in dataset_batch]).to(self.device)

        for i in range(0, len(index_list), self.max_batch):
            batch_slice = slice(i, i + self.max_batch)
            cloud_batch = batched_cloud[batch_slice]
            indices_batch = batched_indices[batch_slice]
            batch_inputs = Inputs(cloud=cloud_batch, indices=indices_batch)
            batch_labels = batched_labels[batch_slice]
            yield batch_inputs, batch_labels

    @abstractmethod
    def __getitems__(self, index_list: list[int]) -> list[Any]:
        """Subclasses must implement this method."""
        pass


class ClassifierMixin:
    """Mixin for datasets that need classifier functionality."""

    @torch.inference_mode()
    def _run_classifier(self, classifier: Model[Inputs, torch.Tensor], batch_inputs: Inputs) -> torch.Tensor:
        """Run classifier on batch inputs."""
        classifier.module.eval()
        return classifier(batch_inputs)


class WDatasetEncoder(ProcessedDataset[VQ], Dataset[tuple[WInputs, WTargets]]):
    """Dataset for training inner autoencoder with discrete codes."""

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        self.autoencoder = self.autoencoder.train(not torch.is_inference_mode_enabled())
        batch_data: list[tuple[WInputs, WTargets]] = []

        for batch_inputs, _ in self._get_data(index_list):
            batch_ae_data = self._encode(batch_inputs)
            batch_w_q, batch_w_e, batch_one_hot_idx = batch_ae_data.w_q, batch_ae_data.w_e, batch_ae_data.one_hot_idx

            for w_q, w_e, one_hot_idx in zip(batch_w_q, batch_w_e, batch_one_hot_idx, strict=True):
                batch_data.append((WInputs(w_q), WTargets(w_e=w_e, one_hot_idx=one_hot_idx)))

        return batch_data

    @torch.inference_mode()
    def _encode(self, inputs: Inputs) -> Outputs:
        """Encode and quantization using autoencoder."""
        data = self.autoencoder.encode(inputs)
        data.w_e, data.idx, _ = self.autoencoder.quantizer.quantize(data.w_q, self.autoencoder.codebook)
        data.one_hot_idx = self.autoencoder.quantizer.create_one_hot(data.idx)
        return data


class WDatasetWithLogits(WDatasetEncoder[CounterfactualVQVAE], ClassifierMixin):
    """W Dataset with classifier logits for conditional training."""

    classifier: Model[Inputs, torch.Tensor]

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
        classifier: Model[Inputs, torch.Tensor],
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder)
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        self.autoencoder = self.autoencoder.train(not torch.is_inference_mode_enabled())
        batch_data: list[tuple[WInputs, WTargets]] = []

        for batch_inputs, _ in self._get_data(index_list):
            batch_ae_data = self._encode(batch_inputs)
            batch_logits = self._run_classifier(self.classifier, batch_inputs)
            batch_w_q, batch_w_e, batch_one_hot_idx = batch_ae_data.w_q, batch_ae_data.w_e, batch_ae_data.one_hot_idx

            for w_q, _, logit, one_hot_idx in zip(batch_w_q, batch_w_e, batch_logits, batch_one_hot_idx, strict=True):
                batch_data.append((WInputs(w_q, logit), WTargets(w_e=w_q, one_hot_idx=one_hot_idx, logits=logit)))

        return batch_data


class WDatasetWithLogitsFrozen(WDatasetWithLogits):
    """W Dataset with classifier logits for conditional training."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
        classifier: Model[Inputs, torch.Tensor],
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder, classifier=classifier)

        batch_size = 32  # Assuming hardware can support batches of 32 on inference for encoder / classifier
        self.w_dataset: list[tuple[WInputs, WTargets]] = []
        for batch_idx in itertools.batched(range(len(self)), batch_size, strict=False):
            self.w_dataset.extend(super().__getitems__(list(batch_idx)))

        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        return [self.w_dataset[i] for i in index_list]


class DoubleReconstructedDatasetEncoder(ProcessedDataset[VQVAE], Dataset[tuple[Inputs, Targets]]):
    """Dataset with reconstructed inputs after double encoding."""

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data: list[tuple[Inputs, Targets]] = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_ae_data = self._reconstruct(batch_inputs)
            recons = batch_ae_data.recon

            for recon, label in zip(recons, batch_labels, strict=True):
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))

        return batch_data

    @torch.inference_mode()
    def _reconstruct(self, inputs: Inputs) -> Outputs:
        """Run autoencoder with double encoding."""
        return self.autoencoder.double_reconstruct(inputs)


class DoubleReconstructedDatasetWithLogits(
    ProcessedDataset[CounterfactualVQVAE], ClassifierMixin, Dataset[tuple[Inputs, Targets]]
):
    """Reconstructed dataset with classifier logits."""

    classifier: Model[Inputs, torch.Tensor]

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
        classifier: Model[Inputs, torch.Tensor],
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder)
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_logits = self._run_classifier(self.classifier, batch_inputs)
            batch_ae_data = self._reconstruct_with_logits(batch_inputs, batch_logits)
            recons = batch_ae_data.recon

            for recon, label in zip(recons, batch_labels, strict=True):
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))

        return batch_data

    @torch.inference_mode()
    def _reconstruct_with_logits(self, inputs: Inputs, batched_logits: torch.Tensor) -> Outputs:
        """Run autoencoder with double encoding."""
        return self.autoencoder.double_reconstruct_with_logits(inputs, batched_logits)


class CounterfactualDatasetEncoder(
    ProcessedDataset[CounterfactualVQVAE], ClassifierMixin, Dataset[tuple[Inputs, Targets]]
):
    """Dataset for counterfactual reconstruction with target conditioning."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
        classifier: Model[Inputs, torch.Tensor],
        target_dim: int,
        target_value: float = 1.0,
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder)
        cfg = Experiment.get_config()
        self.n_classes: int = cfg.data.dataset.n_classes
        self.target_dim: int = target_dim  # don't convert to tensor yet
        self.target_value: float = target_value
        self.classifier: Model[Inputs, torch.Tensor] = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data: list[tuple[Inputs, Targets]] = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_logits = self._run_classifier(self.classifier, batch_inputs)
            batch_ae_data = self.counterfactual_data(batch_inputs, batch_logits, self.target_dim, self.target_value)
            recons = batch_ae_data.recon

            # For targets, use original labels when target_label is 'original'
            target_for_labels = batch_labels if self.target_dim == 'original' else torch.tensor(self.target_dim)

            for recon, original_label in zip(recons, batch_labels, strict=True):
                final_label = original_label if self.target_dim == 'original' else target_for_labels
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=final_label)))

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


class BoundaryDataset(CounterfactualDatasetEncoder):
    """Dataset for boundary reconstruction with neutral conditioning."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
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
