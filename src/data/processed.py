"""Classes with Dataset that have been preprocessed for use with VQVAE models."""

import abc
import itertools

from abc import abstractmethod
from collections.abc import Generator, Sized
from typing import Any, Generic, Literal, TypeVar, override

import torch

from torch.utils.data import Dataset

from drytorch import Model

from src.module.autoencoders import AbstractVQVAE, CounterfactualVQVAE
from src.module.w_autoencoders import BaseWAutoEncoder, CounterfactualWAutoEncoder
from src.data.structures import Inputs, Outputs, Targets, WInputs, WTargets
from src.data.protocols import AbstractSingleton

VQ = TypeVar('VQ', bound=AbstractVQVAE[BaseWAutoEncoder])


class DatasetEncoder(Generic[VQ], abc.ABC, metaclass=AbstractSingleton):
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


class WDatasetEncoder(DatasetEncoder[VQ], Dataset[tuple[WInputs, WTargets]]):
    """Dataset for training inner autoencoder with discrete codes."""

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        self.autoencoder = self.autoencoder.train(not torch.is_inference_mode_enabled())
        batch_data: list[tuple[WInputs, WTargets]] = []

        for batch_inputs, _ in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs)
            batch_w_q, batch_w_e, batch_one_hot_idx = batch_ae_data.w_q, batch_ae_data.w_e, batch_ae_data.one_hot_idx

            for w_q, w_e, one_hot_idx in zip(batch_w_q, batch_w_e, batch_one_hot_idx, strict=True):
                batch_data.append((WInputs(w_q), WTargets(w_e=w_e, one_hot_idx=one_hot_idx)))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(self, inputs: Inputs) -> Outputs:
        """Run autoencoder encoding and quantization."""
        data = self.autoencoder.encode(inputs)
        data.w_e, data.one_hot_idx = self.autoencoder.quantizer.quantize(data.w_q)
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
            batch_ae_data = self._run_autoencoder(batch_inputs)
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
        self.w_dataset: list[tuple[WInputs, WTargets]] = []
        for batch_idx in itertools.batched(range(len(self)), 32, strict=True):
            self.w_dataset.extend(super().__getitems__(list(batch_idx)))

        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[WInputs, WTargets]]:
        return [self.w_dataset[i] for i in index_list]


class ReconstructedDatasetEncoder(DatasetEncoder[VQ], Dataset[tuple[Inputs, Targets]]):
    """Dataset with reconstructed inputs after double encoding."""

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data: list[tuple[Inputs, Targets]] = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs)
            recons = batch_ae_data.recon

            for recon, label in zip(recons, batch_labels, strict=True):
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(self, inputs: Inputs) -> Outputs:
        """Run autoencoder with double encoding."""
        with self.autoencoder.double_encoding:
            return self.autoencoder(inputs)


class ReconstructedDatasetWithLogits(ReconstructedDatasetEncoder[CounterfactualVQVAE], ClassifierMixin):
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
            batch_ae_data = self._run_autoencoder(batch_inputs, batch_logits)
            recons = batch_ae_data.recon

            for recon, label in zip(recons, batch_labels, strict=True):
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=label)))

        return batch_data

    @torch.inference_mode()
    @override
    def _run_autoencoder(self, inputs: Inputs, batched_logits: None | torch.Tensor = None) -> Outputs:
        """Run autoencoder with logits conditioning."""
        if batched_logits is None:
            batched_logits = self._run_classifier(self.classifier, inputs)

        with self.autoencoder.double_encoding:
            data = self.autoencoder.encode(inputs)
            if isinstance(self.autoencoder.w_autoencoder, CounterfactualWAutoEncoder):
                data.probs = self.autoencoder.w_autoencoder.relaxed_softmax(batched_logits)

            return self.autoencoder.decode(data, inputs)


class CounterfactualDatasetEncoder(
    DatasetEncoder[CounterfactualVQVAE], ClassifierMixin, Dataset[tuple[Inputs, Targets]]
):
    """Dataset for counterfactual reconstruction with target conditioning."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
        classifier: Model[Inputs, torch.Tensor],
        target_label: int | Literal['original'] = 'original',
        target_value: float = 1.0,
        n_classes: int = 2,
    ) -> None:
        super().__init__(dataset=dataset, autoencoder=autoencoder)
        self.num_classes: int = n_classes
        self.target_label: int | Literal['original'] = target_label  # don't convert to tensor yet
        self.target_value: float = target_value
        self.classifier: Model[Inputs, torch.Tensor] = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        self.autoencoder = self.autoencoder.eval()
        batch_data: list[tuple[Inputs, Targets]] = []

        for batch_inputs, batch_labels in self._get_data(index_list):
            batch_ae_data = self._run_autoencoder(batch_inputs, None, self.target_label, self.target_value)
            recons = batch_ae_data.recon

            # For targets, use original labels when target_label is 'original'
            target_for_labels = batch_labels if self.target_label == 'original' else torch.tensor(self.target_label)

            for recon, original_label in zip(recons, batch_labels, strict=True):
                final_label = original_label if self.target_label == 'original' else target_for_labels
                batch_data.append((Inputs(cloud=recon), Targets(ref_cloud=recon, label=final_label)))

        return batch_data

    @torch.inference_mode()
    def _run_autoencoder(
        self,
        inputs: Inputs,
        batched_logits: None | torch.Tensor = None,
        target_dim: int | Literal['original'] = 'original',
        value: float = 1.0,
    ) -> Outputs:
        """Run autoencoder with counterfactual conditioning."""
        if batched_logits is None:
            batched_logits = self._run_classifier(self.classifier, inputs)

        with self.autoencoder.double_encoding:
            data = self.autoencoder.encode(inputs)
            probs = self.autoencoder.w_autoencoder.relaxed_softmax(batched_logits)

            # Initialize target tensor
            target = torch.zeros_like(probs)

            if target_dim == 'original':
                # Use original probabilities (no modification)
                data.probs = probs
            else:
                # Apply counterfactual conditioning
                target[:, target_dim] = 1
                data.probs = (1 - value) * probs + value * target

            return self.autoencoder.decode(data, inputs)


class BoundaryDataset(CounterfactualDatasetEncoder):
    """Dataset for boundary reconstruction with neutral conditioning."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVQVAE,
        classifier: Model[Inputs, torch.Tensor],
        target_label: int = 0,
        n_classes: int = 2,
    ) -> None:
        super().__init__(
            dataset=dataset,
            autoencoder=autoencoder,
            classifier=classifier,
            target_label=target_label,
            target_value=1 / n_classes,  # Neutral probability
            n_classes=n_classes,
        )
