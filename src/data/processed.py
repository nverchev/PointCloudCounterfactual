"""Dataset pre-processed by other models."""

import abc

from collections.abc import Sized
from typing import Any, Generic, TypeVar, override

import torch

from torch.utils.data import Dataset

from src.module import BaseClassifier, BaseVAE, CounterfactualVAE
from src.data.structures import Inputs, Outputs, Targets
from src.data.protocols import AbstractSingleton

V = TypeVar('V', bound=BaseVAE)


class ClassifierMixin:
    classifier: BaseClassifier

    @property
    def device(self) -> torch.device:
        return next(self.classifier.parameters()).device

    @torch.inference_mode()
    def _classify(self, batch_inputs: Inputs) -> torch.Tensor:
        self.classifier.eval()
        return self.classifier(batch_inputs)


class AutoencoderMixin(Generic[V]):
    autoencoder: V

    @property
    def device(self) -> torch.device:
        return next(self.autoencoder.parameters()).device

    @torch.inference_mode()
    def _reconstruct(self, batch_inputs: Inputs) -> Outputs:
        self.autoencoder.eval()
        return self.autoencoder(batch_inputs)


class CounterfactualMixin(ClassifierMixin, AutoencoderMixin[CounterfactualVAE]):
    target_dim: int
    target_value: float

    @property
    def device(self) -> torch.device:
        classifier_device = next(self.classifier.parameters()).device
        autoencoder_device = next(self.autoencoder.parameters()).device
        if classifier_device != autoencoder_device:
            raise ValueError('Autoencoder and classifier must be on the same device')

        return autoencoder_device

    @torch.inference_mode()
    def _generate_counterfactual(self, batch_inputs: Inputs) -> Outputs:
        self.autoencoder.eval()
        return self.autoencoder.generate_counterfactual(batch_inputs, self.target_dim, self.target_value)


class ProcessedDataset(abc.ABC, metaclass=AbstractSingleton):
    """Base dataset for VAE models with common functionality."""

    def __init__(self, dataset: Dataset[tuple[Inputs, Targets]], pin_memory: bool = True) -> None:
        self.dataset = dataset
        self.pin_memory = pin_memory
        return

    def __len__(self) -> int:
        assert isinstance(self.dataset, Sized)
        return len(self.dataset)

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    def _collate(self, index_list: list[int]) -> tuple[Inputs, Targets]:
        batch_data = [self.dataset[i] for i in index_list]
        cloud = torch.stack([d[0].cloud for d in batch_data])
        labels = torch.stack([d[1].label for d in batch_data])
        if self.pin_memory:
            cloud, labels = cloud.pin_memory(), labels.pin_memory()

        cloud = cloud.to(self.device, non_blocking=self.pin_memory)
        labels = labels.to(self.device, non_blocking=self.pin_memory)
        return Inputs(cloud=cloud), Targets(ref_cloud=cloud, label=labels)

    @staticmethod
    def _unbind(batch_inputs: Inputs, batch_targets: Targets) -> list[tuple[Inputs, Targets]]:
        return [
            (
                Inputs(cloud=batch_inputs.cloud[i], logits=batch_inputs.logits[i]),
                Targets(ref_cloud=batch_targets.ref_cloud[i], label=batch_targets.label[i]),
            )
            for i in range(len(batch_targets.label))
        ]

    @abc.abstractmethod
    def __getitems__(self, index_list: list[int]) -> list[Any]:
        """Get pre-processed data for the given indices."""


class EvaluatedDataset(ClassifierMixin, ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that attaches classifier logits to each sample."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        classifier: BaseClassifier,
    ) -> None:
        super().__init__(dataset=dataset)
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(logits=self._classify(batch_inputs))
        return self._unbind(batch_inputs, batch_targets)


class ReconstructedDataset(AutoencoderMixin[V], ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that replaces inputs with autoencoder reconstructions."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: V,
    ) -> None:
        super().__init__(dataset=dataset)
        self.autoencoder = autoencoder
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        recons = self._reconstruct(batch_inputs).recon
        batch_inputs = Inputs(cloud=recons)
        batch_targets = Targets(ref_cloud=recons, label=batch_targets.label)
        return self._unbind(batch_inputs, batch_targets)


class ReconstructedEvaluatedDataset(CounterfactualMixin, ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that classifies inputs first, then reconstructs using logits."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVAE,
        classifier: BaseClassifier,
    ) -> None:
        super().__init__(dataset=dataset)
        self.autoencoder = autoencoder
        self.classifier = classifier
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(logits=self._classify(batch_inputs))
        recons = self._reconstruct(batch_inputs).recon
        batch_inputs = Inputs(cloud=recons, logits=batch_inputs.logits)
        batch_targets = Targets(ref_cloud=recons, label=batch_targets.label)
        return self._unbind(batch_inputs, batch_targets)


class CounterfactualDataset(CounterfactualMixin, ProcessedDataset, Dataset[tuple[Inputs, Targets]]):
    """Dataset that generates counterfactual reconstructions conditioned on a target class."""

    def __init__(
        self,
        dataset: Dataset[tuple[Inputs, Targets]],
        autoencoder: CounterfactualVAE,
        classifier: BaseClassifier,
        target_dim: int,
        target_value: float = 1.0,
    ) -> None:
        super().__init__(dataset=dataset)
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.target_dim = target_dim
        self.target_value = target_value
        return

    @override
    def __getitems__(self, index_list: list[int]) -> list[tuple[Inputs, Targets]]:
        batch_inputs, batch_targets = self._collate(index_list)
        batch_inputs = batch_inputs._replace(logits=self._classify(batch_inputs))
        recons = self._generate_counterfactual(batch_inputs).recon
        batch_inputs = Inputs(cloud=recons, logits=batch_inputs.logits)
        batch_targets = batch_targets._replace(label=torch.tensor([self.target_dim]))
        return self._unbind(batch_inputs, batch_targets)
