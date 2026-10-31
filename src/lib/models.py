"""Model wrappers of the three training phases and the loading of the trained ones."""

from typing import Any, Final, Literal

import torch

from drytorch.core import protocols as p
from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib import models

from src.config import Experiment
from src.data.structures import Inputs, Outputs, WInputs, WOutputs
from src.module import DGCNN, VQVAE, CounterfactualVQVAE, WAutoEncoder

__all__ = [
    'ClassifierModel',
    'CounterfactualModel',
    'VQVAEModel',
    'WModel',
    'get_classifier_model',
    'get_counterfactual_model',
    'get_vqvae_model',
    'get_w_model',
    'load_classifier_model',
    'load_counterfactual_model',
    'load_trained_state',
    'load_vqvae_model',
    'load_w_model',
]

_COUNTERFACTUAL_MODEL_NAME: Final = 'counterfactual_model'


class ClassifierModel(models.Model[Inputs, torch.Tensor]):
    """Classifier under audit.

    Attributes:
        module: the DGCNN.
        temperature: Temperature dividing the logits before the softmax.
    """

    module: DGCNN
    temperature: Final[float]

    def __init__(
        self,
        module: DGCNN,
        /,
        *,
        temperature: float,
        checkpoint: p.CheckpointProtocol | None = None,
        device: torch.device | Literal['auto'] = 'auto',
        distribute: bool = True,
        mixed_precision: torch.dtype | None = None,
        name: str = '',
        torch_compile: bool = False,
    ) -> None:
        """Initialize.

        Args:
            module: the DGCNN.
            temperature: temperature dividing the logits before the softmax.
            checkpoint: manages saving and loading of model weights.
            device: the device where to store the weights of the module.
            distribute: whether to wrap the module for ddp training.
            mixed_precision: the autocast precision.
            name: the name of the model.
            torch_compile: whether to compile the module with torch.compile.
        """
        super().__init__(
            module,
            checkpoint=checkpoint,
            device=device,
            distribute=distribute,
            mixed_precision=mixed_precision,
            name=name,
            torch_compile=torch_compile,
        )
        self.temperature = temperature
        return

    @torch.inference_mode()
    def evaluate(self, clouds: torch.Tensor, /) -> torch.Tensor:
        """Get the evaluation of the clouds: their probabilities, remapped by the temperature.

        Args:
            clouds: clouds on the device of the model ``(B, N, IN_CHAN)``.

        Returns:
            The temperature-scaled softmax of the logits ``(B, C)``, a point of the probability simplex.
        """
        return torch.softmax(self.logits(clouds) / self.temperature, dim=1)

    @torch.inference_mode()
    def logits(self, clouds: torch.Tensor, /) -> torch.Tensor:
        """Get the logits of the clouds, in evaluation mode.

        Args:
            clouds: clouds on the device of the model ``(B, N, IN_CHAN)``.

        Returns:
            The logits ``(B, C)``.
        """
        self.module.eval()
        return self(Inputs(cloud=clouds))

    def evaluate_split(self, clouds: torch.Tensor, /, *, batch_size: int) -> torch.Tensor:
        """Get the evaluation of many clouds, in batches.

        Args:
            clouds: clouds on any device ``(N, n_points, IN_CHAN)``.
            batch_size: size of the batches.

        Returns:
            The evaluations, on the CPU ``(N, C)``.
        """
        return torch.cat([self.evaluate(batch.to(self.device)).cpu() for batch in clouds.split(batch_size)])

    def predict_split(self, clouds: torch.Tensor, /, *, batch_size: int) -> torch.Tensor:
        """Get the predicted class of many clouds, in batches.

        Args:
            clouds: clouds on any device ``(N, n_points, IN_CHAN)``.
            batch_size: size of the batches.

        Returns:
            The predictions, on the CPU ``(N,)``.
        """
        return torch.cat([self.logits(batch.to(self.device)).argmax(1).cpu() for batch in clouds.split(batch_size)])


class VQVAEModel(models.Model[Inputs, Outputs]):
    """Outer VQ-VAE.

    Attributes:
        module: the VQ-VAE.
    """

    module: VQVAE


class WModel(models.Model[WInputs, WOutputs]):
    """Inner VAE.

    Attributes:
        module: the variational autoencoder of the codewords.
    """

    module: WAutoEncoder


class CounterfactualModel(models.Model[Inputs, Outputs]):
    """Outer VQ-VAE decoding the codes that the inner VAE chooses.

    Attributes:
        module: the two autoencoders chained.
    """

    module: CounterfactualVQVAE

    @torch.inference_mode()
    def generate(self, clouds: torch.Tensor, conditions: torch.Tensor, /, *, batch_size: int) -> torch.Tensor:
        """Generate clouds from the originals under the given evaluations, without sampling the latents.

        Args:
            clouds: original clouds ``(N, n_points, IN_CHAN)``.
            conditions: evaluations ``(N, C)``: the original ones reconstruct, target ones intervene.
            batch_size: size of the batches.

        Returns:
            The generated clouds, on the CPU ``(N, n_points, OUT_CHAN)``.
        """
        self.module.eval()
        batches = zip(clouds.split(batch_size), conditions.split(batch_size), strict=True)
        return torch.cat(
            [
                self(Inputs(cloud=cloud.to(self.device), condition=condition.to(self.device))).recon.cpu()
                for cloud, condition in batches
            ]
        )


def get_classifier_model() -> ClassifierModel:
    """Get the classifier of the classification phase."""
    cfg = Experiment.get_config()
    return ClassifierModel(
        DGCNN(),
        temperature=cfg.w_autoencoding.model.temperature,
        device=cfg.user.device,
        name=cfg.classification.model.name,
    )


def get_counterfactual_model(vqvae: VQVAEModel, w_model: WModel, /) -> CounterfactualModel:
    """Chain the trained autoencoders.

    Args:
        vqvae: trained outer VQ-VAE.
        w_model: trained inner VAE.

    Returns:
        The counterfactual model.
    """
    module = CounterfactualVQVAE(vqvae.module, w_model.module)
    return CounterfactualModel(module, device=vqvae.device, name=_COUNTERFACTUAL_MODEL_NAME)


def get_vqvae_model() -> VQVAEModel:
    """Get the outer VQ-VAE of the autoencoding phase."""
    cfg = Experiment.get_config()
    return VQVAEModel(VQVAE(), device=cfg.user.device, name=cfg.autoencoding.model.name)


def get_w_model() -> WModel:
    """Get the inner VAE of the codeword phase."""
    cfg = Experiment.get_config()
    return WModel(WAutoEncoder(), device=cfg.user.device, name=cfg.w_autoencoding.model.name)


def load_classifier_model() -> ClassifierModel:
    """Get the classifier with the weights of the classification phase.

    Raises:
        RuntimeError: if the experiment holds no trained classifier.
    """
    model = get_classifier_model()
    load_trained_state(model)
    return model


def load_counterfactual_model() -> CounterfactualModel:
    """Chain the trained autoencoders of the run.

    Returns:
        The counterfactual model, holding the last checkpoints.

    Raises:
        RuntimeError: if the experiment holds no trained autoencoders.
    """
    return get_counterfactual_model(load_vqvae_model(), load_w_model())


def load_trained_state(model: models.Model[Any, Any], /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.par_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return


def load_vqvae_model() -> VQVAEModel:
    """Get the outer VQ-VAE with the weights of the autoencoding phase.

    Raises:
        RuntimeError: if the experiment holds no trained VQ-VAE.
    """
    model = get_vqvae_model()
    load_epoch = Experiment.get_config().user.load or -1
    load_trained_state(model, epoch=load_epoch)
    return model


def load_w_model() -> WModel:
    """Get the inner VAE with the weights of the codeword phase.

    Raises:
        RuntimeError: if the experiment holds no trained inner VAE.
    """
    model = get_w_model()
    load_trained_state(model)
    return model
