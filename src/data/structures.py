"""Classes for inputs, outputs and targets."""

import dataclasses

from typing import NamedTuple, Self

import torch


class Inputs(NamedTuple):
    """Input for the outer autoencoder.

    Attributes:
        cloud: the input cloud.
        initial_sampling: specify the sampling for the reconstructed cloud from the initial sampling space.
    """

    cloud: torch.Tensor
    initial_sampling: torch.Tensor = torch.empty(0)


class Targets(NamedTuple):
    """Targets for the outer autoencoder.

    Attributes:
        ref_cloud: the reference cloud for reconstruction.
        scale: original scale factor for the cloud.
        label: label of the sample.
    """

    ref_cloud: torch.Tensor
    scale: torch.Tensor = torch.empty(0)
    label: torch.Tensor = torch.empty(0)


@dataclasses.dataclass(init=False, slots=True)
class Outputs:
    """Outputs of the inner and outer autoencoder.

    Attributes:
        model_epoch: the epoch of the model (used for annealing in some losses)
        recon: the reconstructed point cloud.
        word: the discrete encodings' embeddings (with straight-through gradients to word_approx).
        word_approx: outer encoder approximation of the discrete encodings' embedding.
        word_quantised: the discrete encodings' embeddings vectorized (gradients flow to codebook).
        word_recon: inner autoencoder reconstruction of the discrete encodings' embeddings.
        quantization_error: square distances between w and the embeddings.
        idx: the discrete encoding as the index for the embedding.
        one_hot_idx: the discrete encoding as one hot encoding for the index.
        z1: the first latent variable.
        z2: the second latent variable.
        mu1: the mean of the distribution for z1.
        log_var1: the log variance of the distribution for z1.
        pseudo_mu1: the mean of the distribution for the VAMP loss for z1.
        pseudo_log_var1: the log variance of the distribution for the VAMP loss for z1
        p_mu2: the mean of the prior distribution for z2 | probs.
        p_log_var2:  the log variance of the prior distribution for z2 | probs.
        d_mu2: the difference in mean between the prior and posterior distributions for z2 | probs.
        d_log_var2: the difference in log variance between the prior and posterior distributions for z2 | probs.
        probs: classifier or counterfactual probabilities for z2 | probs.
    """

    model_epoch: int
    recon: torch.Tensor
    word: torch.Tensor
    word_approx: torch.Tensor
    word_quantised: torch.Tensor
    word_recon: torch.Tensor
    quantization_error: torch.Tensor
    idx: torch.Tensor
    one_hot_idx: torch.Tensor
    z1: torch.Tensor
    z2: torch.Tensor
    mu1: torch.Tensor
    log_var1: torch.Tensor
    pseudo_mu1: torch.Tensor
    pseudo_log_var1: torch.Tensor
    p_mu2: torch.Tensor
    p_log_var2: torch.Tensor
    d_mu2: torch.Tensor
    d_log_var2: torch.Tensor
    probs: torch.Tensor
    v_pred: torch.Tensor
    v: torch.Tensor

    def update(self, other: Self) -> None:
        """Update the state with another instance's one."""
        for attribute in other.__slots__:
            try:
                setattr(self, attribute, getattr(other, attribute))
            except AttributeError:  # slot not yet initialized
                pass

    def __repr__(self):
        field_names = [f.name for f in dataclasses.fields(self) if f.repr]

        parts = []
        for name in field_names:
            if hasattr(self, name):
                try:
                    value = getattr(self, name)
                    parts.append(f'{name}={value!r}')
                except Exception:
                    parts.append(f'{name}=<error>')
            else:
                parts.append(f'{name}=<uninitialized>')

        return f'{self.__class__.__name__}({", ".join(parts)})'


class WInputs(NamedTuple):
    """Targets for training the inner autoencoder.

    Attributes:
        word_approx: outer encoder approximation of the discrete encodings' embedding.
        logits: classifier evaluation needed for CounterFactualWAutoencoder.
    """

    word_approx: torch.Tensor
    logits: torch.Tensor = torch.empty(0)


class WTargets(NamedTuple):
    """Targets for training the inner autoencoder.

    Attributes:
        word_quantized: the discrete encodings' embedding vectorized.
        one_hot_idx: torch.Tensor
        logits: classifier evaluation needed for CounterFactualWAutoencoder.
    """

    word_quantized: torch.Tensor
    one_hot_idx: torch.Tensor
    logits: torch.Tensor = torch.empty(0)
