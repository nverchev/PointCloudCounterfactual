"""Classes containing data samples and outputs."""

import dataclasses
from typing import NamedTuple, Self

import torch

IN_CHAN = 3
OUT_CHAN = 3


class Inputs(NamedTuple):
    """
    Input for the outer autoencoder.

    Attributes:
        cloud: the input cloud.
        indices: precalculated indices for the nearest neighbors (number must match architecture).
        initial_sampling: specify the sampling for the reconstructed cloud from the initial sampling space.
    """
    cloud: torch.Tensor
    indices: torch.Tensor = torch.empty(0)
    initial_sampling: torch.Tensor = torch.empty(0)


class Targets(NamedTuple):
    """
    Targets for the outer autoencoder.

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
    """
    Outputs of the inner and outer autoencoder.

    Attributes:
        model_epoch: the epoch of the model (used for annealing in some losses)
        recon: the reconstruction output.
        w: the discrete encodings' embeddings (with straight-through gradients).
        w_q: outer encoder approximation of the discrete encodings' embedding.
        w_e: the discrete encodings' embeddings (no gradients).
        w_recon: inner autoencoder approximations of the discrete encodings' embeddings.
        w_dist_2: square distances between w and the embeddings.
        idx: the discrete encoding as the index for the embedding.
        one_hot_idx: the discrete encoding as one hot encoding for the index.
        attention_weights: torch.Tensor
        components: torch.Tensor
        z1: the first latent variable.
        z2: the second latent variable.
        mu1: the mean of the distribution for z1.
        log_var1: the log variance of the distribution for z1.
        pseudo_mu1: the mean of the distribution for the VAMP loss for z1.
        pseudo_log_var1: the log variance of the distribution for the VAMP loss for z1
        p_mu2: the mean of the prior distribution for z2.
        p_log_var2:  the log variance of the prior distribution for z2.
        d_mu2: the difference in mean between the prior and posterior distributions for z2.
        d_log_var2: the difference in log variance between the prior and posterior distributions for z2.
        p_mu2: the mean of the prior distribution for z2.
        p_log_var2:  the log variance of the prior distribution for z2.
        h: hidden features for the hierarchical inner autoencoder.
        probs: optional condition value for the z2.
        y1: output of the discriminator for the conditional inner autoencoder after detaching inputs.
        y2: output of the evaluated discriminator for the conditional inner autoencoder discriminative loss.

    """
    model_epoch: int
    recon: torch.Tensor
    w: torch.Tensor
    w_q: torch.Tensor
    w_e: torch.Tensor
    w_recon: torch.Tensor
    w_dist_2: torch.Tensor
    idx: torch.Tensor
    one_hot_idx: torch.Tensor
    attention_weights: torch.Tensor
    components: torch.Tensor
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
    h: torch.Tensor
    probs: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor

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
                    parts.append(f"{name}={value!r}")
                except Exception:
                    parts.append(f"{name}=<error>")
            else:
                parts.append(f"{name}=<uninitialized>")

        return f"{self.__class__.__name__}({', '.join(parts)})"


class WInputs(NamedTuple):
    """
    Targets for training the inner autoencoder.

    Attributes:
        w_q: outer encoder approximation of the discrete encodings' embedding.
        logits: optional argument for learning a conditional distribution given a classifier evaluation.
    """
    w_q: torch.Tensor
    logits: torch.Tensor = torch.empty(0)


class WTargets(NamedTuple):
    """
    Targets for training the inner autoencoder.

    Attributes:
        w_e: the discrete encodings' embedding vectorized.
        one_hot_idx: torch.Tensor
        logits: optional argument to get a conditional distribution given a classifier's evaluation.
    """
    w_e: torch.Tensor
    one_hot_idx: torch.Tensor
    logits: torch.Tensor = torch.empty(0)
