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
    logits: torch.Tensor = torch.empty(0)


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
        latent_features: the latent features from the latent decoder.
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
    latent_features: torch.Tensor
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
