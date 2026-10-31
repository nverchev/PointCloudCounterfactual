"""Typed I/O structures for the models, the losses and the metrics."""

import dataclasses

from typing import Final, NamedTuple

import torch

__all__ = [
    'IN_CHAN',
    'OUT_CHAN',
    'Inputs',
    'Outputs',
    'Quantization',
    'Targets',
    'WInputs',
    'WOutputs',
    'get_quantization',
]

IN_CHAN: Final = 3
"""Coordinates of a point in an input cloud."""

OUT_CHAN: Final = 3
"""Coordinates of a point in a generated cloud."""


class Inputs(NamedTuple):
    """Point cloud fed to the models.

    Attributes:
        cloud: Coordinates ``(B, N, IN_CHAN)``.
        condition: Evaluation conditioning the inner VAE ``(B, C)``; empty for the other models.
        initial_sampling: Sampling of the generating surface ``(B, D, M)``; empty to draw one.
    """

    cloud: torch.Tensor
    condition: torch.Tensor = torch.empty(0)
    initial_sampling: torch.Tensor = torch.empty(0)


class Targets(NamedTuple):
    """Reference cloud and label.

    Attributes:
        ref_cloud: Reference coordinates ``(B, N, IN_CHAN)``.
        label: Class of each cloud ``(B,)``.
    """

    ref_cloud: torch.Tensor
    label: torch.Tensor


class WInputs(NamedTuple):
    """Codeword and evaluation fed to the inner VAE.

    Attributes:
        w_q: Codewords before quantization ``(B, W)``.
        condition: Evaluation of the classifier ``(B, C)``.
    """

    w_q: torch.Tensor
    condition: torch.Tensor


class Quantization(NamedTuple):
    """Codewords of the outer VQ-VAE.

    Attributes:
        w_q: Codeword before quantization ``(B, W)``.
        w_e: Quantized codeword ``(B, W)``.
        one_hot_idx: Indicator of the chosen codes ``(B, dim_codes, book_size)``.
    """

    w_q: torch.Tensor
    w_e: torch.Tensor
    one_hot_idx: torch.Tensor


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs of the outer VQ-VAE.

    Attributes:
        recon: Reconstructed coordinates ``(B, M, OUT_CHAN)``.
        quantization: Codewords, ``None`` when the codes were chosen by the inner VAE.
    """

    recon: torch.Tensor
    quantization: Quantization | None = None


@dataclasses.dataclass(slots=True)
class WOutputs:
    """Outputs of the inner VAE.

    Attributes:
        w_recon: Reconstructed codeword ``(B, W)``.
        mu_u: Mean of the posterior of the latent unrelated to the classification ``(B, Z_u)``.
        log_var_u: Log variance of that posterior ``(B, Z_u)``.
        d_mu_y: Shift of the mean of the conditioned latent from its prior ``(B, Z_y)``.
        d_log_var_y: Shift of the log variance of the conditioned latent from its prior ``(B, Z_y)``.
        p_log_var_y: Log variance of the prior of the conditioned latent ``(B, Z_y)``.
    """

    w_recon: torch.Tensor
    mu_u: torch.Tensor
    log_var_u: torch.Tensor
    d_mu_y: torch.Tensor
    d_log_var_y: torch.Tensor
    p_log_var_y: torch.Tensor


def get_quantization(outputs: Outputs, /) -> Quantization:
    """Get the codewords of the outer VQ-VAE.

    Args:
        outputs: outputs of a quantizing pass.

    Returns:
        The codewords before and after quantization, with the chosen codes.

    Raises:
        ValueError: if the inner VAE chose the codes.
    """
    if outputs.quantization is None:
        raise ValueError('The outputs do not come from a quantizing pass.')

    return outputs.quantization
