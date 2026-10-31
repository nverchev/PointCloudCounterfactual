"""Online augmentation of the point clouds, applied per sample during training."""

import torch

__all__ = ['jitter']


def jitter(cloud: torch.Tensor, /, *, sigma: float, clip: float) -> torch.Tensor:
    """Add clipped Gaussian noise to the coordinates.

    Args:
        cloud: coordinates ``(N, IN_CHAN)``, left unchanged.
        sigma: standard deviation of the noise.
        clip: largest absolute value of the noise.

    Returns:
        The perturbed coordinates.
    """
    return cloud + torch.clamp(sigma * torch.randn(cloud.shape), min=-clip, max=clip)
