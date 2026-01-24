"""Augmentation functions and classes."""

from typing import Any
from collections.abc import Callable, Iterable

import numpy as np
import torch
from numpy import typing as npt

from src.config.experiment import Experiment


def normalise(cloud: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], float]:
    """Standard normalization to unit sphere."""
    cloud -= cloud.mean(axis=0)
    std = np.max(np.sqrt(np.sum(cloud**2, axis=1)))
    cloud /= std
    return cloud, std


def jitter(cloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
    """Add noise to points coordinates."""
    jitter_noise = torch.randn(cloud.shape) * torch.tensor(sigma)
    new_cloud = cloud.clone()
    new_cloud += torch.clamp(jitter_noise, min=-clip, max=clip)
    return new_cloud


def random_rotation() -> Callable[[torch.Tensor], torch.Tensor]:
    """Define random rotation to be applied to input and reference clouds."""
    theta = torch.tensor(2 * torch.pi) * torch.rand(1)
    s = torch.sin(theta)
    rotation_matrix = torch.eye(2) * torch.cos(theta)
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s

    def _rotate(cloud: torch.Tensor) -> torch.Tensor:
        new_cloud = cloud.clone()
        new_cloud[:, [0, 2]] = cloud[:, [0, 2]].mm(rotation_matrix)
        return new_cloud

    return _rotate


def random_scale_and_translate() -> Callable[[torch.Tensor], torch.Tensor]:
    """Define random scaling and translation to be applied to input and reference clouds."""
    scale = torch.rand(1, 3) * 5 / 6 + 2 / 3
    translate = torch.rand(1, 3) * 0.4 - 0.2

    def _scale_and_translate(cloud: torch.Tensor) -> torch.Tensor:
        new_cloud = cloud.clone()
        new_cloud *= scale
        new_cloud += translate
        return new_cloud

    return _scale_and_translate


class CloudAugmenter:
    """Augmentation class for rotation, scaling, and translation."""

    def __init__(self, rotate: bool, translate_and_scale: bool):
        self.rotate = rotate
        self.translation_and_scale = translate_and_scale

    def __call__(self, clouds: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        if self.rotate:
            rotate = random_rotation()
            clouds = map(rotate, clouds)
        if self.translation_and_scale:
            scale_and_translate = random_scale_and_translate()
            clouds = map(scale_and_translate, clouds)
        return clouds


class CloudJitterer:
    """Jitter class."""

    def __init__(self, jitter_sigma: float | None, jitter_clip: float | None):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

    def __call__(self, cloud: torch.Tensor) -> torch.Tensor:
        if self.jitter_sigma and self.jitter_clip:
            return jitter(cloud, self.jitter_sigma, self.jitter_clip)

        return cloud


def augment_clouds() -> CloudAugmenter:
    """Create a callable for augmentation based on configuration."""
    cfg_data = Experiment.get_config().data
    return CloudAugmenter(rotate=cfg_data.rotate, translate_and_scale=cfg_data.translate)


def jitter_cloud() -> CloudJitterer:
    """Create jitter callable based on configuration."""
    cfg_data = Experiment.get_config().data
    return CloudJitterer(jitter_sigma=cfg_data.jitter_sigma, jitter_clip=cfg_data.jitter_clip)
