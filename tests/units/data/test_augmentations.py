"""Tests for the augmentation."""

import torch

from src.data.augmentations import jitter


def test_jitter_is_clipped() -> None:
    """The noise never exceeds the clip and leaves the input unchanged."""
    cloud = torch.zeros(64, 3)

    jittered = jitter(cloud, sigma=1.0, clip=0.01)

    assert jittered.abs().max() <= 0.01
    assert cloud.abs().max() == 0
