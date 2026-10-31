"""Tests for the splits of the clouds."""

import numpy as np
import pytest
import torch

from src.data.split import CloudSplit


class TestCloudSplit:
    """Test the samples of a split."""

    @pytest.fixture
    def clouds(self) -> np.ndarray:
        """Provide two clouds."""
        return np.random.rand(2, 32, 3).astype(np.float32)

    def test_without_augmentation_the_cloud_is_unchanged(self, clouds: np.ndarray) -> None:
        """An evaluation split gives the stored cloud."""
        split = CloudSplit(clouds, np.array([0, 1]), augment=False)

        inputs, targets = split[1]

        torch.testing.assert_close(inputs.cloud, torch.from_numpy(clouds[1]))
        assert inputs.cloud is targets.ref_cloud
        assert targets.label == 1

    def test_with_augmentation_the_target_is_the_input(self, clouds: np.ndarray) -> None:
        """A training split perturbs the cloud, and the model reconstructs that same version."""
        split = CloudSplit(clouds, np.array([0, 1]), augment=True)

        inputs, targets = split[0]
        other_inputs, _ = split[0]

        assert inputs.cloud is targets.ref_cloud
        assert not torch.equal(inputs.cloud, other_inputs.cloud)
