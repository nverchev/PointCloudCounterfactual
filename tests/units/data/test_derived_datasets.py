"""Tests for the datasets of what the models derive."""

import numpy as np
import torch

from src.data.derived_datasets import EvaluatedSplit, GeneratedSplit, WDataset
from src.data.split import CloudSplit
from src.lib.models import get_classifier_model, get_vqvae_model


def _get_split() -> CloudSplit:
    return CloudSplit(np.random.rand(4, 32, 3).astype(np.float32), np.array([0, 1, 0, 1]), augment=False)


def test_evaluated_split_adds_the_condition() -> None:
    """A sample carries the evaluation of its cloud."""
    conditions = torch.rand(4, 2)

    inputs, _ = EvaluatedSplit(_get_split(), conditions)[2]

    torch.testing.assert_close(inputs.condition, conditions[2])


def test_generated_split_scores_the_given_labels() -> None:
    """A generated cloud is compared with the label it was given."""
    clouds = torch.rand(2, 32, 3)

    inputs, targets = GeneratedSplit(clouds, torch.tensor([1, 0]))[1]

    torch.testing.assert_close(inputs.cloud, clouds[1])
    assert targets.label == 0


class TestWDataset:
    """Test the codewords with their evaluations."""

    def test_batch_gives_one_entry_per_cloud(self) -> None:
        """Each entry has the codeword, an evaluation on the simplex and the chosen codes."""
        vqvae, classifier = get_vqvae_model(), get_classifier_model()
        dataset = WDataset(_get_split(), vqvae, classifier)

        (inputs, one_hot_idx), *others = dataset.__getitems__([0, 1, 2])

        assert len(others) == 2
        assert inputs.w_q.shape == (16,)
        torch.testing.assert_close(inputs.condition.sum(), torch.tensor(1.0))
        assert one_hot_idx.sum() == vqvae.module.dim_codes

    def test_training_mode_keeps_the_running_statistics(self) -> None:
        """Encoding the clouds leaves the batch normalization of the trained VQ-VAE as it was."""
        vqvae, classifier = get_vqvae_model(), get_classifier_model()
        dataset = WDataset(_get_split(), vqvae, classifier)
        before = {key: value.clone() for key, value in vqvae.module.state_dict().items()}

        dataset.__getitems__([0, 1, 2])

        for key, value in vqvae.module.state_dict().items():
            torch.testing.assert_close(value, before[key])
