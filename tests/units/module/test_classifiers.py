"""Tests for the classifier."""

import torch

from src.data.structures import Inputs
from src.module import DGCNN


def test_dgcnn_gives_one_logit_per_class(example_clouds: torch.Tensor) -> None:
    """The output has one logit per configured class."""
    logits = DGCNN()(Inputs(cloud=example_clouds))

    assert logits.shape == (4, 2)
