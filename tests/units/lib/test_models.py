"""Tests for the model wrappers."""

import torch

from src.config import Experiment
from src.data.structures import WInputs
from src.lib import models


class TestClassifierModel:
    """Test the evaluation of the clouds."""

    def test_evaluation_is_the_tempered_softmax(self, mocker, example_clouds: torch.Tensor) -> None:
        """The evaluation divides the logits by the temperature, before the softmax."""
        temperature = Experiment.get_config().w_autoencoding.model.temperature
        model = models.get_classifier_model()
        logits = torch.tensor([[4.0, -4.0]]).repeat(4, 1)
        mocker.patch.object(models.ClassifierModel, 'logits', return_value=logits)

        evaluation = model.evaluate(example_clouds)

        torch.testing.assert_close(evaluation, torch.softmax(logits / temperature, dim=1))

    def test_batches_cover_all_the_clouds(self, example_clouds: torch.Tensor) -> None:
        """Splitting the clouds in batches gives one result for each of them."""
        model = models.get_classifier_model()

        evaluations = model.evaluate_split(example_clouds, batch_size=3)
        predictions = model.predict_split(example_clouds, batch_size=3)

        assert evaluations.shape == (4, 2)
        assert predictions.shape == (4,)


def test_counterfactual_model_generates_one_cloud_per_condition(example_clouds: torch.Tensor) -> None:
    """The generated clouds keep the order and the size of the originals."""
    model = models.get_counterfactual_model(models.get_vqvae_model(), models.get_w_model())
    conditions = torch.rand(8, 2).softmax(1)
    model.module.w_autoencoder(WInputs(torch.randn(8, 16), conditions))

    generated = model.generate(example_clouds, conditions[:4], batch_size=3)

    assert generated.shape == example_clouds.shape
