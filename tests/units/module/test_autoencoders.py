"""Tests for the outer VQ-VAE."""

import pytest
import torch

from src.data.structures import Inputs, get_quantization
from src.module import VQVAE


class TestVQVAE:
    """Test the quantization and the reconstruction."""

    @pytest.fixture
    def module(self) -> VQVAE:
        """Provide the VQ-VAE."""
        return VQVAE()

    def test_forward(self, module: VQVAE, example_clouds: torch.Tensor) -> None:
        """The reconstruction has the size of the cloud, and every group chooses one code."""
        outputs = module(Inputs(cloud=example_clouds))

        quantization = get_quantization(outputs)
        assert outputs.recon.shape == example_clouds.shape
        assert quantization.w_q.shape == quantization.w_e.shape == (4, 16)
        torch.testing.assert_close(quantization.one_hot_idx.sum(2), torch.ones(4, module.dim_codes))

    def test_gradient_reaches_the_encoder_through_the_quantization(
        self, module: VQVAE, example_clouds: torch.Tensor
    ) -> None:
        """The straight-through estimator lets the reconstruction train the encoder."""
        outputs = module(Inputs(cloud=example_clouds))

        outputs.recon.sum().backward()

        assert next(module.encoder.parameters()).grad is not None

    def test_quantise_chooses_the_nearest_code(self, module: VQVAE) -> None:
        """A codeword made of codes of the codebook is left unchanged."""
        codes = module.codebook.data[:, 1].reshape(1, -1)

        quantized, one_hot_idx = module.quantise(codes)

        torch.testing.assert_close(quantized, codes)
        assert (one_hot_idx.argmax(2) == 1).all()

    def test_square_distances_vanish_on_the_chosen_code(self, module: VQVAE) -> None:
        """The distances compare each chunk with each code of its own group."""
        codes = module.codebook.data[:, 2].reshape(1, -1)

        distances = module.square_distances(codes)

        assert distances.shape == (1, module.dim_codes, module.book_size)
        assert (distances.argmin(2) == 2).all()
