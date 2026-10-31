"""Tests for the inner VAE and the chained model."""

import pytest
import torch

from src.data.structures import Inputs, WInputs
from src.module import VQVAE, CounterfactualVQVAE, WAutoEncoder


class TestWAutoEncoder:
    """Test the conditioning of the inner VAE."""

    @pytest.fixture
    def module(self) -> WAutoEncoder:
        """Provide the inner VAE, with its statistics set by a training batch."""
        module = WAutoEncoder()
        module(WInputs(torch.randn(8, 16), torch.rand(8, 2).softmax(1)))
        return module.eval()

    def test_outputs(self, module: WAutoEncoder) -> None:
        """The codeword is reconstructed and each latent has its dimension."""
        outputs = module(WInputs(torch.randn(4, 16), torch.full((4, 2), 0.5)))

        assert outputs.w_recon.shape == (4, 16)
        assert outputs.mu_u.shape == outputs.log_var_u.shape == (4, 2)
        assert outputs.d_mu_y.shape == outputs.p_log_var_y.shape == (4, 2)

    def test_evaluation_mode_does_not_sample(self, module: WAutoEncoder) -> None:
        """Without sampling noise, a counterfactual stays tied to the sample it explains."""
        inputs = WInputs(torch.randn(4, 16), torch.full((4, 2), 0.5))

        first, second = module(inputs).w_recon, module(inputs).w_recon

        torch.testing.assert_close(first, second)

    def test_intervention_changes_only_z_y(self, module: WAutoEncoder) -> None:
        """A target evaluation moves the posterior of ``z_y`` and leaves the one of ``z_u``."""
        w_q = torch.randn(4, 16)
        original = module(WInputs(w_q, torch.tensor([[0.9, 0.1]]).repeat(4, 1)))
        target = module(WInputs(w_q, torch.tensor([[0.1, 0.9]]).repeat(4, 1)))

        torch.testing.assert_close(original.mu_u, target.mu_u)
        assert not torch.allclose(original.d_mu_y, target.d_mu_y)
        assert not torch.allclose(original.w_recon, target.w_recon)


def test_chain_decodes_the_codes_the_inner_vae_chooses(example_clouds: torch.Tensor) -> None:
    """The chained model gives a cloud of the configured size under an evaluation."""
    module = CounterfactualVQVAE(VQVAE(), WAutoEncoder()).train()
    inputs = Inputs(cloud=example_clouds, condition=torch.full((4, 2), 0.5))

    outputs = module(inputs)

    assert outputs.recon.shape == example_clouds.shape
    assert outputs.quantization is None
