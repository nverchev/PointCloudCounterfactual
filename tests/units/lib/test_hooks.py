"""Tests for the training hooks."""

import pytest
import torch

from src.data.structures import Outputs, Quantization
from src.lib.hooks import ReviveCodebook
from src.module import VQVAE


class TestReviveCodebook:
    """Test what happens to the codes no sample chose."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        """Set up a diagnostic where every group uses only its first code."""
        self.module = VQVAE()
        one_hot_idx = torch.zeros(2, self.module.dim_codes, self.module.book_size)
        one_hot_idx[:, :, 0] = 1
        quantization = Quantization(torch.empty(0), torch.empty(0), one_hot_idx)
        self.diagnostic = mocker.Mock(outputs_list=[Outputs(torch.empty(0), quantization)])
        self.trainer = mocker.Mock()
        return

    def _run(self, *, epoch: int, noise: float) -> None:
        hook = ReviveCodebook(self.diagnostic, self.module, noise=noise, final_epoch=10)
        self.trainer.model.epoch = epoch
        hook._trainer = self.trainer
        hook.run()
        return

    def test_unused_codes_move_onto_the_used_ones(self) -> None:
        """Before the last epoch, an unused code becomes a copy of a used one."""
        self._run(epoch=5, noise=0.0)

        codebook = self.module.codebook.data
        torch.testing.assert_close(codebook[:, 1], codebook[:, 0])

    def test_no_noise_at_the_last_epoch(self) -> None:
        """At the end, the unused codes copy a used one exactly, whatever the noise."""
        self._run(epoch=10, noise=2.0)

        codebook = self.module.codebook.data
        torch.testing.assert_close(codebook[:, 1], codebook[:, 0])

    def test_noise_before_the_last_epoch(self) -> None:
        """Before the end, the noise separates an unused code from the one it copies."""
        self._run(epoch=5, noise=2.0)

        codebook = self.module.codebook.data
        assert not torch.allclose(codebook[:, 1], codebook[:, 0])

    def test_diagnostic_runs_with_stored_outputs(self) -> None:
        """The codes come from a pass that stored the outputs."""
        self._run(epoch=5, noise=0.0)

        self.diagnostic.assert_called_once_with(store_outputs=True)
