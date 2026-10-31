"""Tests for the layers."""

import pytest
import torch
import torch.nn as nn

from src.module.layers import LinearLayer, Max, Mean, PointsConvLayer, SphereProjection, Transpose


class TestBaseLayer:
    """Test the behaviour shared by the layers."""

    def test_bias_only_without_batch_norm(self) -> None:
        """Batch normalization replaces the bias of the layer."""
        with_norm = LinearLayer(4, 2, act_cls=None)
        without_norm = LinearLayer(4, 2, act_cls=None, batch_norm=False)

        assert with_norm.module.get_parameter('weight') is not None
        assert with_norm.module.bias is None
        assert without_norm.module.bias is not None

    @pytest.mark.parametrize(('in_dim', 'out_dim'), [(8, 8), (8, 4), (4, 8)])
    def test_residual_repeats_the_input_channels(self, in_dim: int, out_dim: int) -> None:
        """With the weights at zero the output is the input, each channel repeated and truncated."""
        layer = PointsConvLayer(in_dim, out_dim, act_cls=None, batch_norm=False, residual=True)
        nn.init.zeros_(layer.module.get_parameter('weight'))
        nn.init.zeros_(layer.module.get_parameter('bias'))
        x = torch.randn(2, in_dim, 5)

        y = layer(x)

        expected = x.repeat_interleave(out_dim // in_dim + 1, 1)[:, :out_dim]
        torch.testing.assert_close(y, expected)

    def test_final_layer_starts_close_to_zero(self) -> None:
        """A layer ending in the identity has small initial weights."""
        layer = LinearLayer(64, 64, act_cls=nn.Identity, batch_norm=False)

        assert layer.module.get_parameter('weight').abs().max() < 0.05


def test_pooling_and_transpose() -> None:
    """The reductions remove their dimension and the transposition swaps two."""
    x = torch.rand(2, 3, 4)

    assert Max(dim=2)(x).shape == (2, 3)
    assert Mean(dim=2)(x).shape == (2, 3)
    assert Transpose(2, 1)(x).shape == (2, 4, 3)


def test_sphere_projection() -> None:
    """Every point ends up at unit distance from the origin."""
    x = torch.randn(2, 8, 16)

    projected = SphereProjection()(x)

    torch.testing.assert_close(projected.norm(dim=1), torch.ones(2, 16))
