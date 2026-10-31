"""Tests for the composition of the entry configurations."""

import pytest

from src.config import get_experiment_config


@pytest.mark.parametrize(
    ('config_name', 'n_classes', 'selected'),
    [
        ('modelnet_dt', 2, ('desk', 'table')),
        ('modelnet_bbcv', 4, ('bottle', 'bowl', 'cup', 'vase')),
        ('modelnet40', 40, ()),
    ],
)
def test_entry_configurations_select_their_classes(config_name: str, n_classes: int, selected: tuple[str, ...]) -> None:
    """Each entry configuration composes and selects the classes of its dataset."""
    cfg = get_experiment_config(config_name)

    assert cfg.data.dataset.n_classes == n_classes
    assert cfg.data.dataset.selected_classes == selected


def test_published_latent_split() -> None:
    """The latent space of the inner VAE is divided in two equal parts."""
    cfg = get_experiment_config()

    assert cfg.w_autoencoding.model.z_u_dim == 16
    assert cfg.w_autoencoding.model.z_y_dim == 16
    assert cfg.autoencoding.model.dim_codes == 256


def test_mismatched_classes_are_rejected() -> None:
    """A selection that disagrees with the number of classes fails the validation."""
    with pytest.raises(ValueError, match='do not match'):
        get_experiment_config(overrides=['data.dataset.n_classes=3'])
