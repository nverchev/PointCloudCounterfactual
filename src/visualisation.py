"""A module for visualizing point clouds and latent spaces using PyVista and Visdom."""

import pathlib

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from numpy import typing as npt


BLUE = np.array([0.3, 0.3, 0.9])
RED = np.array([0.9, 0.3, 0.3])
GREEN = np.array([0.3, 0.9, 0.3])
VIOLET = np.array([0.6, 0.0, 0.9])
ORANGE = np.array([0.9, 0.6, 0.0])
COLOR_TUPLE = (BLUE, RED, GREEN, VIOLET, ORANGE)


if TYPE_CHECKING:
    from matplotlib.pyplot import Figure
else:
    Figure = Any


def render_cloud(
    clouds: Sequence[npt.NDArray[Any]],
    colorscale: Literal['blue_red', 'sequence'] = 'sequence',
    interactive: bool = True,
    arrows: torch.Tensor | None = None,
    title: str = 'Cloud',
    save_dir: pathlib.Path = pathlib.Path() / 'images',
) -> None:
    """Renders a sequence of point clouds with optional arrows using PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        print('pyvista not installed. Please install it using pip: pip install pyvista.')
        return

    plotter = pv.Plotter(lighting='three lights', window_size=[1024, 1024], notebook=False, off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((-3, 1, -2.5), focal_point=(0, 0, 0), viewup=(0, 1, 0))

    for light_point in ((3, 3, -2), (3, 3, 2)):
        light = pv.Light(position=light_point, focal_point=(0, 0, 0), intensity=1, positional=True)
        plotter.add_light(light)

    for i, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        if colorscale == 'blue_red':
            i_norm = i / (len(clouds) - 1)
            color = (1 - i_norm) * BLUE + i_norm * RED
        elif colorscale == 'sequence':
            color = COLOR_TUPLE[i]
        else:
            raise ValueError('Colorscale not available')

        n = cloud.shape[0]
        cloud_pv = pv.PolyData(cloud[:, :3])
        geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
        cloud_pv['radius'] = 0.01 * np.ones(n)
        glyphs = cast(pv.PolyData, cloud_pv.glyph(scale='radius', geom=geom, orient=False))
        plotter.add_mesh(
            glyphs,
            color=color,
            point_size=15,
            render_points_as_spheres=True,
            smooth_shading=True,
            show_edges=False,
            style='points',
        )
        if arrows is not None:
            geom = pv.Arrow(shaft_radius=0.1, tip_radius=0.2, scale=1)
            cloud_pv['vectors'] = arrows[:, [0, 2, 1]].numpy()
            cloud_pv.set_active_vectors('vectors')
            arrows_glyph = cast(pv.MultiBlock, cloud_pv.glyph(orient='vectors', geom=geom))
            plotter.add_mesh(
                arrows_glyph, lighting=True, line_width=10, color=RED, show_scalar_bar=False, edge_color=RED
            )
    # effects
    pv.Plotter.enable_eye_dome_lighting(plotter)  # call unbound because of a type inference bug
    pv.Plotter.enable_shadows(plotter)  # call unbound because of a type inference bug

    if interactive:
        pv.Plotter.set_background(plotter, color='white')  # call unbound because of a type inference bug
        plotter.show()
    else:
        save_dir.mkdir(exist_ok=True, parents=True)
        save_name = save_dir / title
        plotter.screenshot(save_name.with_suffix('.png'), window_size=(1024, 1024), transparent_background=True)
    plotter.close()
    return


def plot_confusion_matrix_heatmap(
    cm_array: npt.NDArray[Any], class_names: list[str], title: str = 'Confusion Matrix', dpi: int = 300
) -> Figure | None:
    """Plots the confusion matrix as a heatmap using Matplotlib and Seaborn."""
    try:
        import seaborn as sns

        from matplotlib import pyplot as plt
    except ImportError:
        return None

    num_classes = len(class_names)
    plt.figure(figsize=(num_classes + 2, num_classes + 2), dpi=dpi)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='black',
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    return plt.gcf()  # Get the current matplotlib figure to log
