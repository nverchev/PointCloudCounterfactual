"""Rendering of point clouds."""

import pathlib

from typing import Final, cast

import numpy as np
import pyvista as pv
import torch

__all__ = ['render_cloud']

COLOR: Final = (0.25, 0.25, 0.9)
"""Color of the rendered points."""

RADIUS: Final = 0.05
"""Radius of the sphere drawn at every point."""


def render_cloud(cloud: torch.Tensor, /, *, name: str, save_dir: pathlib.Path, show: bool) -> None:
    """Render a point cloud with PyVista and save the image.

    Args:
        cloud: coordinates ``(N, IN_CHAN)``.
        name: name of the image file, without its suffix.
        save_dir: directory where the image is saved.
        show: whether a window displays the cloud.

    Side Effects:
        Creates the directory and writes the image.
    """
    plotter = pv.Plotter(lighting='three lights', window_size=[1024, 1024], off_screen=not show)
    plotter.camera_position = pv.CameraPosition((-3.0, -2.1, 1.1), focal_point=(0, 0, 0), viewup=(0, 0, 1))
    poly = pv.PolyData(cloud[:, [0, 2, 1]].detach().cpu().numpy())
    poly['radius'] = RADIUS * np.ones(cloud.shape[0])
    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    glyphs = cast(pv.PolyData, poly.glyph(scale='radius', geom=sphere, orient=False))
    plotter.add_mesh(glyphs, color=COLOR, smooth_shading=True)
    if show:
        plotter.background_color = pv.Color('white')
        plotter.show(auto_close=False)

    save_dir.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(save_dir / f'{name}.png'), window_size=[1024, 1024], transparent_background=True)
    plotter.close()
    return
