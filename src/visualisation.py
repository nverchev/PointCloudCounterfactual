"""A module for visualizing point clouds and latent spaces using PyVista and Visdom."""

import pathlib
from typing import Literal, Optional, Sequence, TYPE_CHECKING

import numpy as np
from numpy import typing as npt
import torch
from sklearn.decomposition import PCA  # type: ignore

from src.config_options import Experiment
from src.data_structures import Inputs
from src.autoencoder import AutoEncoder

BLUE = np.array([0.3, 0.3, 0.9])
RED = np.array([0.9, 0.3, 0.3])
GREEN = np.array([0.3, 0.9, 0.3])
VIOLET = np.array([0.6, 0.0, 0.9])
ORANGE = np.array([0.9, 0.6, 0.0])
COLOR_TUPLE = (BLUE, RED, GREEN, VIOLET, ORANGE)


if TYPE_CHECKING:
    import matplotlib.pyplot as plt
else:
    plt = None

def render_scan(clouds: Sequence[npt.NDArray],
                colorscale: Literal['blue_red', 'sequence'] = 'sequence',
                interactive: bool = True,
                title: str = 'Cloud',
                save_dir: pathlib.Path = pathlib.Path() / 'images',
                ) -> None:
    """Renders a sequence of point clouds using a scan-like visualization with PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        print('pyvista not installed. Please install it using pip: pip install pyvista.')
        return

    plotter = pv.Plotter(lighting='three lights',
                         window_size=(1024, 1024),
                         notebook=False,
                         off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((-1, 1, 1.5), focal_point=(0, 0, 0), viewup=(0, 0, 1))

    for light_point in ((3, -3, 2), (3, 3, 2)):
        light = pv.Light(position=light_point, focal_point=(0, 0, 0), intensity=1, positional=True)
        plotter.add_light(light)

    for i, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        if colorscale == 'blue_red':
            i_norm = i / (len(clouds) - 1)
            color = (1 - i_norm) * BLUE + i_norm * RED
        elif colorscale == 'sequence':
            if len(clouds) > len(COLOR_TUPLE):
                raise ValueError('Color scale sequence too short for the number of point clouds.')
            color = COLOR_TUPLE[i]
        else:
            raise ValueError('Colorscale not available.')

        plotter.add_mesh(pv.PolyData(cloud[:, :3]),
                         color=color,
                         point_size=15,
                         render_points_as_spheres=True,
                         smooth_shading=True,
                         show_edges=False)
    # effects
    plotter.enable_eye_dome_lighting()
    plotter.enable_shadows()

    if interactive:
        plotter.set_background(color='white')
        plotter.show()
    else:
        save_dir.mkdir(exist_ok=True)
        save_name = save_dir / title
        plotter.screenshot(save_name.with_suffix('.png'), window_size=(1024, 1024), transparent_background=True)
    plotter.close()
    return


def render_cloud(clouds: Sequence[npt.NDArray],
                 colorscale: Literal['blue_red', 'sequence'] = 'sequence',
                 interactive: bool = True,
                 arrows: Optional[torch.Tensor] = None,
                 title: str = 'Cloud',
                 save_dir: pathlib.Path = pathlib.Path() / 'images',
                 ) -> None:
    """Renders a sequence of point clouds with optional arrows using PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        print('pyvista not installed. Please install it using pip: pip install pyvista.')
        return

    plotter = pv.Plotter(lighting='three lights',
                         window_size=(1024, 1024),
                         notebook=False,
                         off_screen=not interactive)
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
        cloud_pv["radius"] = .01 * np.ones(n)
        glyphs = cloud_pv.glyph(scale='radius', geom=geom, orient=False)
        plotter.add_mesh(glyphs,
                         color=color,
                         point_size=15,
                         render_points_as_spheres=True,
                         smooth_shading=True,
                         show_edges=False,
                         style='points'
                         )
        if arrows is not None:
            geom = pv.Arrow(shaft_radius=.1, tip_radius=.2, scale=1)
            cloud_pv["vectors"] = arrows[:, [0, 2, 1]].numpy()
            cloud_pv.set_active_vectors("vectors")
            arrows_glyph = cloud_pv.glyph(orient="vectors", geom=geom)
            plotter.add_mesh(arrows_glyph,
                             lighting=True,
                             line_width=10,
                             color=RED,
                             show_scalar_bar=False,
                             edge_color=RED)
    # effects
    plotter.enable_eye_dome_lighting()
    plotter.enable_shadows()

    if interactive:
        plotter.set_background(color='white')
        plotter.show()
    else:
        save_dir.mkdir(exist_ok=True)
        save_name = save_dir / title
        plotter.screenshot(save_name.with_suffix('.png'), window_size=(1024, 1024), transparent_background=True)
    plotter.close()
    return


def infer_and_visualize(model: AutoEncoder,
                        n_clouds: Optional[int] = None,
                        mode: Literal['recon', 'gen'] = 'recon',
                        z_bias: Optional[torch.Tensor] = None,
                        input_pc: Optional[torch.Tensor] = None) -> None:
    """Performs inference with an autoencoder and visualizes the results."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    cfg_ae_arc = cfg.autoencoder.architecture
    n_clouds = len(input_pc) if input_pc is not None else n_clouds
    if n_clouds is None:
        raise ValueError('Number of clouds must be provided.')
    s = torch.randn(n_clouds, cfg_ae_arc.decoder.sample_dim, cfg_ae_arc.training_output_points, device=cfg_user.device)
    att = torch.empty(
        n_clouds, cfg_ae_arc.training_output_points, cfg_ae_arc.decoder.n_components, device=cfg_user.device
    )
    components = torch.empty(n_clouds, 3, cfg_ae_arc.training_output_points, cfg_ae_arc.decoder.n_components)

    if mode == 'recon':
        assert z_bias is None
        assert input_pc is not None
        with torch.inference_mode():
            model.eval()
            inputs = Inputs(cloud=input_pc, initial_sampling=s)
            samples_and_loop = model(inputs)
    elif mode == 'gen':
        assert z_bias is not None
        assert input_pc is None
        samples_and_loop = model.random_sampling(n_clouds, s, att, components, z_bias)
    else:
        raise ValueError('Mode can only be "recon" or "gen".')
    samples_and_loop = samples_and_loop.recon.cpu()
    samples, *loops = samples_and_loop.split(cfg_ae_arc.training_output_points, dim=1)

    def _naming_syntax(num: int, viz_name: Optional[str] = None) -> str:
        viz_name_list = [viz_name] if viz_name is not None else []
        return '_'.join([mode] + viz_name_list + [str(num)])

    for i, sample in enumerate(samples):
        sample_np: npt.NDArray = sample.numpy()
        sample_name = _naming_syntax(i)
        render_cloud((sample_np,), title=sample_name, interactive=cfg_user.plot.interactive)
        pass


def show_latent(mu: npt.NDArray, pseudo_mu: npt.NDArray, model_name: str) -> None:
    """Visualizes latent space embeddings using Visdom."""
    try:
        import visdom
    except ImportError:
        print('visdom not installed. Please install it using pip: pip install visdom.')
        return
    pca = PCA(3)
    test_mu_pca = pca.fit_transform(mu)
    test_labels = np.ones(test_mu_pca.shape[0])
    pseudo_mu_pca = pca.transform(pseudo_mu)
    pseudo_labels = 2 * np.ones(pseudo_mu_pca.shape[0])
    mu_pca = np.vstack((test_mu_pca, pseudo_mu_pca))
    labels = np.hstack((test_labels, pseudo_labels))
    title = 'Continuous Latent Space'
    exp_name = Experiment.get_config().name
    vis = visdom.Visdom(env='_'.join((exp_name, model_name)), raise_exceptions=False)
    vis.scatter(X=mu_pca, Y=labels, win=title,
                opts=dict(title=title, markersize=5, legend=['Validation', 'Pseudo-Inputs']))


def plot_confusion_matrix_heatmap(cm_array: npt.NDArray,
                                  class_names: list[str],
                                  title: str = 'Confusion Matrix',
                                  dpi: int = 300) -> plt.Figure | None:
    """Plots the confusion matrix as a heatmap using Matplotlib and Seaborn."""
    try:
        import seaborn as sns
        from matplotlib import pyplot as plt
    except ImportError:
        return None

    num_classes = len(class_names)
    plt.figure(figsize=(num_classes + 2, num_classes + 2), dpi=dpi)
    sns.heatmap(cm_array,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=.5,
                linecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    return plt.gcf()  # Get the current matplotlib figure to log
