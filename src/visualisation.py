import pathlib
from typing import Literal, Optional, Sequence

import numpy as np
from numpy import typing as npt
import torch
import pyvista as pv

from src.config_options import ExperimentAE
from src.data_structures import Inputs
from src.autoencoder import AutoEncoder

BLUE = np.array([0.3, 0.3, 0.9])
RED = np.array([0.9, 0.3, 0.3])
GREEN = np.array([0.3, 0.9, 0.3])
VIOLET = np.array([0.6, 0.0, 0.9])
ORANGE = np.array([0.9, 0.6, 0.0])
COLOR_TUPLE = (BLUE, RED, GREEN, VIOLET, ORANGE)


def render_scan(clouds: Sequence[npt.NDArray],
                colorscale: Literal['blue_red', 'sequence'] = 'sequence',
                interactive: bool = True,
                title: str = 'Cloud',
                save_dir: pathlib.Path = pathlib.Path() / 'images',
                ) -> None:
    plotter = pv.Plotter(lighting='three lights',
                         window_size=(1024, 1024),
                         notebook=False,
                         off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((1.8, 0, 0), focal_point=(0, 0, 0), viewup=(0, 0, 1))

    for light_point in ((30, -30, 2), (30, 30, 20)):
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

        n = cloud.shape[0]
        # cloud_pv = pv.PolyData(cloud[:, :3])
        # geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
        # cloud_pv["radius"] = .2 * np.ones(n)
        # glyphs = cloud_pv.glyph(scale='radius', geom=geom, orient=False)
        plotter.add_mesh(pv.PolyData(cloud[:, :3]),
                         color=color,
                         point_size=15,
                         render_points_as_spheres=True,
                         smooth_shading=True)
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
    plotter = pv.Plotter(lighting='three lights',
                         window_size=(1024, 1024),
                         notebook=False,
                         off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((0, 1, 0), focal_point=(0, 0, 0), viewup=(0, 0, 1))

    # later lightning
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
            color = COLOR_TUPLE[i]
        else:
            raise ValueError('Colorscale not available')

        n = cloud.shape[0]
        cloud_pv = pv.PolyData(cloud[:, :3])
        geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
        cloud_pv["radius"] = .01 * np.ones(n)
        glyphs = cloud_pv.glyph(scale='radius', geom=geom, orient=False)
        plotter.add_mesh(glyphs, color=color, point_size=15, render_points_as_spheres=True, smooth_shading=True)
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
    cfg = ExperimentAE.get_config()
    cfg_ae = cfg.autoencoder
    n_clouds = len(input_pc) if input_pc is not None else n_clouds
    if n_clouds is None:
        raise ValueError('Number of clouds must be provided.')
    s = torch.randn(n_clouds, cfg_ae.decoder.sample_dim, cfg.autoencoder.output_points, device=cfg.user.device)
    att = torch.empty(n_clouds, cfg.autoencoder.output_points, cfg_ae.decoder.n_components, device=cfg.user.device)
    components = torch.empty(n_clouds, 3, cfg.autoencoder.output_points, cfg_ae.decoder.n_components)
    # if cfg.add_viz == 'sampling_loop':
    #     bbox1 = torch.eye(cfg.sample_dim, device=cfg.device, dtype=torch.float32)
    #     bbox2 = -torch.eye(cfg.sample_dim, device=cfg.device, dtype=torch.float32)
    #     bbox = torch.cat((bbox1, bbox2), dim=1).unsqueeze(0).expand(n_clouds, -1, -1)
    #     s = torch.cat([s] + [t * bbox.roll(1, dims=2) + (1 - t) * bbox for t in torch.arange(0, 1, 0.03)], dim=2)
    #     model.decoder.filtering = False
    # elif cfg.add_viz == 'filter':
    #     model.decoder.filtering = False
    # elif cfg.add_viz == 'none':
    #     pass
    # elif cfg.add_viz:
    #     raise ValueError(f'{cfg.add_viz} is not a recognized argument')
    if mode == 'recon':
        assert z_bias is None
        assert input_pc is not None
        with torch.inference_mode():
            model.eval()
            inputs = Inputs(cloud=input_pc, initial_sampling=s, viz_att=att, viz_components=components)
            samples_and_loop = model(inputs)
    elif mode == 'gen':
        assert z_bias is not None
        assert input_pc is None
        samples_and_loop = model.random_sampling(n_clouds, s, att, components, z_bias)
    else:
        raise ValueError('Mode can only be "recon" or "gen".')
    samples_and_loop = samples_and_loop.recon.cpu()
    samples, *loops = samples_and_loop.split(cfg.autoencoder.output_points, dim=1)

    def naming_syntax(num: int, viz_name: Optional[str] = None) -> str:
        # if mode == 'recon':
        #     num = cfg.viz[num]
        viz_name_list = [viz_name] if viz_name is not None else []
        return '_'.join([mode] + viz_name_list + [str(num)])

    for i, sample in enumerate(samples):
        sample_np: npt.NDArray = sample.numpy()
        # if cfg.add_viz == 'sampling_loop':
        #     sample_name = naming_syntax(i, 'sampling_loop')
        #     render_cloud((sample_np, loops[0][i].numpy()), title=sample_name, interactive=cfg.interactive_plot)
        # elif cfg.add_viz == 'components':
        #     threshold = 0.  # boundary points shown in blue
        #     att_max, att_argmax = att[i].max(dim=1)
        #     indices = (att_argmax.cpu() + 1) * (att_max > threshold).bool().cpu()
        #     pc_list = [sample_np[indices == component] for component in range(cfg.n_components + 1)]
        #     sample_name = naming_syntax(i, 'attention')
        #     render_cloud(pc_list, title=sample_name, interactive=cfg.interactive_plot)
        #     component = components[i].cpu().transpose(1, 0)
        #     components_cloud = [np.empty(0)]
        #     for j, j_component in enumerate(component.unbind(2)):
        #         j_component = j_component + torch.FloatTensor([[(1 - cfg.n_components) / 2 + j, 0, 0]])
        #         components_cloud.append(j_component.numpy() / cfg.n_components)
        #     sample_name = naming_syntax(i, 'components')
        #     render_cloud(components_cloud, title=sample_name, interactive=cfg.interactive_plot)
        # elif cfg.add_viz == 'filter':
        #     filter_direction = graph_filtering(sample.transpose(0, 1).unsqueeze(0)).squeeze().transpose(0, 1)
        #     filter_arrows = filter_direction.numpy() - sample_np
        #     sample_name = naming_syntax(i, 'filter')
        #     render_cloud((sample_np,), title=sample_name, arrows=filter_arrows, interactive=cfg.interactive_plot)
        # elif cfg.add_viz == 'none':
        sample_name = naming_syntax(i)
        render_cloud((sample_np,), title=sample_name, interactive=cfg.user.plot.interactive)
        pass

# def show_latent(mu: npt.NDArray, pseudo_mu: npt.NDArray, model_name: str) -> None:
#     try:
#         import visdom
#     except ImportError:
#         print('visdom not installed. Please install it using pip: pip install visdom.')
#         return
#     pca = PCA(3)
#     test_mu_pca = pca.fit_transform(mu)
#     test_labels = np.ones(test_mu_pca.shape[0])
#     pseudo_mu_pca = pca.transform(pseudo_mu)
#     pseudo_labels = 2 * np.ones(pseudo_mu_pca.shape[0])
#     mu_pca = np.vstack((test_mu_pca, pseudo_mu_pca))
#     labels = np.hstack((test_labels, pseudo_labels))
#     title = 'Continuous Latent Space'
#     exp_name = Experiment.current().name
#     vis = visdom.Visdom(env='_'.join((exp_name, model_name)), raise_exceptions=False)
#     vis.scatter(X=mu_pca, Y=labels, win=title,
#                 opts=dict(title=title, markersize=5, legend=['Validation', 'Pseudo-Inputs']))
