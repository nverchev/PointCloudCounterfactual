"""Generate random samples from the autoencoder."""

import torch

from drytorch import Model

from src.module import DiffusionModel
from src.config import AllConfig, Experiment, hydra_main
from src.utils.visualization import render_cloud


@torch.inference_mode()
def generate_random_samples() -> None:
    """Generate random samples from the autoencoder."""
    cfg = Experiment.get_config()
    cfg_diff = cfg.diffusion
    cfg_user = cfg.user
    save_dir = cfg.user.path.version_dir / 'images' / cfg.name / 'generated'
    module = DiffusionModel().eval()

    model = Model(module, name=cfg_diff.model.name, device=cfg_user.device)
    model.load_state()
    clouds = module.sample(n_samples=1, n_points=2048, device=model.device)
    cloud: torch.Tensor
    for i, cloud in enumerate(clouds):
        np_cloud = cloud.cpu().numpy().squeeze()
        render_cloud((np_cloud,), title=str(i), interactive=cfg_user.plot.interactive, save_dir=save_dir)

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Generate random samples from the autoencoder."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        generate_random_samples()

    return


if __name__ == '__main__':
    main()
