"""Generate random samples from the autoencoder."""

import torch

from drytorch import Model

from src.module import CounterfactualVQVAE
from src.config import ConfigAll, Experiment, hydra_main
from src.utils.visualization import render_cloud


torch.inference_mode()


def generate_random_samples() -> None:
    """Generate random samples from the autoencoder."""
    cfg = Experiment.get_config()
    n_classes = cfg.data.dataset.n_classes
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    cfg_generate = cfg_user.generate
    save_dir = cfg.user.path.version_dir / 'images' / cfg.name / 'generated'

    module = CounterfactualVQVAE().eval()
    model = Model(module, name=cfg_ae.architecture.name, device=cfg_user.device)
    model.load_state()
    if module.w_autoencoder.pseudo_manager is not None:
        module.w_autoencoder.pseudo_manager.update_pseudo_latent(module.w_autoencoder.encode)

    z1_bias = torch.zeros(
        cfg_generate.batch_size, cfg_ae.architecture.n_codes, cfg_ae.architecture.z1_dim, device=cfg_user.device
    )
    probs = torch.ones(cfg_generate.batch_size, n_classes, device=cfg_user.device) // n_classes
    clouds = module.generate(batch_size=cfg_generate.batch_size, z1_bias=z1_bias, probs=probs).recon
    cloud: torch.Tensor
    for i, cloud in enumerate(clouds):
        np_cloud = cloud.cpu().numpy()
        render_cloud((np_cloud,), title=str(i), interactive=cfg_user.plot.interactive, save_dir=save_dir)


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Generate random samples from the autoencoder."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        generate_random_samples()
    return


if __name__ == '__main__':
    main()
