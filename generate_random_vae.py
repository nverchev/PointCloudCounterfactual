"""Generate random samples from the autoencoder."""

import torch


from src.train.models import load_extract_autoencoder_module
from src.config import AllConfig, Experiment, hydra_main
from src.utils.visualization import render_cloud


@torch.inference_mode()
def generate_random_samples() -> None:
    """Generate random samples from the autoencoder."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    cfg_generate = cfg_user.generate
    save_dir = cfg.user.path.version_dir / 'images' / cfg.name / 'generated_vae'
    ema_module = load_extract_autoencoder_module()
    if ema_module.pseudo_manager is not None:
        ema_module.pseudo_manager.update_pseudo_latent(ema_module.encode_z1)

    z1_bias = torch.zeros(cfg_generate.batch_size, cfg_ae.model.z1_dim, device=cfg_user.device)
    clouds = ema_module.generate(batch_size=cfg_generate.batch_size, z1_bias=z1_bias).recon
    cloud: torch.Tensor
    for i, cloud in enumerate(clouds):
        np_cloud = cloud.cpu().numpy()
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
