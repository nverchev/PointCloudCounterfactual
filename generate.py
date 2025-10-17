"""Generate random samples from the autoencoder."""

import torch

from src.autoencoder import CounterfactualVQVAE
from src.config_options import Experiment, hydra_main, ConfigAll
from src.visualisation import render_cloud
from drytorch import Model


def generate_random_samples():
    """Generate random samples from the autoencoder."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    cfg_generate = cfg_user.generate
    module = CounterfactualVQVAE().eval()
    model = Model(module, name=cfg_ae.architecture.name, device=cfg_user.device)
    model.load_state()
    if module.w_autoencoder.pseudo_manager is not None:
        module.w_autoencoder.pseudo_manager.update_pseudo_latent()

    z1_bias = torch.zeros(cfg_generate.batch_size, cfg_ae.architecture.z1_dim, device=cfg_user.device)
    clouds = module.generate(batch_size=cfg_generate.batch_size, z1_bias=z1_bias).recon
    for cloud in clouds:
        np_cloud = cloud.cpu().numpy()
        render_cloud((np_cloud,), title='generated_{i}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Generate random samples from the autoencoder."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        generate_random_samples()
    return


if __name__ == "__main__":
    main()
