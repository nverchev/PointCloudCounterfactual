import torch

from src.autoencoder import VQVAE
from src.config_options import ExperimentAE, MainExperiment, hydra_main, ConfigAll
from src.data_structures import Outputs
from src.visualisation import render_cloud
from dry_torch import Model


def generate_random_samples():
    cfg = MainExperiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    cfg_generate = cfg_user.generate
    module = VQVAE().eval()
    model = Model(module, name=cfg_ae.model.name, device=cfg_user.device)
    model.load_state()
    module.w_autoencoder.update_pseudo_latent()
    z_bias = torch.zeros(cfg_generate.batch_size, cfg_ae.model.z_dim, device=cfg_user.device)
    num_classes = cfg_ae.data.dataset.n_classes
    for i in range(num_classes):
        z_bias[:, :num_classes] = max((1 - cfg_user.generate.bias_value) / (num_classes - 1), 0)

        z_bias[:, i] = cfg_generate.bias_value

        data = Outputs()
        data.z = z_bias
        with torch.inference_mode():
            with module.double_encoding:
                clouds = module.decode(data).recon
        for cloud in clouds:
            np_cloud = cloud.cpu().numpy()
            render_cloud((np_cloud,), title='generated_{i}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    parent_experiment = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    exp_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    parent_experiment.register_child(exp_ae)
    with exp_ae:
        generate_random_samples()
    return


if __name__ == "__main__":
    main()
