import hydra
import torch

from src.autoencoder import VQVAE
from src.config_options import set_up_experiment, get_config
from src.visualisation import infer_and_visualize
from dry_torch import Model


def generate_random_samples():
    cfg = get_config()
    cfg_generate = cfg.user.generate
    module = VQVAE()
    model = Model(module, name=cfg.model.name, device=cfg.user.device)
    model.load_state(-1)
    module.w_encoder.update_pseudo_latent()
    z_bias = torch.zeros(cfg_generate.batch_size, cfg.model.z_dim, device=cfg.user.device)
    z_bias[:, cfg_generate.bias_dim] = cfg_generate.bias_value
    infer_and_visualize(module, n_clouds=cfg_generate.batch_size, mode='gen', z_bias=z_bias)


def main() -> None:
    with hydra.initialize(version_base=None, config_path="hydra_conf/autoencoder_conf/"):
        dict_cfg = hydra.compose(config_name="defaults")
        set_up_experiment(dict_cfg, output=False)
        generate_random_samples()
    return


if __name__ == "__main__":
    main()
