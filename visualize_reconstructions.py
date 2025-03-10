import torch
import numpy.typing as npt

from src.data_structures import Inputs
from src.datasets import get_dataset, Partitions
from src.autoencoder import VQVAE
from src.config_options import ExperimentAE, MainExperiment, ConfigAll, hydra_main
from src.visualisation import render_cloud
from dry_torch import Model


def visualize_reconstructions() -> None:
    cfg = MainExperiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)

    module = VQVAE().eval()
    model = Model(module, name=cfg_ae.model.name, device=cfg_user.device)
    model.load_state()

    extracted_clouds = list[torch.Tensor]()
    extracted_indices = list[torch.Tensor]()

    for i in cfg_user.plot.indices_to_reconstruct:
        assert i < len(test_dataset), 'Index is too large for the selected dataset'
        # inference mode prevents random augmentation
        with torch.inference_mode():
            input_pc = test_dataset[i][0].cloud
            indices = test_dataset[i][0].indices
        extracted_clouds.append(input_pc)
        extracted_indices.append(indices)

    input_clouds = torch.stack(extracted_clouds).to(model.device)
    input_indices = torch.stack(extracted_indices).to(model.device)

    batch = Inputs(cloud=input_clouds, indices=input_indices)

    if cfg.user.plot.double_encoding:
        with model.module.double_encoding:
            data = module(batch)
    else:
        data = module(batch)
    np_input_cloud = input_clouds.cpu().numpy()
    np_recon = data.recon.detach().cpu().numpy()
    for input_cloud, recon, i in zip(np_input_cloud, np_recon, cfg_user.plot.indices_to_reconstruct):

        render_cloud((input_cloud, ), title=f'sample_{i}', interactive=cfg_user.plot.interactive)
        render_cloud((recon, ), title=f'recon_{i}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    parent_experiment = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    exp_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    parent_experiment.register_child(exp_ae)
    with exp_ae:
        visualize_reconstructions()
    return


if __name__ == "__main__":
    main()
