"""Visualize reconstructions of the autoencoder."""

import torch

from src.data_structures import Inputs
from src.datasets import get_dataset, Partitions
from src.autoencoder import CounterfactualVQVAE
from src.config_options import Experiment, ConfigAll, hydra_main
from src.visualisation import render_cloud
from drytorch import Model


def visualize_reconstructions() -> None:
    """Visualize the selected point clouds in the dataset."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user

    test_dataset = get_dataset(Partitions.train_val if cfg.final else Partitions.val)

    module = CounterfactualVQVAE().eval()
    autoencoder = Model(module, name=cfg_ae.architecture.name, device=cfg_user.device)
    autoencoder.load_state()
    extracted_clouds = list[torch.Tensor]()
    extracted_indices = list[torch.Tensor]()

    for i in cfg_user.plot.indices_to_reconstruct:
        assert i < len(test_dataset), f'Index {i} is too large for the selected dataset of length {len(test_dataset)}'
        # inference mode prevents random augmentation
        with torch.inference_mode():
            input_pc = test_dataset[i][0].cloud
            indices = test_dataset[i][0].indices
        extracted_clouds.append(input_pc)
        extracted_indices.append(indices)

    input_clouds = torch.stack(extracted_clouds).to(autoencoder.device)
    input_indices = torch.stack(extracted_indices).to(autoencoder.device)

    batch = Inputs(cloud=input_clouds, indices=input_indices)

    if cfg.user.plot.double_encoding:
        with autoencoder.module.double_encoding:
            data = module(batch)
    else:
        data = module(batch)
    np_input_cloud = input_clouds.cpu().numpy()
    np_recon = data.recon.detach().cpu().numpy()
    for input_cloud, recon, i in zip(np_input_cloud, np_recon, cfg_user.plot.indices_to_reconstruct):
        render_cloud((input_cloud,), title=f'sample_{i}', interactive=cfg_user.plot.interactive)
        render_cloud((recon,), title=f'recon_{i}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Visualize reconstructions of the autoencoder."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        visualize_reconstructions()
    return


if __name__ == "__main__":
    main()
