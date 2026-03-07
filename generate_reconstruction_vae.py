"""Generate VAE point cloud reconstructions, and visualize them together with the original point clouds."""

import logging
from collections.abc import Sized

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.train.models import load_extract_autoencoder_module
from src.utils.visualization import render_cloud


@torch.inference_mode()
def create_and_render_reconstructions() -> None:
    """Create and visualize the reconstructions using the VAE model."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user

    interactive = cfg_user.plot.interactive
    save_dir_base = cfg_user.path.version_dir / 'images' / cfg.name / 'vae_reconstruction'

    # 1. Load Models
    logging.info('Loading autoencoder...')
    ema_module = load_extract_autoencoder_module()
    device = ema_module.device

    # 2. Collect Batch Data
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    test_dataset = get_dataset(partition)
    class_names = test_dataset.class_names

    clouds = []
    labels = []

    for i in cfg_user.plot.sample_indices:
        assert isinstance(test_dataset, Sized)
        if i >= len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')

        sample_i, target_i = test_dataset[i]
        clouds.append(sample_i.cloud)
        labels.append(target_i.label)

    # 3. Batch Inference
    batch_cloud = torch.stack(clouds).to(device)
    inputs = Inputs(cloud=batch_cloud)

    # VAE Reconstruction
    out_recon = ema_module(inputs)
    recon_clouds = out_recon.recon.detach().cpu().numpy()

    # 4. Visualization and Reporting
    for idx, sample_idx in enumerate(cfg_user.plot.sample_indices):
        label_idx = labels[idx]
        label_name = class_names[label_idx]
        logging.info('Sample %d with label "%s" (%d):', sample_idx, label_name, label_idx)
        save_dir = save_dir_base / f'sample_{sample_idx}'
        save_dir.mkdir(parents=True, exist_ok=True)
        for old_file in save_dir.iterdir():
            old_file.unlink()

        # Original
        original_title = f'Original (Sample {sample_idx}, Label: {label_name})'
        render_cloud(
            (batch_cloud[idx].cpu().numpy(),),
            title=original_title,
            interactive=interactive,
            save_dir=save_dir,
        )

        # Reconstruction
        recon_title = f'VAE Reconstruction (Sample {sample_idx}, Label: {label_name})'
        render_cloud(
            (recon_clouds[idx],),
            title=recon_title,
            interactive=interactive,
            save_dir=save_dir,
        )

        # Combined
        render_cloud(
            (batch_cloud[idx].cpu().numpy(), recon_clouds[idx]),
            title='Original vs Reconstruction',
            interactive=interactive,
            save_dir=save_dir,
        )
        print()

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the VAE reconstruction visualization."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_and_render_reconstructions()

    return


if __name__ == '__main__':
    main()
