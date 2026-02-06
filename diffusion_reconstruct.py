"""Visualize diffusion autoencoder reconstructions."""

import torch

from drytorch import Model

from src.module import get_diffusion_module, DiffusionAutoencoder
from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.utils.visualization import render_cloud


@torch.inference_mode()
def create_and_render_reconstructions() -> None:
    """Create and visualize the selected point clouds in the dataset."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    cfg_diff = cfg.diffusion
    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / 'images' / cfg.name / 'reconstructions'

    module = get_diffusion_module().eval()
    if not isinstance(module, DiffusionAutoencoder):
        raise TypeError('Model must be a DiffusionAutoencoder')

    model = Model(module, name=cfg_diff.model.name, device=cfg_user.device)
    model.load_state()
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)

    # Force re-computation of schedule buffers to fix the instability (0.9999 -> 0.999)
    # This is necessary because load_state overwrites buffers with the bad values from training checkpoint.
    # The source code in src/module/diffusion.py has been updated to clip at 0.999, so calling this method works.
    device = model.device
    betas = module._get_beta_schedule().to(device)
    alphas = 1.0 - betas
    alphas_cum_prod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cum_prod = torch.sqrt(alphas_cum_prod)
    sqrt_one_minus_alphas_cum_prod = torch.sqrt(1.0 - alphas_cum_prod)

    module.register_buffer('betas', betas)
    module.register_buffer('alphas', alphas)
    module.register_buffer('alphas_cum_prod', alphas_cum_prod)
    module.register_buffer('sqrt_alphas_cum_prod', sqrt_alphas_cum_prod)
    module.register_buffer('sqrt_one_minus_alphas_cum_prod', sqrt_one_minus_alphas_cum_prod)

    print('Re-computed diffusion schedule with max beta:', betas.max().item())

    # Collect samples for batch processing
    indices = cfg_user.plot.sample_indices
    clouds = []
    labels = []

    for i in indices:
        if i >= len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')
        sample_data = test_dataset[i]
        clouds.append(sample_data[0].cloud)
        labels.append(sample_data[1].label)

    # Batch inference
    batch_clouds = torch.stack(clouds).to(model.device)
    batch_inputs = Inputs(cloud=batch_clouds)

    # Check Input Stats
    print('\nInput Stats (Batch):')
    print(f'  Min: {batch_clouds.min().item():.4f}, Max: {batch_clouds.max().item():.4f}')
    print(f'  Mean: {batch_clouds.mean().item():.4f}, Std: {batch_clouds.std().item():.4f}')

    # Reconstruction
    z = module.encode(batch_inputs)

    # Check Latent Stats
    print('\nLatent z Stats:')
    print(f'  Shape: {z.shape}')
    print(f'  Min: {z.min().item():.4f}, Max: {z.max().item():.4f}')
    print(f'  Mean: {z.mean().item():.4f}, Std: {z.std().item():.4f}')

    recon_clouds = module.decode(z, n_points=batch_clouds.shape[1], device=model.device)

    # Check Recons Stats
    print('\nReconstruction Stats:')
    print(f'  Min: {recon_clouds.min().item():.4f}, Max: {recon_clouds.max().item():.4f}')
    print(f'  Mean: {recon_clouds.mean().item():.4f}, Std: {recon_clouds.std().item():.4f}')

    # Visualization loop
    for k, i in enumerate(indices):
        save_dir = save_dir_base / f'sample_{i}'
        save_dir.mkdir(parents=True, exist_ok=True)
        for old_file in save_dir.iterdir():
            old_file.unlink()

        label_i = labels[k]
        print(f'Sample {i} with label {label_i}:')

        input_pc = batch_clouds[k].cpu().numpy()
        recon_pc = recon_clouds[k].detach().cpu().numpy()

        render_cloud((input_pc,), title='Original', interactive=interactive, save_dir=save_dir)
        render_cloud((recon_pc,), title='Reconstruction', interactive=interactive, save_dir=save_dir)

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the reconstruction visualization."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_and_render_reconstructions()

    return


if __name__ == '__main__':
    main()
