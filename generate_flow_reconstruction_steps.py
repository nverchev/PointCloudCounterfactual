"""Self-standing script for full flow reconstruction and partial reconstruction debugging."""

import logging

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.train.models import (
    load_extract_autoencoder_module,
    load_extract_cond_flow_module,
)
from src.utils.visualization import render_cloud


def format_probs(probs: torch.Tensor, class_names: list[str]) -> str:
    """Format probability distribution for logging."""
    probs_np = probs.cpu().numpy()
    return f'({", ".join(f"{name}: {100 * p:.1f}%" for name, p in zip(class_names, probs_np, strict=True))})'


@torch.inference_mode()
def run_debug_reconstructions() -> None:
    """Run full and partial reconstructions showing all steps."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user

    interactive = cfg_user.plot.interactive
    save_dir_base = cfg_user.path.version_dir / 'images' / cfg.name / 'debug_full_vs_partial'

    # 1. Load Models
    logging.info('Loading models...')
    ae = load_extract_autoencoder_module()
    stage1 = load_extract_cond_flow_module(cfg.flow_stage1, ae)
    stage2 = load_extract_cond_flow_module(cfg.flow_stage2, ae)
    stage3 = load_extract_cond_flow_module(cfg.flow_stage3, ae)
    device = ae.device

    # 2. Dataset
    train_dataset = get_dataset(Partitions.train)
    class_names = train_dataset.class_names

    for i in cfg_user.plot.sample_indices:
        sample_i, target_i = train_dataset[i]
        label_name = class_names[target_i.label]
        logging.info(f'Processing sample {i} ({label_name})...')

        # --- FULL ECONSTRUCTION ---
        logging.info('Running full reconstruction...')
        full_recon_dir = save_dir_base / f'sample_{i}' / 'full'
        full_recon_dir.mkdir(parents=True, exist_ok=True)

        cloud_tensor = (
            sample_i.cloud.unsqueeze(0).to(device)
            if torch.is_tensor(sample_i.cloud)
            else torch.from_numpy(sample_i.cloud).unsqueeze(0).to(device)
        )
        inputs = Inputs(cloud=cloud_tensor)
        ae_out = ae.encode(inputs)
        z1 = ae_out.mu1
        z2 = ae_out.p_mu2 + ae_out.d_mu2

        stages = [
            (stage3, cfg.flow_stage3.objective.n_timesteps, 128),
            (stage2, cfg.flow_stage2.objective.n_timesteps, 512),
            (stage1, cfg.flow_stage1.objective.n_timesteps, 2048),
        ]

        x_current = None
        all_full_steps = []
        for stage, n_timesteps, n_points in stages:
            if x_current is not None:
                ratio = n_points // x_current.shape[1]
                if ratio > 1:
                    x_current = x_current.repeat_interleave(ratio, dim=1)
                    x_current = stage._add_transition_noise(x_current)

            step_list = stage.sample(
                n_samples=1,
                n_timesteps=n_timesteps,
                n_points=n_points,
                device=device,
                x_0=x_current,
                z1=z1,
                z2=z2,
            )
            all_full_steps.extend(step_list)
            x_current = step_list[-1]

        for step, x_t in enumerate(all_full_steps):
            render_cloud(
                (x_t[0].cpu().numpy(),), title=f'full_step_{step:03d}', save_dir=full_recon_dir, interactive=interactive
            )

        # --- PARTIAL RECONSTRUCTION (from 512) ---
        logging.info('Running partial reconstruction from 512...')
        partial_recon_dir = save_dir_base / f'sample_{i}' / 'partial_512'
        partial_recon_dir.mkdir(parents=True, exist_ok=True)

        x_current = (
            sample_i.cloud_512.unsqueeze(0).to(device)
            if torch.is_tensor(sample_i.cloud_512)
            else torch.from_numpy(sample_i.cloud_512).unsqueeze(0).to(device)
        )
        ratio = 4
        x_current = x_current.repeat_interleave(ratio, dim=1)
        x_current = stage1._add_transition_noise(x_current)

        all_partial_steps = stage1.sample(
            n_samples=1,
            n_timesteps=cfg.flow_stage1.objective.n_timesteps,
            n_points=2048,
            device=device,
            x_0=x_current,
            z1=z1,
            z2=z2,
        )

        for step, x_t in enumerate(all_partial_steps):
            render_cloud(
                (x_t[0].cpu().numpy(),),
                title=f'partial_step_{step:03d}',
                save_dir=partial_recon_dir,
                interactive=interactive,
            )

        # --- PARTIAL RECONSTRUCTION (from 128) ---
        logging.info('Running partial reconstruction from 128...')
        partial_128_dir = save_dir_base / f'sample_{i}' / 'partial_128'
        partial_128_dir.mkdir(parents=True, exist_ok=True)

        # Start from Stage 2
        x_current = (
            sample_i.cloud_128.unsqueeze(0).to(device)
            if torch.is_tensor(sample_i.cloud_128)
            else torch.from_numpy(sample_i.cloud_128).unsqueeze(0).to(device)
        )
        ratio = 4
        x_current = x_current.repeat_interleave(ratio, dim=1)
        x_current = stage2._add_transition_noise(x_current)

        steps_s2 = stage2.sample(
            n_samples=1,
            n_timesteps=cfg.flow_stage2.objective.n_timesteps,
            n_points=512,
            device=device,
            x_0=x_current,
            z1=z1,
            z2=z2,
        )

        # Transition to Stage 1
        x_current = steps_s2[-1].repeat_interleave(4, dim=1)
        x_current = stage1._add_transition_noise(x_current)
        steps_s1 = stage1.sample(
            n_samples=1,
            n_timesteps=cfg.flow_stage1.objective.n_timesteps,
            n_points=2048,
            device=device,
            x_0=x_current,
            z1=z1,
            z2=z2,
        )

        all_steps_128 = steps_s2 + steps_s1
        for step, x_t in enumerate(all_steps_128):
            render_cloud(
                (x_t[0].cpu().numpy(),),
                title=f'partial_128_step_{step:03d}',
                save_dir=partial_128_dir,
                interactive=interactive,
            )

        logging.info(f'Finished sample {i}')

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Entry point."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        run_debug_reconstructions()


if __name__ == '__main__':
    main()
