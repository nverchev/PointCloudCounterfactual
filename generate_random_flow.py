"""Generate random samples from the multi-stage flow matching model."""

import logging

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.config.options import FlowModels
from src.utils.visualization import render_cloud
from src.train.models import load_extract_autoencoder_module, load_extract_cond_flow_module, load_extract_flow_module


@torch.inference_mode()
def generate_random_flow() -> None:
    """Generate random samples from the multi-stage flow matching model."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    cfg_generate = cfg_user.generate
    n_samples = cfg_generate.batch_size
    n_final = cfg.data.n_target_points
    save_base_dir = cfg.user.path.version_dir / 'images' / cfg.name / 'generated_flow_stages'

    if cfg.flow_stage1.model.class_name == FlowModels.CondFlowMatching:
        ae = load_extract_autoencoder_module()
        ema_module3 = load_extract_cond_flow_module(cfg.flow_stage3, ae)
        ema_module2 = load_extract_cond_flow_module(cfg.flow_stage2, ae)
        ema_module1 = load_extract_cond_flow_module(cfg.flow_stage1, ae)
        ae_out = ae.generate(batch_size=n_samples)
        z1, z2 = ae_out.z1, ae_out.z2
    else:
        ema_module3 = load_extract_flow_module(cfg.flow_stage3)
        ema_module2 = load_extract_flow_module(cfg.flow_stage2)
        ema_module1 = load_extract_flow_module(cfg.flow_stage1)
        z1, z2 = None, None

    device = ema_module1.device
    stages = [
        (ema_module3, cfg.flow_stage3.objective.n_timesteps, 128),
        (ema_module2, cfg.flow_stage2.objective.n_timesteps, 512),
        (ema_module1, cfg.flow_stage1.objective.n_timesteps, n_final),
    ]

    all_steps = []
    x_current = None
    for i, (stage, n_timesteps, n_points) in enumerate(stages):
        logging.info(f'Running Stage {3 - i} ({"Noise" if x_current is None else x_current.shape[1]} -> {n_points})...')

        if x_current is not None:
            ratio = n_points // x_current.shape[1]
            if ratio > 1:
                x_current = x_current.repeat_interleave(ratio, dim=1)

        step_list = stage.sample(
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            n_points=n_points,
            device=device,
            x_0=x_current,
            z1=z1,
            z2=z2,
        )
        all_steps.extend(step_list)
        x_current = step_list[-1]

    for i in range(n_samples):
        sample_dir = save_base_dir / f'sample_{i}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        for step, x_t in enumerate(all_steps):
            render_cloud((x_t[i].cpu().numpy(),), title=f'step_{step:03d}', interactive=False, save_dir=sample_dir)

        logging.info(f'Generated multi-stage sample {i} with {len(all_steps)} steps in {sample_dir}')

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Generate random samples from the flow matching model."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        generate_random_flow()

    return


if __name__ == '__main__':
    main()
