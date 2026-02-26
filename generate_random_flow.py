"""Generate random samples from the flow matching model."""

import torch

from src.train.models import EMAModel
from src.module import get_flow_module, FlowMatchingModel
from src.config import AllConfig, Experiment, hydra_main
from src.utils.visualization import render_cloud


@torch.inference_mode()
def generate_random_flow() -> None:
    """Generate random samples from the flow matching model."""
    cfg = Experiment.get_config()
    cfg_flow = cfg.flow
    cfg_user = cfg.user
    cfg_generate = cfg_user.generate

    # Base save directory
    save_base_dir = cfg.user.path.version_dir / 'images' / cfg.name / 'generated_flow'
    module = get_flow_module().eval()
    model = EMAModel(module, name=cfg_flow.model.name, device=cfg_user.device)

    # Load the latest checkpoint
    model.load_state(-1)
    ema_module = model.averaged_module
    assert isinstance(ema_module, FlowMatchingModel)

    n_samples = cfg_generate.batch_size
    n_points = cfg_flow.objective.n_inference_output_points

    # Generate samples and intermediate steps
    x_list = ema_module.sample(n_samples=n_samples, n_points=n_points, device=model.device)

    # Process each sample
    for i in range(n_samples):
        # Create a folder for each sample to store intermediate versions
        sample_dir = save_base_dir / f'sample_{i}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save each intermediate step
        for step, x_t in enumerate(x_list):
            cloud = x_t[i].cpu().numpy()
            step_name = f'step_{step:03d}'
            render_cloud(
                (cloud,),
                title=step_name,
                interactive=False,
                save_dir=sample_dir,
            )

        print(f'Generated sample {i} with {len(x_list)} steps in {sample_dir}')

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
