"""Generate flow counterfactual point clouds, and visualize them."""

import logging
from collections.abc import Sized

import numpy as np
import torch

from src.module import CounterfactualVAE
from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.utils.visualization import render_cloud
from src.train.models import (
    load_extract_autoencoder_module,
    load_extract_cond_flow_module,
    load_extract_classifier_module,
)


def format_probs(probs: np.ndarray, class_names: list[str]) -> str:
    """Format probability distribution for logging."""
    return f'({", ".join(f"{name}: {100 * p:.1f}%" for name, p in zip(class_names, probs, strict=True))})'


@torch.inference_mode()
def create_render_cf_flow() -> None:
    """Create and visualize counterfactuals using the multi-stage flow model."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    interactive = cfg_user.plot.interactive
    save_dir_base = cfg_user.path.version_dir / 'images' / cfg.name / 'counterfactual_flow'
    n_final = cfg.data.n_target_points

    # 1. Load Models
    logging.info('Loading models...')
    classifier = load_extract_classifier_module()
    ae = load_extract_autoencoder_module()
    if not isinstance(ae, CounterfactualVAE):
        raise TypeError('Averaged module is not a CounterfactualVAE')

    ema_module3 = load_extract_cond_flow_module(cfg.flow_stage3, ae)
    ema_module2 = load_extract_cond_flow_module(cfg.flow_stage2, ae)
    ema_module1 = load_extract_cond_flow_module(cfg.flow_stage1, ae)
    device = ema_module1.device

    stages = [
        (ema_module3, cfg.flow_stage3.objective.n_timesteps, 128),
        (ema_module2, cfg.flow_stage2.objective.n_timesteps, 512),
        (ema_module1, cfg.flow_stage1.objective.n_timesteps, n_final),
    ]

    # 2. Data
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
        sample_i, target_i = test_dataset[i]
        clouds.append(sample_i.cloud)
        labels.append(target_i.label)

    batch_cloud = torch.stack(clouds).to(device)
    inputs = Inputs(cloud=batch_cloud)

    # 3. Get Classifier Logits & Probabilities
    logits_original = classifier(inputs)
    probs_original = torch.softmax(logits_original, dim=1).cpu().numpy()
    inputs = inputs._replace(logits=logits_original)

    # 4. Generate Counterfactuals for each target class
    cf_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for j in range(len(class_names)):
        logging.info(f'Generating counterfactuals for class: {class_names[j]}...')

        # Get counterfactual features from VAE
        out_cf_vae = ae.get_counterfactual_output(
            inputs,
            target_dim=j,
            target_value=cfg.autoencoder.objective.counterfactual_value,
        )
        z1, z2 = out_cf_vae.z1, out_cf_vae.z2

        # Multi-stage flow sampling
        x_current = None
        for stage, n_timesteps, n_points in stages:
            x_current = stage.sample(
                n_samples=len(clouds),
                n_timesteps=n_timesteps,
                n_points=n_points,
                device=device,
                x_prev=x_current,
                z1=z1,
                z2=z2,
            )[-1]

        assert x_current is not None
        cf_clouds = x_current
        logits_cf = classifier(Inputs(cloud=cf_clouds))
        probs_cf = torch.softmax(logits_cf, dim=1).cpu().numpy()
        cf_results[j] = (cf_clouds.cpu().numpy(), probs_cf)

    # 5. Visualization
    for idx, sample_idx in enumerate(cfg_user.plot.sample_indices):
        label_name = class_names[labels[idx]]
        logging.info(f'Sample {sample_idx} with label "{label_name}":')

        save_dir = save_dir_base / f'sample_{sample_idx}'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Original
        p_orig = probs_original[idx]
        str_original = f'Original: {format_probs(p_orig, class_names)}'
        logging.info(str_original)
        render_cloud((batch_cloud[idx].cpu().numpy(),), title=str_original, interactive=interactive, save_dir=save_dir)

        # Counterfactuals
        cf_clouds_list = []
        for j, class_name in enumerate(class_names):
            clouds_j, probs_j = cf_results[j]
            cloud_np = clouds_j[idx]
            p_cf = probs_j[idx]
            str_cf = f'CF to {class_name}: {format_probs(p_cf, class_names)}'
            logging.info(str_cf)
            render_cloud((cloud_np,), title=str_cf, interactive=interactive, save_dir=save_dir)
            cf_clouds_list.append(cloud_np)

        render_cloud(cf_clouds_list, title='Flow Counterfactuals', interactive=interactive, save_dir=save_dir)

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Generate counterfactual point clouds using the flow model."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_render_cf_flow()
    return


if __name__ == '__main__':
    main()
