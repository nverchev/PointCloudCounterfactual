"""Generate flow model reconstructions, and visualize them together with the original point clouds."""

import logging
from collections.abc import Sized

import torch

from drytorch import Model

from src.module.flow import FlowReconstruction
from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.train.models import (
    load_extract_autoencoder_module,
    load_extract_cond_flow_module,
    load_extract_classifier_module,
)
from src.utils.visualization import render_cloud


def format_probs(probs: torch.Tensor, class_names: list[str]) -> str:
    """Format probability distribution for logging."""
    probs_np = probs.cpu().numpy()
    return f'({", ".join(f"{name}: {100 * p:.1f}%" for name, p in zip(class_names, probs_np, strict=True))})'


@torch.inference_mode()
def create_and_render_reconstructions() -> None:
    """Create and visualize the reconstructions using the multi-stage flow model."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user

    interactive = cfg_user.plot.interactive
    save_dir_base = cfg_user.path.version_dir / 'images' / cfg.name / 'flow_reconstruction'

    # 1. Load Models
    logging.info('Loading models...')
    classifier = load_extract_classifier_module()
    ae = load_extract_autoencoder_module()
    stage1_module = load_extract_cond_flow_module(cfg.flow_stage1, ae)
    stage2_module = load_extract_cond_flow_module(cfg.flow_stage2, ae)
    stage3_module = load_extract_cond_flow_module(cfg.flow_stage3, ae)
    device = ae.device

    reconstruction_module = FlowReconstruction(
        autoencoder=ae,
        stage1=stage1_module,
        stage2=stage2_module,
        stage3=stage3_module,
    )
    reconstruction_model = Model(
        reconstruction_module,
        name='FlowReconstruction',
        device=device,
    )

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

    # Classifier (Original)
    logits_original = classifier(inputs)
    probs_original = torch.softmax(logits_original, dim=1)

    # Flow Reconstruction
    out_recon = reconstruction_model(inputs)
    recon_clouds = out_recon.recon
    logits_recon = classifier(Inputs(cloud=out_recon.recon))
    probs_recon = torch.softmax(logits_recon, dim=1)

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
        p_orig = probs_original[idx]
        str_original = f'Original: {format_probs(p_orig, class_names)}'
        logging.info(str_original)
        render_cloud(
            (batch_cloud[idx].cpu().numpy(),),
            title=str_original,
            interactive=interactive,
            save_dir=save_dir,
        )

        # Reconstruction
        p_recon = probs_recon[idx]
        str_recon = f'Flow Reconstruction: {format_probs(p_recon, class_names)}'
        logging.info(str_recon)
        render_cloud(
            (recon_clouds[idx].cpu().numpy(),),
            title=str_recon,
            interactive=interactive,
            save_dir=save_dir,
        )

        # Combined
        render_cloud(
            (batch_cloud[idx].cpu().numpy(), recon_clouds[idx].cpu().numpy()),
            title='Original vs Reconstruction',
            interactive=interactive,
            save_dir=save_dir,
        )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the flow reconstruction visualization."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_and_render_reconstructions()

    return


if __name__ == '__main__':
    main()
