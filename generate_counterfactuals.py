"""Generate counterfactual point clouds, and visualize them together with the original point clouds."""

import logging
from collections.abc import Sized

import numpy as np
import torch

from drytorch import Model

from src.module import CounterfactualVAE, get_classifier, get_autoencoder
from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.train.models import EMAModel
from src.utils.visualization import render_cloud


def format_probs(probs: np.ndarray, class_names: list[str]) -> str:
    """Format probability distribution for logging."""
    return f'({", ".join(f"{name}: {100 * p:.1f}%" for name, p in zip(class_names, probs, strict=True))})'


@torch.inference_mode()
def create_and_render_counterfactuals() -> None:
    """Create and visualize the selected point clouds in the dataset."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    counterfactual_value = cfg_ae.objective.counterfactual_value

    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / 'images' / cfg.name

    dgcnn_module = get_classifier().eval()
    classifier = Model(dgcnn_module, name=cfg.classifier.model.name, device=cfg_user.device)
    classifier.load_state()
    if cfg_user.plot.use_train:
        partition = Partitions.train_val if cfg.final else Partitions.train
    else:
        partition = Partitions.test if cfg.final else Partitions.val

    test_dataset = get_dataset(partition)
    class_names = test_dataset.class_names
    module = get_autoencoder().eval()
    model = EMAModel(module, name=cfg_ae.model.name, device=cfg_user.device)
    model.load_state()
    ema_module = model.averaged_module
    if not isinstance(ema_module, CounterfactualVAE):
        raise TypeError('Averaged module is not a CounterfactualVAE')

    # 1. Collect Batch Data
    clouds = []
    labels = []

    for i in cfg_user.plot.sample_indices:
        assert isinstance(test_dataset, Sized)
        if i >= len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')

        sample_i, target_i = test_dataset[i]
        clouds.append(sample_i.cloud)
        labels.append(target_i.label)

    # 2. Batch Inference
    batch_cloud = torch.stack(clouds).to(model.device)
    inputs = Inputs(cloud=batch_cloud)

    # Classifier (Original)
    logits_original = classifier(inputs)
    probs_original = torch.softmax(logits_original, dim=1).cpu().numpy()
    inputs = inputs._replace(logits=logits_original)

    # VAE Reconstruction
    out_recon = ema_module(inputs)
    recon_clouds = out_recon.recon.detach().cpu().numpy()
    logits_recon = classifier(Inputs(cloud=out_recon.recon))
    probs_recon = torch.softmax(logits_recon, dim=1).cpu().numpy()

    # Counterfactuals
    cf_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # Map target_class -> (clouds, probs)
    for j in range(len(class_names)):
        out_cf = ema_module.generate_counterfactual(
            inputs,
            target_dim=j,
            target_value=counterfactual_value,
        )
        cf_clouds = out_cf.recon.detach().cpu().numpy()
        logits_cf = classifier(Inputs(cloud=out_cf.recon))
        probs_cf = torch.softmax(logits_cf, dim=1).cpu().numpy()
        cf_results[j] = (cf_clouds, probs_cf)

    # 3. Visualization and Reporting
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
        render_cloud((batch_cloud[idx].cpu().numpy(),), title=str_original, interactive=interactive, save_dir=save_dir)

        # Reconstruction
        p_recon = probs_recon[idx]
        str_recon = f'Reconstruction: {format_probs(p_recon, class_names)}'
        logging.info(str_recon)
        render_cloud((recon_clouds[idx],), title=str_recon, interactive=interactive, save_dir=save_dir)

        # Counterfactuals
        cf_clouds_list = []
        for j, class_name in enumerate(class_names):
            clouds_j, probs_j = cf_results[j]
            cloud_np = clouds_j[idx]
            p_cf = probs_j[idx]
            str_cf = f'Counterfactual to {class_name} ({j}): {format_probs(p_cf, class_names)}'
            logging.info(str_cf)
            render_cloud((cloud_np,), title=str_cf, interactive=interactive, save_dir=save_dir)
            cf_clouds_list.append(cloud_np)

        # All counterfactuals combined
        render_cloud(cf_clouds_list, title='Counterfactuals', interactive=interactive, save_dir=save_dir)
        print()

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the counterfactual visualization."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_and_render_counterfactuals()

    return


if __name__ == '__main__':
    main()
