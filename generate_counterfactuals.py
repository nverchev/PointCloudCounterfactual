"""Generate reconstructions and counterfactuals."""

from collections.abc import Sized

import numpy as np
import torch

from drytorch import Model

from src.module import CounterfactualVAE, get_classifier
from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.utils.visualization import render_cloud


@torch.inference_mode()
def create_and_render_counterfactuals() -> None:
    """Create and visualize the selected point clouds in the dataset."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    counterfactual_value = cfg_user.counterfactual_value
    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / 'images' / cfg.name

    dgcnn_module = get_classifier().eval()
    classifier = Model(dgcnn_module, name=cfg.classifier.model.name, device=cfg.user.device)
    classifier.load_state()
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    num_classes = cfg.data.dataset.n_classes
    vae_module = CounterfactualVAE().eval()
    model = Model(vae_module, name=cfg_ae.model.name, device=cfg_user.device)
    model.load_state()

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
    out_recon = vae_module(inputs)
    recon_clouds = out_recon.recon.detach().cpu().numpy()
    logits_recon = classifier(Inputs(cloud=out_recon.recon))
    probs_recon = torch.softmax(logits_recon, dim=1).cpu().numpy()

    # Counterfactuals
    cf_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # Map target_class -> (clouds, probs)
    for j in range(num_classes):
        out_cf = vae_module.generate_counterfactual(
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
        print(f'Sample {sample_idx} with label {labels[idx]}:')
        save_dir = save_dir_base / f'sample_{sample_idx}'
        save_dir.mkdir(parents=True, exist_ok=True)
        for old_file in save_dir.iterdir():
            old_file.unlink()

        # Original
        p_orig = probs_original[idx]
        str_original = f'Original: ({", ".join(f"{i}: {100 * p:.2f}%" for i, p in enumerate(p_orig))})'
        print(str_original)
        render_cloud((batch_cloud[idx].cpu().numpy(),), title=str_original, interactive=interactive, save_dir=save_dir)

        # Reconstruction
        p_recon = probs_recon[idx]
        str_recon = f'Reconstruction: ({", ".join(f"{i}: {100 * p:.2f}%" for i, p in enumerate(p_recon))})'
        print(str_recon)
        render_cloud((recon_clouds[idx],), title=str_recon, interactive=interactive, save_dir=save_dir)

        # Counterfactuals
        cf_clouds_list = []
        for j in range(num_classes):
            clouds_j, probs_j = cf_results[j]
            cloud_np = clouds_j[idx]
            p_cf = probs_j[idx]
            str_cf = f'Counterfactual to {j}: ({", ".join(f"{i}: {100 * p:.2f}%" for i, p in enumerate(p_cf))})'
            print(str_cf)
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
