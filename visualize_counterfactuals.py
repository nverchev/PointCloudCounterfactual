"""Visualize counterfactuals."""

from collections.abc import Sized
from typing import Any

import numpy.typing as npt
import torch

from drytorch import Model

from src.module import CounterfactualVQVAE, get_classifier
from src.config import AllConfig, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.utils.visualization import render_cloud


def calculate_and_print_probs(
    classifier: Model[Inputs, torch.Tensor],
    cloud: torch.Tensor,
    label_prefix: str,
) -> tuple[torch.Tensor, str]:
    """Calculate probabilities and print them."""
    logits = classifier(Inputs(cloud=cloud))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    str_out = f'{label_prefix}: ({" ".join(f"{p:.2f}" for p in probs)})'
    print(str_out)
    return logits, str_out


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
    vqvae_module = CounterfactualVQVAE().eval()
    model = Model(vqvae_module, name=cfg_ae.model.name, device=cfg_user.device)
    model.load_state()

    for i in cfg_user.plot.sample_indices:
        assert isinstance(test_dataset, Sized)
        if i >= len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')

        save_dir = save_dir_base / f'sample_{i}'
        save_dir.mkdir(parents=True, exist_ok=True)
        for old_file in save_dir.iterdir():
            old_file.unlink()

        sample_i = test_dataset[i][0]
        cloud = sample_i.cloud.to(model.device).unsqueeze(0)
        indices = sample_i.indices.to(model.device).unsqueeze(0)
        sample_i = Inputs(cloud=cloud, indices=indices)
        label_i = test_dataset[i][1].label
        print(f'Sample {i} with label {label_i}:')

        input_pc = cloud.squeeze().cpu().numpy()
        logits, str_original = calculate_and_print_probs(classifier, sample_i.cloud, 'Original')

        data = vqvae_module(sample_i)
        recon = data.recon.detach().squeeze().cpu().numpy()
        _, str_recon = calculate_and_print_probs(classifier, data.recon, 'Reconstruction')

        data = vqvae_module.double_reconstruct_with_logits(sample_i, logits)
        double_recon = data.recon.detach().squeeze().cpu().numpy()
        _, str_double_recon = calculate_and_print_probs(classifier, data.recon, 'Double Reconstruction')

        counterfactuals = list[npt.NDArray[Any]]()
        str_counterfactual = list[str]()
        for j in range(num_classes):
            data = vqvae_module.generate_counterfactual(
                sample_i,
                sample_logits=logits,
                target_dim=j,
                target_value=counterfactual_value,
            )
            _, str_out = calculate_and_print_probs(classifier, data.recon, f'Counterfactual to {j}')
            str_counterfactual.append(str_out)
            counterfactual = data.recon.detach().squeeze().cpu().numpy()
            counterfactuals.append(counterfactual)

        print()
        render_cloud((input_pc,), title=str_original, interactive=interactive, save_dir=save_dir)
        render_cloud((recon,), title=str_recon, interactive=interactive, save_dir=save_dir)
        render_cloud((double_recon,), title=str_double_recon, interactive=interactive, save_dir=save_dir)
        for j in range(num_classes):
            render_cloud((counterfactuals[j],), title=str_counterfactual[j], interactive=interactive, save_dir=save_dir)

        render_cloud(counterfactuals, title='Counterfactuals', interactive=interactive, save_dir=save_dir)

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
