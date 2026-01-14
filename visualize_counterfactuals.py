"""Visualize counterfactuals."""

from collections.abc import Sized
from typing import Any

import numpy.typing as npt
import torch

from drytorch import Model

from src.module import CounterfactualVQVAE, DGCNN
from src.config import ConfigAll, Experiment, hydra_main
from src.data import Inputs, Partitions, get_dataset
from src.utils.visualisation import render_cloud


def calculate_and_print_probs(
    classifier: Model[Inputs, torch.Tensor],
    cloud: torch.Tensor,
    label_prefix: str,
) -> torch.Tensor:
    """Calculate probabilities and print them."""
    logits = classifier(Inputs(cloud=cloud))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    print(f'{label_prefix}: ({" ".join(f"{p:.2f}" for p in probs)})')
    return logits


@torch.inference_mode()
def create_and_render_counterfactuals() -> None:
    """Create and visualize the selected point clouds in the dataset."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / 'images' / cfg.name

    counterfactual_value = cfg_user.counterfactual_value
    dgcnn_module = DGCNN().eval()
    classifier = Model(dgcnn_module, name=cfg.classifier.architecture.name, device=cfg.user.device)
    classifier.load_state()

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    num_classes = cfg.data.dataset.n_classes

    vqvae_module = CounterfactualVQVAE().eval()
    model = Model(vqvae_module, name=cfg_ae.architecture.name, device=cfg_user.device)
    model.load_state()

    for i in cfg_user.plot.sample_indices:
        assert isinstance(test_dataset, Sized)
        if i >= len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')

        save_dir = save_dir_base / f'sample_{i}'
        sample_i = test_dataset[i][0]
        cloud = sample_i.cloud.to(model.device).unsqueeze(0)
        indices = sample_i.indices.to(model.device).unsqueeze(0)
        sample_i = Inputs(cloud=cloud, indices=indices)
        label_i = test_dataset[i][1].label
        input_pc = cloud.squeeze().cpu().numpy()
        logits = calculate_and_print_probs(classifier, sample_i.cloud, f'Probs for sample {i} with label {label_i}')
        data = vqvae_module(sample_i)
        recon = data.recon.detach().squeeze().cpu().numpy()
        _ = calculate_and_print_probs(classifier, data.recon, 'Reconstruction')
        relaxed_probs = torch.softmax(logits / cfg.autoencoder.architecture.encoder.w_encoder.cf_temperature, dim=1)
        with vqvae_module.double_encoding:
            data = vqvae_module.encode(sample_i)
            data.probs = relaxed_probs
            data = vqvae_module.decode(data, sample_i)

        double_recon = data.recon.detach().squeeze().cpu().numpy()
        _ = calculate_and_print_probs(classifier, data.recon, 'Double Reconstruction')
        counterfactuals = list[npt.NDArray[Any]]()
        for j in range(num_classes):
            with vqvae_module.double_encoding:
                target = torch.zeros_like(relaxed_probs)
                target[:, j] = 1
                data.probs = (1 - counterfactual_value) * relaxed_probs + counterfactual_value * target
                data = vqvae_module.decode(data, sample_i)
                _ = calculate_and_print_probs(classifier, data.recon, f'Counterfactual to {j}')
                counterfactual = data.recon.detach().squeeze().cpu().numpy()
                counterfactuals.append(counterfactual)

        print()
        render_cloud((input_pc,), title=f'Sample with Label {label_i}', interactive=interactive, save_dir=save_dir)
        render_cloud((recon,), title='Reconstruction', interactive=interactive, save_dir=save_dir)
        render_cloud((double_recon,), title='Double Reconstruction', interactive=interactive, save_dir=save_dir)
        for j in range(num_classes):
            counterfactual = (counterfactuals[j],)
            render_cloud(counterfactual, title=f'Counterfactual_to_{j}', interactive=interactive, save_dir=save_dir)

        render_cloud(counterfactuals, title='Counterfactuals', interactive=interactive, save_dir=save_dir)
        return


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up the experiment and launch the counterfactual visualization."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_and_render_counterfactuals()

    return


if __name__ == '__main__':
    main()
