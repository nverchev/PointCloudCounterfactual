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


torch.inference_mode()


def visualize_counterfactuals() -> None:
    """Visualize the selected point clouds in the dataset."""
    cfg = Experiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    save_dir = cfg.user.path.version_dir / 'images' / cfg.name

    value = cfg_user.counterfactual_value
    dgcnn_module = DGCNN().eval()
    classifier = Model(dgcnn_module, name=cfg.classifier.architecture.name, device=cfg.user.device)
    classifier.load_state()

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    num_classes = cfg.data.dataset.n_classes

    vqvae_module = CounterfactualVQVAE().eval()
    model = Model(vqvae_module, name=cfg_ae.architecture.name, device=cfg_user.device)
    model.load_state()

    for i in cfg_user.plot.indices_to_reconstruct:
        assert isinstance(test_dataset, Sized)
        if i < len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')

        input_pc = test_dataset[i][0].cloud
        indices = test_dataset[i][0].indices
        label = test_dataset[i][1].label

        render_cloud((input_pc.numpy(),), title=f'sample_{i}', interactive=cfg_user.plot.interactive, save_dir=save_dir)
        input_pc = input_pc.to(model.device)
        indices = indices.to(model.device)
        with torch.inference_mode():
            logits = classifier(Inputs(cloud=input_pc.unsqueeze(0), indices=indices))
        np_probs = torch.softmax(logits, dim=1).cpu().numpy()
        relaxed_probs = torch.softmax(logits / cfg.autoencoder.architecture.encoder.w_encoder.cf_temperature, dim=1)
        print(f'Probs for sample {i} with label {label}: (', end='')
        for prob in np_probs[0]:
            print(f'{prob:.2f}', end=' ')
        print(')')

        with vqvae_module.double_encoding:
            inputs = Inputs(input_pc.unsqueeze(0), indices=indices)
            data = vqvae_module.encode(inputs)
            data.probs = relaxed_probs
            data = vqvae_module.decode(data, inputs)

        with torch.inference_mode():
            logits = classifier(Inputs(cloud=data.recon))
        np_recon = data.recon.detach().squeeze().cpu().numpy()
        render_cloud((np_recon,), title=f'reconstruction_{i}', interactive=cfg_user.plot.interactive, save_dir=save_dir)
        recon_probs = torch.softmax(logits, dim=1).cpu().numpy()
        print(f'Reconstruction {i}: (', end='')
        for prob in recon_probs[0]:
            print(f'{prob:.2f}', end=' ')

        print(')')

        np_recons = list[npt.NDArray[Any]]()
        for j in range(num_classes):
            with vqvae_module.double_encoding:
                target = torch.zeros_like(relaxed_probs)
                target[:, j] = 1
                data.probs = (1 - value) * relaxed_probs + value * target
                data = vqvae_module.decode(data, inputs)
                np_recon = data.recon.detach().squeeze().cpu().numpy()
                render_cloud(
                    (np_recon,),
                    title=f'counterfactual_{i}_to_{j}',
                    interactive=cfg_user.plot.interactive,
                    save_dir=save_dir,
                )
                np_recons.append(np_recon)
                with torch.inference_mode():
                    probs = torch.softmax(classifier(Inputs(cloud=data.recon)), dim=1).cpu().numpy()

                print(f'Counterfactual {i} to {j}: (', end='')
                for prob in probs[0]:
                    print(f'{prob:.2f}', end=' ')
                print(')')

        render_cloud(np_recons, title=f'counterfactuals_{i}', interactive=cfg_user.plot.interactive, save_dir=save_dir)


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up the experiment and launch the counterfactual visualization."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        visualize_counterfactuals()
    return


if __name__ == '__main__':
    main()
