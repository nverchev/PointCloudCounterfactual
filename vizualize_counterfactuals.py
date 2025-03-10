import numpy.typing as npt
import torch

from src.classifier import DGCNN
from src.data_structures import Inputs
from src.datasets import get_dataset, Partitions
from src.autoencoder import VQVAE
from src.config_options import ExperimentAE, MainExperiment, ConfigAll, hydra_main, ExperimentClassifier
from src.visualisation import render_cloud
from dry_torch import Model


def visualize_counterfactuals(classifier) -> None:
    cfg = MainExperiment.get_config()
    cfg_ae = cfg.autoencoder
    cfg_user = cfg.user
    value = cfg_user.counterfactual_value

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    num_classes = cfg.data.dataset.n_classes

    module = VQVAE().eval()
    model = Model(module, name=cfg_ae.model.name, device=cfg_user.device)
    model.load_state()

    for i in cfg_user.plot.indices_to_reconstruct:
        assert i < len(test_dataset), 'Index is too large for the selected dataset'
        # inference mode prevents random augmentation
        with torch.inference_mode():
            input_pc = test_dataset[i][0].cloud
            indices = test_dataset[i][0].indices

        render_cloud((input_pc.numpy(),), title=f'sample_{i}', interactive=cfg_user.plot.interactive)
        input_pc = input_pc.to(model.device)
        indices = indices.to(model.device)
        with torch.inference_mode():
            logits = classifier(Inputs(cloud=input_pc.unsqueeze(0), indices=indices))
        np_probs = torch.softmax(logits, dim=1).cpu().numpy()
        relaxed_probs = torch.softmax(logits / cfg.autoencoder.model.encoder.w_encoder.cf_temperature, dim=1)
        print(f'Sample {i}: (', end='')
        for prob in np_probs[0]:
            print(f'{prob:.2f}', end=' ')
        print(')')

        with model.module.double_encoding:
            data = module.encode(input_pc.unsqueeze(0), indices=indices)
            data.probs = relaxed_probs
            data = module.decode(data)

        with torch.inference_mode():
            logits = classifier(Inputs(cloud=data.recon))
        np_recon = data.recon.detach().squeeze().cpu().numpy()
        render_cloud((np_recon,),
                     title=f'reconstruction_{i}',
                     interactive=cfg_user.plot.interactive)
        recon_probs = torch.softmax(logits, dim=1).cpu().numpy()
        print(f'Reconstruction {i}: (', end='')
        for prob in recon_probs[0]:
            print(f'{prob:.2f}', end=' ')
        print(')')

        np_recons = list[npt.NDArray]()
        for j in range(num_classes):
            with model.module.double_encoding:
                target = torch.zeros_like(relaxed_probs)
                target[:, j] = 1
                data.probs = (1 - value) * relaxed_probs + value * target
                data = module.decode(data)
                np_recon = data.recon.detach().squeeze().cpu().numpy()
                render_cloud((np_recon,),
                             title=f'counterfactual_{i}_to_{j}',
                             interactive=cfg_user.plot.interactive)
                np_recons.append(np_recon)
                with torch.inference_mode():
                    probs = torch.softmax(classifier(Inputs(cloud=data.recon)), dim=1).cpu().numpy()
                print(f'Counterfactual {i} to {j}: (', end='')
                for prob in probs[0]:
                    print(f'{prob:.2f}', end=' ')
                print(')')

        render_cloud(np_recons, title=f'counterfactuals_{i}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    parent_experiment = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    exp_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    exp_class = ExperimentClassifier(cfg.classifier.name, config=cfg.classifier)
    parent_experiment.register_child(exp_ae)
    parent_experiment.register_child(exp_class)
    with exp_class:
        module = DGCNN().eval()
        classifier = Model(module, name=cfg.classifier.model.name, device=cfg.user.device)
        classifier.load_state()
    with exp_ae:
        visualize_counterfactuals(classifier)
    return


if __name__ == "__main__":
    main()
