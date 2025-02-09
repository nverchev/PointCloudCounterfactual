from typing import cast

import hydra
from omegaconf import OmegaConf
import torch

from src.datasets import get_dataset, Partitions
from src.autoencoder import get_module
from src.config_options import ConfigTrainAE, ExperimentAE, ParentExperiment
from src.visualisation import infer_and_visualize, render_cloud
from dry_torch import Model


def visualize_reconstructions():
    cfg = ExperimentAE.get_config()
    test_dataset = get_dataset(Partitions.test if cfg.exp.final else Partitions.val)

    module = get_module()
    model = Model(module, name=cfg.autoencoder.name, device=cfg.user.device)
    model.load_state(-1)
    input_pcs: list[torch.Tensor] = []
    for i in cfg.user.plot.indices_to_reconstruct:
        assert i < len(test_dataset), 'Index is too large for the selected dataset'
        dataset_row = test_dataset
        input_pc = dataset_row[i][0].cloud
        render_cloud([input_pc.numpy()], title=f'sample_{i}', interactive=cfg.user.plot.interactive)
        input_pcs.append(input_pc)
    input_batch = torch.stack(input_pcs).to(cfg.user.device)
    infer_and_visualize(module, input_pc=input_batch, mode='recon')


def main() -> None:
    with hydra.initialize(version_base=None, config_path="hydra_conf/autoencoder_conf/"):
        dict_cfg = hydra.compose(config_name="defaults")
        cfg = cast(ConfigTrainAE, OmegaConf.to_object(dict_cfg))
        exp = ExperimentAE(cfg.exp.name, config=cfg)
        ParentExperiment(cfg.exp.main_name, par_dir=cfg.user.path.exp_par_dir).register_child(exp)
        with exp:
            visualize_reconstructions()
    return


if __name__ == "__main__":
    main()
