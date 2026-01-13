"""Visualize the dataset after pre-processing."""

import torch

from src.config_options import ConfigAll, Experiment, hydra_main
from src.datasets import Partitions, PointCloudDataset, get_dataset
from src.visualisation import render_cloud


@torch.inference_mode()
def visualize_dataset(dataset: PointCloudDataset) -> None:
    """Visualize the first point cloud in the dataset."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    save_dir = cfg.user.path.version_dir / 'images' / cfg.name

    for i in range(len(dataset)):
        row = dataset[i]
        input_pc = row[0].cloud
        label = row[1].label + 1
        render_cloud([input_pc.numpy()], title=f'{label=}', interactive=cfg_user.plot.interactive, save_dir=save_dir)


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Visualize the dataset after pre-processing."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
        visualize_dataset(dataset)
    return


if __name__ == '__main__':
    main()
