"""Script to explore and visualize the point cloud dataset hierarchy."""

import logging
from collections.abc import Sized

import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data import Partitions, get_dataset
from src.utils.visualization import render_cloud


@torch.inference_mode()
def explore_dataset() -> None:
    """Visualize the downsampled point cloud hierarchy."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user

    interactive = cfg_user.plot.interactive
    save_dir_base = cfg.user.path.version_dir / 'images' / 'dataset'

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    class_names = test_dataset.class_names

    for i in cfg_user.plot.sample_indices:
        assert isinstance(test_dataset, Sized)
        if i >= len(test_dataset):
            raise ValueError(f'Index {i} is too large for the selected dataset of length {len(test_dataset)}')

        inputs, targets = test_dataset[i]
        label_idx = int(targets.label.item())
        label_name = class_names[label_idx]

        logging.info('Exploring Dataset for Sample %d (label: %s):', i, label_name)
        save_dir = save_dir_base / f'sample_{i}'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Render clouds individually
        clouds_to_render = {
            'original': inputs.cloud,
            'downsampled_512': inputs.cloud_512,
            'downsampled_128': inputs.cloud_128,
        }

        for name, cloud in clouds_to_render.items():
            logging.info('  Rendering: %s', name)
            render_cloud(
                (cloud.numpy(),),
                title=f'{label_name}_{name}',
                interactive=interactive,
                save_dir=save_dir,
            )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and launch the dataset exploration."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        explore_dataset()

    return


if __name__ == '__main__':
    main()
