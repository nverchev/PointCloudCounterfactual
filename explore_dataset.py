"""Visualize the dataset after pre-processing."""

from src.datasets import get_dataset, Partitions, PointCloudDataset
from src.config_options import Experiment, hydra_main, ConfigAll
from src.visualisation import render_cloud


def visualize_dataset(dataset: PointCloudDataset) -> None:
    """Visualize the first point cloud in the dataset."""
    cfg_user = Experiment.get_config().user

    for i in range(len(dataset)):
        row = dataset[i]
        input_pc = row[0].cloud
        label = row[1].label + 1
        render_cloud([input_pc.numpy()], title=f'{label=}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    """Visualize the dataset after pre-processing."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
        visualize_dataset(dataset)
    return


if __name__ == "__main__":
    main()
