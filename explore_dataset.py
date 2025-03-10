from src.datasets import get_dataset, Partitions
from src.config_options import ExperimentAE, MainExperiment, hydra_main, ConfigAll
from src.visualisation import render_cloud


def visualize_reconstructions() -> None:
    cfg = MainExperiment.get_config()
    cfg_user = cfg.user

    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)

    for i in range(len(test_dataset)):
        row = test_dataset[i]
        input_pc = row[0].cloud
        label = row[1].label + 1
        print(f'Label is {label}')
        render_cloud([input_pc.numpy()], title=f'{label=}', interactive=cfg_user.plot.interactive)


@hydra_main
def main(cfg: ConfigAll) -> None:
    parent_experiment = MainExperiment(cfg.name, cfg.user.path.exp_par_dir, cfg)
    exp_ae = ExperimentAE(cfg.autoencoder.name, config=cfg.autoencoder)
    parent_experiment.register_child(exp_ae)
    with exp_ae:
        visualize_reconstructions()
    return


if __name__ == "__main__":
    main()
