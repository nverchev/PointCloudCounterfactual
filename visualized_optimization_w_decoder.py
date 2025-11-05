"""Visualize the hyperparameters' optimization's results for the w-autoencoder."""

import pathlib
import hydra
import optuna
import yaml
from omegaconf import DictConfig
from optuna import visualization
from src.config_options import ConfigPath


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_W_AUTOENCODER.absolute(), config_name='defaults')
def tune(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    with (ConfigPath.CONFIG_ALL.get_path() / 'defaults').with_suffix('.yaml').open() as f:
        version = f'v{yaml.safe_load(f)['version']}'

    study = optuna.load_study(study_name=tune_cfg.tune.study_name + '_'.join(['', version] + tune_cfg.overrides[1:]),
                              storage=tune_cfg.storage)
    visualization.plot_optimization_history(study).show(renderer=tune_cfg.renderer)
    visualization.plot_slice(study).show(renderer=tune_cfg.renderer)
    visualization.plot_contour(study).show(renderer=tune_cfg.renderer)
    visualization.plot_parallel_coordinate(study).show(renderer=tune_cfg.renderer)
    visualization.plot_param_importances(study).show(renderer=tune_cfg.renderer)
    return


if __name__ == '__main__':
    tune()
