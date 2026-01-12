"""Visualize the hyperparameters' optimization's results for the w-autoencoder."""

import pathlib

import hydra
import optuna
import yaml
from omegaconf import DictConfig
from optuna import visualization

from src.config_options import ConfigPath, VERSION


@hydra.main(version_base=None, config_path=ConfigPath.TUNE_W_AUTOENCODER.absolute(), config_name='defaults')
def tune(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)
    version = f"v{VERSION}"
    with (ConfigPath.CONFIG_ALL.get_path() / 'defaults').with_suffix('.yaml').open() as f:
        loaded_cfg = yaml.safe_load(f)
        variation = loaded_cfg['variation']

    study_name = '_'.join([tune_cfg.tune.study_name, version, variation] + tune_cfg.overrides[1:])
    study = optuna.load_study(study_name=study_name, storage=tune_cfg.storage)
    visualization.plot_optimization_history(study).show(renderer=tune_cfg.renderer)
    visualization.plot_slice(study).show(renderer=tune_cfg.renderer)
    visualization.plot_contour(study).show(renderer=tune_cfg.renderer)
    visualization.plot_parallel_coordinate(study).show(renderer=tune_cfg.renderer)
    visualization.plot_param_importances(study).show(renderer=tune_cfg.renderer)
    return


if __name__ == '__main__':
    tune()
