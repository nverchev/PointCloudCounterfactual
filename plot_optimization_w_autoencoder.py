"""Visualize the hyperparameters' optimization's results for the w-autoencoder."""

import hydra
import optuna

from omegaconf import DictConfig

from src.config import ConfigPath
from src.utils.tuning import visualize_study, get_study_name


@hydra.main(version_base=None, config_path=ConfigPath.TUNING_W_AUTOENCODER.absolute(), config_name='defaults')
def tune(tune_cfg: DictConfig):
    """Set up the study and launch the optimization."""
    study_name = get_study_name(tune_cfg.tune.study_name, tune_cfg.overrides)
    study = optuna.load_study(study_name=study_name, storage=tune_cfg.storage)
    visualize_study(study, tune_cfg.renderer)
    return


if __name__ == '__main__':
    tune()
