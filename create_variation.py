"""Create a variation of an existing experiment allowing for a different w_autoencoder architecture."""

import torch

from drytorch.lib.checkpoints import LocalCheckpoint
from drytorch import Model

from src.config_options import ConfigAll, Experiment, hydra_main
from src.autoencoder import CounterfactualVQVAE
from src.classifier import DGCNN

def create_variation() -> None:
    """Convenience method that copies the autoencoder and classifier state from the main experiment to a new one."""
    cfg = Experiment.get_config()

    # Copy the autoencoder state
    main_exp_dir = cfg.user.path.exp_par_dir / 'checkpoints' / 'VAEX-PCGen'
    checkpoint_dir = sorted(sorted(main_exp_dir.iterdir())[-1].iterdir())[-1]  # get the latest checkpoint

    vqvae_module = CounterfactualVQVAE()
    vqvae = Model(vqvae_module, name=cfg.autoencoder.model.name)
    vqvae.epoch = cfg.autoencoder.train.epochs
    main_vqvae_checkpoint = LocalCheckpoint(checkpoint_dir)
    main_vqvae_checkpoint.register_model(vqvae)
    main_state_path = main_vqvae_checkpoint.paths.model_state_path
    state_dict = torch.load(main_state_path, weights_only=True)
    state_dict = {key: value for key, value in state_dict.items() if 'w_autoencoder' not in key}
    vqvae_module.load_state_dict(state_dict, strict=False)
    vqvae.save_state()

    # Copy the classifier state
    dgcnn_module = DGCNN()
    dgcnn = Model(dgcnn_module, name=cfg.classifier.model.name)
    main_dgcnn_checkpoint = LocalCheckpoint(checkpoint_dir)
    main_dgcnn_checkpoint.register_model(dgcnn)
    epoch = main_dgcnn_checkpoint._get_last_saved_epoch()
    dgcnn.epoch = epoch
    main_state_path = main_dgcnn_checkpoint.paths.model_state_path
    dgcnn_module.load_state_dict(torch.load(main_state_path, weights_only=True))
    dgcnn.save_state()

@hydra_main
def main(cfg: ConfigAll) -> None:
    """Set up the experiment and launches the variation creation."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.exp_par_dir, tags=cfg.tags)
    with exp.create_run():
        create_variation()
    return


if __name__ == "__main__":
    main()
