"""Evaluate reconstruction performance of the multi-stage flow matching model."""

import torch

from drytorch import DataLoader, Test, Model
from src.config import AllConfig, Experiment, hydra_main
from src.data import get_dataset, Partitions
from src.module.flow import FlowReconstruction
from src.train.metrics_and_losses import get_chamfer_loss
from src.train.models import load_extract_autoencoder_module, load_extract_cond_flow_module


@torch.inference_mode()
def evaluate_flow_reconstruction() -> None:
    """Evaluate reconstruction performance of the multi-stage flow matching model."""
    cfg = Experiment.get_config()
    cfg_user = cfg.user

    ae = load_extract_autoencoder_module()
    stage1_module = load_extract_cond_flow_module(cfg.flow_stage1, ae)
    stage2_module = load_extract_cond_flow_module(cfg.flow_stage2, ae)
    stage3_module = load_extract_cond_flow_module(cfg.flow_stage3, ae)
    reconstruction_module = FlowReconstruction(
        autoencoder=ae,
        stage1=stage1_module,
        stage2=stage2_module,
        stage3=stage3_module,
    )
    reconstruction_model = Model(reconstruction_module, name='FlowReconstruction', device=cfg_user.device)
    test_dataset = get_dataset(Partitions.test if cfg.final else Partitions.val)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.user.generate.batch_size)
    metric = get_chamfer_loss()
    test = Test(reconstruction_model, loader=test_loader, metric=metric)
    test()
    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Evaluate reconstruction performance of the multi-stage flow matching model."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        evaluate_flow_reconstruction()


if __name__ == '__main__':
    main()
