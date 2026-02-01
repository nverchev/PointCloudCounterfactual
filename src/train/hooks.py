"""Hooks to execute during training."""

import logging
from typing import cast

import numpy as np
import torch
import torch.distributed as dist

from torch.utils import data

from drytorch.core import protocols as p
from drytorch.lib.load import take_from_dataset
from drytorch.lib.runners import ModelRunner
from src.module.autoencoders import BaseVQVAE, WAutoEncoder
from src.config.experiment import Experiment
from src.data.structures import Inputs, Outputs, Targets


class DiscreteSpaceOptimizer:
    """Optimize the discrete latent space usage in Vector Quantized models.

    This optimizer monitors and adjusts the usage of codebook entries during training
    to ensure efficient utilization of the discrete latent space. Unused codebook
    entries are reassigned based on the usage patterns of active entries.
    """

    module: BaseVQVAE[WAutoEncoder]

    def __init__(self, model_runner: ModelRunner[Inputs, Targets, Outputs]) -> None:
        """Initialize the optimizer.

        Args:
            model_runner: Runner containing the model to optimize.
        """
        cfg_ae = Experiment.get_config().autoencoder
        cfg_ae_model = cfg_ae.model
        self.n_codes = cfg_ae_model.n_codes
        self.book_size = cfg_ae_model.book_size
        self.vq_noise = cfg_ae_model.vq_noise
        self.model_runner = model_runner
        if isinstance(model_runner.model.module, torch.nn.parallel.DistributedDataParallel):
            self.module = cast(BaseVQVAE[WAutoEncoder], model_runner.model.module.module)
        else:
            self.module = cast(BaseVQVAE[WAutoEncoder], model_runner.model.module)

        if not isinstance(self.module, BaseVQVAE):
            raise ValueError('Model not supported for VQ optimization.')

        return

    def __call__(self) -> None:
        """Optimize codebook usage by reassigning unused entries."""
        self.model_runner(store_outputs=True)
        with torch.no_grad():
            if not dist.is_initialized() or dist.get_rank() == 0:
                zeros = torch.zeros(self.n_codes, self.book_size)
                codebook_usage = sum((out.one_hot_idx.sum(0) for out in self.model_runner.outputs_list), zeros)
                unused_mask = codebook_usage == 0
                usage_pct = (~unused_mask).float().mean() * 100
                logging.info('Codebook usage: %.2f%%', usage_pct)
                for code in range(self.n_codes):
                    usage_freq = codebook_usage[code].float()
                    prob = (usage_freq / usage_freq.sum()).cpu().numpy()
                    dead_indices = torch.where(unused_mask[code])[0]
                    for code_idx in dead_indices:
                        sampled_idx = np.random.choice(self.book_size, p=prob)
                        template = self.module.codebook[code, sampled_idx]
                        noise = torch.randn_like(template) * self.module.codebook[code].std() * self.vq_noise
                        self.module.codebook[code, code_idx] = template + noise

            if dist.is_initialized():
                dist.broadcast(self.module.codebook, src=0)

        return


class WandbLogReconstruction:
    """Log sample reconstructions to Weights & Biases during training."""

    def __init__(self, dataset: data.Dataset[tuple[Inputs, Targets]], num_samples: int = 1):
        """Initialize the logging hook.

        Args:
            dataset: Dataset containing input samples to visualize.
            num_samples: Number of samples to log in each iteration.
        """
        import wandb

        from drytorch.trackers.wandb import Wandb

        self._dataset = dataset
        self._num_samples = num_samples
        self.run = Wandb.get_current().run
        inputs, targets = take_from_dataset(dataset, num_samples)
        for i, (input_, label) in enumerate(zip(inputs.cloud.numpy(), targets.label, strict=True)):
            self.run.log({f'Sample {i} with label: {label.item()}': wandb.Object3D(input_)})

        return

    def __call__(self, trainer: p.TrainerProtocol[Inputs, Targets, Outputs]) -> None:
        """Log reconstructed samples to Weights & Biases.

        Args:
            trainer: Training protocol containing the model to evaluate.
        """
        import wandb

        model = trainer.model

        # Get samples and generate reconstructions
        inputs, _targets = take_from_dataset(self._dataset, self._num_samples, device=model.device)
        out = model(inputs)

        # Log reconstructions as 3D objects
        for i, recon in enumerate(out.recon.detach().cpu().numpy()):
            self.run.log({f'Recon {i}': wandb.Object3D(recon)})


class TensorBoardLogReconstruction:
    """Log sample reconstructions to Tensorboard during training."""

    def __init__(self, dataset: data.Dataset[tuple[Inputs, Targets]], num_samples: int = 1):
        """Initialize the logging hook.

        Args:
            dataset: Dataset containing input samples to visualize.
            num_samples: Number of samples to log in each iteration.
        """
        from drytorch.trackers.tensorboard import TensorBoard

        self._dataset = dataset
        self._num_samples = num_samples
        self.writer = TensorBoard.get_current().writer
        inputs, targets = take_from_dataset(dataset, num_samples)
        for i, (input_, label) in enumerate(zip(inputs.cloud, targets.label, strict=True)):
            self.writer.add_mesh(
                tag=f'Sample {i} with label: {label.item()}',
                vertices=input_.unsqueeze(0),
                global_step=0,
            )
        return

    def __call__(self, trainer: p.TrainerProtocol[Inputs, Targets, Outputs]) -> None:
        """Log reconstructed samples to Tensorboard.

        Args:
            trainer: Training protocol containing the model to evaluate.
        """
        model = trainer.model

        inputs, _targets = take_from_dataset(self._dataset, self._num_samples, device=model.device)
        out = model(inputs)

        for i, recon in enumerate(out.recon.detach().cpu()):
            self.writer.add_mesh(
                tag=f'Recon {i}',
                vertices=recon.unsqueeze(0),
                global_step=model.epoch,
            )
