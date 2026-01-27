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
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        codebook_usage = sum((out.one_hot_idx.sum(0) for out in self.model_runner.outputs_list), torch.tensor(0))
        unused_entries = torch.eq(codebook_usage, 0)

        logging.info('Codebook usage: %.2f %%', unused_entries.sum() / unused_entries.numel() * 100)
        for code in range(self.n_codes):
            idx_frequency = np.array(codebook_usage[code])
            idx_frequency = idx_frequency / idx_frequency.sum()

            # Reassign unused entries
            for code_idx in range(self.book_size):
                if unused_entries[code, code_idx]:
                    # Sample from used entries and add noise to create new embedding
                    sampled_idx = np.random.choice(np.arange(self.book_size), p=idx_frequency)
                    template_embedding = self.module.codebook.data[code, sampled_idx]

                    # Add noise to template embedding
                    noise = self.vq_noise * torch.randn_like(template_embedding)
                    self.module.codebook.data[code, code_idx] = template_embedding + noise

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
        outputs = model(inputs)

        # Log reconstructions as 3D objects
        for i, recon in enumerate(outputs.recon.detach().cpu().numpy()):
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
        outputs = model(inputs)

        for i, recon in enumerate(outputs.recon.detach().cpu()):
            self.writer.add_mesh(
                tag=f'Recon {i}',
                vertices=recon.unsqueeze(0),
                global_step=model.epoch,
            )
