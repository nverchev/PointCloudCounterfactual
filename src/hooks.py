"""Hooks for model training and optimization."""

import numpy as np
import torch
import wandb
from torch.utils import data
import torch.distributed as dist

from drytorch.core import protocols as p
from drytorch.lib.load import take_from_dataset
from drytorch.lib.runners import ModelRunner

from src.config_options import Experiment
from src.autoencoder import AbstractVQVAE
from src.data_structures import Inputs, Outputs, Targets


class DiscreteSpaceOptimizer:
    """Optimize the discrete latent space usage in Vector Quantized models.

    This optimizer monitors and adjusts the usage of codebook entries during training
    to ensure efficient utilization of the discrete latent space. Unused codebook
    entries are reassigned based on the usage patterns of active entries.
    """

    def __init__(self, model_runner: ModelRunner) -> None:
        """Initialize the optimizer.

        Args:
            model_runner: Runner containing the model to optimize.
        """
        self.model_runner = model_runner
        if isinstance(model_runner.model.module, torch.nn.parallel.DistributedDataParallel):
            self.module = model_runner.model.module.module
        else:
            self.module = model_runner.model.module

        if not isinstance(self.module, AbstractVQVAE):
            raise ValueError('Model not supported for VQ optimization.')

        cfg_ae = Experiment.get_config().autoencoder
        self.cfg_ae_arc = cfg_ae.architecture
        self.final_epoch = cfg_ae.train.n_epochs

    def __call__(self) -> None:
        """Optimize codebook usage by reassigning unused entries."""
        self.model_runner(store_outputs=True)
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # Calculate codebook usage statistics
        codebook_usage = torch.vstack([output.one_hot_idx for output in self.model_runner.outputs_list]).sum(0)
        unused_entries = torch.eq(codebook_usage, 0)

        # Process each codebook
        for book_idx in range(self.cfg_ae_arc.w_dim // self.cfg_ae_arc.embedding_dim):
            # Calculate the probability distribution of used codebook entries
            usage_probs = np.array(codebook_usage[book_idx])
            usage_probs = usage_probs / usage_probs.sum()

            # Reassign unused entries
            for entry_idx in range(self.cfg_ae_arc.book_size):
                if unused_entries[book_idx, entry_idx]:
                    # Sample from used entries and add noise to create new embedding
                    sampled_idx = np.random.choice(np.arange(self.cfg_ae_arc.book_size), p=usage_probs)
                    template_embedding = self.module.codebook.data[book_idx, sampled_idx]

                    if self.model_runner.model.epoch == self.final_epoch:
                        # Disable unused entries in final epoch
                        self.module.codebook.data[book_idx, entry_idx] = 1000
                    else:
                        # Add noise to template embedding
                        noise = self.cfg_ae_arc.vq_noise * torch.randn_like(template_embedding)
                        self.module.codebook.data[book_idx, entry_idx] = template_embedding + noise


class WandbLogReconstruction:
    """Log sample reconstructions to Weights & Biases during training.

    This hook visualizes the original input samples and their reconstructions
    in 3D using Weights & Biases logging functionality. It helps monitor the
    quality of the autoencoder's reconstruction capabilities.
    """

    def __init__(self, dataset: data.Dataset[tuple[Inputs, Targets]], num_samples: int = 1):
        """Initialize the logging hook.

        Args:
            dataset: Dataset containing input samples to visualize.
            num_samples: Number of samples to log in each iteration.
        """
        from drytorch.trackers.wandb import Wandb

        self._dataset = dataset
        self._num_samples = num_samples
        self.run = Wandb.get_current().run
        inputs, targets = take_from_dataset(dataset, num_samples)
        for i, (input_, label) in enumerate(zip(inputs.cloud.numpy(), targets.label)):
            self.run.log({f'Sample {i} with label: {label.item()}': wandb.Object3D(input_)})

        return

    def __call__(self, trainer: p.TrainerProtocol[Inputs, Targets, Outputs]) -> None:
        """Log reconstructed samples to Weights & Biases.

        Args:
            trainer: Training protocol containing the model to evaluate.
        """
        model = trainer.model

        # Get samples and generate reconstructions
        inputs, targets = take_from_dataset(self._dataset, self._num_samples, device=model.device)
        outputs = model(inputs)

        # Log reconstructions as 3D objects
        for i, recon in enumerate(outputs.recon.detach().cpu().numpy()):
            self.run.log({f'Recon {i}': wandb.Object3D(recon)})
