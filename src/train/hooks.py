"""Hooks to execute during training."""

from typing import Any

from torch.utils import data

from drytorch import Trainer
from drytorch.core import protocols as p
from drytorch.core.exceptions import TrackerNotUsedError
from drytorch.lib.hooks import EarlyStoppingCallback, Hook, call_every, saving_hook
from drytorch.lib.load import take_from_dataset
from drytorch.utils.averages import get_moving_average, get_trailing_mean

from src.data.structures import Inputs, Outputs, Targets


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


def register_checkpointing(trainer: Trainer, checkpoint_every: int | None) -> None:
    """Register the checkpointing hook."""
    if checkpoint_every:
        trainer.post_epoch_hooks.register(saving_hook.bind(call_every(checkpoint_every)))
    return


def register_early_stopping(trainer: Trainer, window: int, patience: int = 0) -> None:
    """Register the early stopping hook."""
    trainer.post_epoch_hooks.register(
        EarlyStoppingCallback(metric=trainer.objective, filter_fn=get_trailing_mean(window), patience=patience)
    )
    return


def register_pruning(trainer: Trainer, trial: Any) -> None:
    """Register the pruning hook."""
    from drytorch.contrib.optuna import TrialCallback

    prune_hook = TrialCallback(trial, metric=trainer.objective, filter_fn=get_moving_average())
    trainer.post_epoch_hooks.register(prune_hook)
    return


def register_reconstruction_hook(trainer: Trainer, restart_interval: int) -> None:
    """Register the reconstruction hook."""
    try:
        from src.train.hooks import TensorBoardLogReconstruction

        trainer.post_epoch_hooks.register(
            Hook(TensorBoardLogReconstruction(trainer.loader.dataset)).bind(call_every(restart_interval))
        )
    except TrackerNotUsedError:  # tracker is not subscribed
        pass
    except (ImportError, ModuleNotFoundError):  # library is not installed
        pass

    return
