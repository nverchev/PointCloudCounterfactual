"""Hooks to execute during training."""

from torch.utils import data

from drytorch.core import protocols as p
from drytorch.lib.load import take_from_dataset

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
