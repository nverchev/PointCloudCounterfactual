"""Evaluation and logging for training."""

import logging

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torchmetrics import ConfusionMatrix

from drytorch import Test, Trainer
from drytorch.core.exceptions import TrackerNotUsedError
from drytorch.core import protocols as p

from src.data.structures import Inputs, Targets
from src.utils.visualization import plot_confusion_matrix_heatmap


class ClassificationEvaluation(Test):
    """Evaluation of classification performance, including confusion matrix and misclassified logging."""

    def __init__(
        self,
        model: p.ModelProtocol[Inputs, torch.Tensor],
        loader: p.LoaderProtocol[tuple[Inputs, Targets]],
        metric: p.ObjectiveProtocol[torch.Tensor, Targets],
        n_classes: int,
        class_names: list[str],
        name: str = 'Test',
    ):
        """Initialize the classification evaluation."""
        super().__init__(model=model, loader=loader, metric=metric, name=name)
        self.n_classes = n_classes
        self.class_names = class_names

    def log_results(self, trainer: Trainer) -> None:
        """Calculate and log classification metrics.

        Args:
            trainer: Training protocol containing the model and trackers.
        """
        if dist.is_initialized() and dist.get_rank() != 0:  # type: ignore[attr-defined]
            return

        if not self.outputs_list:
            logging.warning('No outputs recorded in evaluation. Did you run the evaluation first?')
            return

        predictions, labels = self._get_predictions_and_labels()
        misclassified_indices, misclassified_str = self._get_misclassified_info(predictions, labels)
        cf_matrix_numpy = self._get_confusion_matrix(predictions, labels)
        fig_cm = plot_confusion_matrix_heatmap(cf_matrix_numpy, self.class_names, title='Model Confusion Matrix')
        self._log_to_trackers(trainer, fig_cm, len(misclassified_indices), misclassified_str)
        self._log_to_logger(cf_matrix_numpy, misclassified_str)
        return

    @torch.inference_mode()
    def _get_predictions_and_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate outputs and collect labels from the loader."""
        outputs_probs = torch.cat(self.outputs_list)
        predictions = outputs_probs.argmax(dim=1)
        labels = torch.cat([batch_data[1].label for batch_data in self.loader])
        return predictions, labels

    def _get_misclassified_info(self, predictions: torch.Tensor, labels: torch.Tensor) -> tuple[list[int], str]:
        """Identify misclassified samples and format them as a string."""
        indices = [int(x.item()) for x in (predictions != labels).nonzero().squeeze(1)]
        max_log = 100
        s = str(indices[:max_log])
        if len(indices) > max_log:
            s += f' ... (and {len(indices) - max_log} more)'

        return indices, s

    def _get_confusion_matrix(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Calculate the confusion matrix using torchmetrics."""
        cf_matrix = ConfusionMatrix(task='multiclass', num_classes=self.n_classes)
        cf_matrix_tensor = cf_matrix(predictions.cpu(), labels.cpu())
        return cf_matrix_tensor.numpy()

    def _log_to_trackers(self, trainer: Trainer, fig, total_misclassified: int, misclassified_str: str) -> None:
        """Log results to available trackers (e.g., TensorBoard)."""
        try:
            from drytorch.trackers.tensorboard import TensorBoard

            writer = TensorBoard.get_current().writer

            if fig is not None:
                writer.add_figure(f'{trainer.model.name}/{self.name}-Confusion Matrix', fig)

            writer.add_text(
                f'{trainer.model.name}/{self.name}- Misclassified Indices',
                f'Total misclassified samples: {total_misclassified}\nIndices: {misclassified_str}',
                global_step=trainer.model.epoch,
            )
        except (TrackerNotUsedError, ImportError, ModuleNotFoundError):
            pass

        return

    def _log_to_logger(self, matrix: npt.NDArray[np.int_], misclassified_str: str) -> None:
        """Log results to the console with clean formatting."""
        # Format confusion matrix as a table
        col_width = 15
        header = ' ' * col_width + ''.join(f'{name:>{col_width}}' for name in self.class_names)
        rows = []
        for i, row in enumerate(matrix):
            row_str = f'{self.class_names[i]:>{col_width}}' + ''.join(f'{int(val):>{col_width}}' for val in row)
            rows.append(row_str)

        cm_table = '\n'.join([header, *rows])
        logging.info('Evaluation results for %s:', self.name)
        logging.info('Confusion Matrix:\n%s', cm_table)
        logging.info('Misclassified indices: %s', misclassified_str)
        return
