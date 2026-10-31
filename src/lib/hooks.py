"""Hooks to execute during training."""

import logging

from typing import Any, Final, override

import torch

from drytorch.lib import evaluations, hooks, monitoring
from drytorch.lib.training import TrainerBuilder
from drytorch.utils.averages import get_trailing_mean

from src.data.structures import Inputs, Outputs, Targets, get_quantization
from src.module import VQVAE

__all__ = [
    'ReviveCodebook',
    'register_checkpointing',
    'register_codebook_revival',
    'register_early_stopping',
]


class ReviveCodebook(hooks.TrainerHook[monitoring.Condition]):
    """Move the codes no sample chose next to a code that is used.

    A code the encoder never picks gets no gradient, so it stays where it was initialized and the
    codebook loses capacity. Moving it onto a code chosen often, with noise, gives it a share of
    that code's samples to claim. At the last epoch no noise is added, so that a code the decoder
    was not trained on decodes like the one it copies.

    Attributes:
        diagnostic: Pass over the training clouds, collecting the chosen codes.
        module: VQ-VAE whose codebook is revived.
        noise: Scale of the noise added to the code an unused one is moved onto.
        final_epoch: Epoch of the end of training.
        when: Decides whether the hook runs when the trainer calls it.
    """

    diagnostic: Final[evaluations.Evaluation[Inputs, Targets, Outputs]]
    module: Final[VQVAE]
    noise: Final[float]
    final_epoch: Final[int]

    def __init__(
        self,
        diagnostic: evaluations.Evaluation[Inputs, Targets, Outputs],
        module: VQVAE,
        /,
        *,
        noise: float,
        final_epoch: int,
        when: monitoring.Condition = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            diagnostic: pass over the training clouds, which stores the codes every sample chose.
            module: VQ-VAE whose codebook is revived.
            noise: scale of the noise added to the code an unused one is moved onto.
            final_epoch: epoch of the end of training.
            when: decides whether the hook runs. Default runs every time.
        """
        super().__init__(when=when)
        self.diagnostic = diagnostic
        self.module = module
        self.noise = noise
        self.final_epoch = final_epoch
        return

    @override
    def run(self) -> None:
        """Replace every unused code and report how much of the book is in use.

        Side Effects:
            Evaluates the model on the training clouds and writes the codebook.
        """
        self.diagnostic(store_outputs=True)
        chosen = torch.vstack([get_quantization(outputs).one_hot_idx for outputs in self.diagnostic.outputs_list])
        counts = chosen.sum(0)
        unused = counts == 0
        logging.info(f'Codebook usage: {1 - unused.float().mean().item():.1%} of the codes.')
        codebook = self.module.codebook.data
        noise = 0.0 if self.trainer.model.epoch == self.final_epoch else self.noise
        for group in range(counts.shape[0]):
            weights = counts[group] / counts[group].sum()
            for code in torch.nonzero(unused[group]).flatten().tolist():
                donor = int(torch.multinomial(weights, 1).item())
                used = codebook[group, donor]
                codebook[group, code] = used + noise * torch.randn_like(used)

        return


def register_checkpointing(builder: TrainerBuilder[Any, Any, Any], /, *, checkpoint_every: int) -> None:
    """Register the checkpointing hook.

    Args:
        builder: trainer builder to register the saving hook to.
        checkpoint_every: save frequency in epochs; ``0`` adds no saving.
    """
    if checkpoint_every:
        builder.add_saving(checkpoint_every)

    return


def register_codebook_revival(
    builder: TrainerBuilder[Any, Any, Any],
    /,
    *,
    diagnostic: evaluations.Evaluation[Inputs, Targets, Outputs],
    module: VQVAE,
    noise: float,
    revive_every: int,
    final_epoch: int,
) -> None:
    """Register the codebook revival hook.

    Args:
        builder: trainer builder to register the hook to.
        diagnostic: pass over the training clouds collecting the chosen codes.
        module: VQ-VAE whose codebook is revived.
        noise: scale of the noise added to the code an unused one is moved onto.
        revive_every: epochs between two revivals.
        final_epoch: epoch of the end of training.
    """
    hook = ReviveCodebook(diagnostic, module, noise=noise, final_epoch=final_epoch, when=monitoring.Every(revive_every))
    builder.add_post_epoch_hook(hook)
    return


def register_early_stopping(builder: TrainerBuilder[Any, Any, Any], /, *, patience: int, window: int) -> None:
    """Register the early stopping hook.

    Args:
        builder: trainer builder to register early stopping to.
        patience: number of evaluations without improvement before stopping.
        window: trailing window size for the smoothed metric.
    """
    builder.add_early_stopping(patience=patience, watch=builder.watch(filter_fn=get_trailing_mean(window)))
    return
