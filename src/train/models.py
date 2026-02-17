"""DRYTorch Model subtypes."""

from drytorch import Model

from src.data import Inputs, Outputs


class ModelEpoch(Model[Inputs, Outputs]):
    """This class adds a hook to include the epoch in the outputs (to anneal the kld loss)."""

    def __call__(self, inputs: Inputs) -> Outputs:
        out = super().__call__(inputs)
        out.model_epoch = self.epoch
        return out
