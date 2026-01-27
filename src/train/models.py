"""DRYTorch Model subtypes."""

from drytorch import Model

from src.data import WInputs, Outputs


class ModelEpoch(Model[WInputs, Outputs]):
    """This class adds a hook to include the epoch in the outputs (to anneal the kld loss)."""

    def __call__(self, inputs: WInputs) -> Outputs:
        out = super().__call__(inputs)
        out.model_epoch = self.epoch
        return out
