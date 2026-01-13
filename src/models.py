"""DRYTorch Model subtypes."""

from drytorch import Model
from drytorch.lib.models import Input, Output


class ModelEpoch(Model[Input, Output]):
    """This class adds a hook to include the epoch in the outputs (to anneal the kld loss)."""

    def __call__(self, inputs: Input) -> Output:
        outputs = super().__call__(inputs)
        outputs.model_epoch = self.epoch
        return outputs
