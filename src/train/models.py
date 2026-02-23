"""DRYTorch Model subtypes."""

from drytorch.core import protocols as p
from drytorch.lib.models import EMAModel as EMAModelBase

from src.data import Inputs, Outputs


class LogEpochMixin(p.ModelProtocol[Inputs, Outputs]):
    """Mixin to log the epoch to the outputs."""

    def __call__(self, inputs: Inputs) -> Outputs:
        out = super().__call__(inputs)
        out.model_epoch = self.epoch
        return out


class EMAModel(LogEpochMixin, EMAModelBase[Inputs, Outputs]):
    """EMA Model with epoch logging."""
