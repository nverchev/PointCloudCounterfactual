"""DRYTorch Model subtypes."""

from drytorch.core import protocols as p
from drytorch.lib.models import Model, EMAModel

from src.data import Inputs, Outputs
from src.config import Experiment
from src.config.specs import FlowExperimentConfig
from src.module import get_autoencoder, BaseVAE, get_flow_module, CondFlowMatching, FlowMatching, get_classifier


class LogEpochMixin(p.ModelProtocol[Inputs, Outputs]):
    """Mixin to log the epoch to the outputs."""

    def __call__(self, inputs: Inputs) -> Outputs:
        out = super().__call__(inputs)
        out.model_epoch = self.epoch
        return out


class ModelEpoch(LogEpochMixin, Model[Inputs, Outputs]):
    """Model with epoch logging."""


class EMAModelEpoch(LogEpochMixin, EMAModel[Inputs, Outputs]):
    """EMA Model with epoch logging."""


def load_extract_autoencoder_module() -> BaseVAE:
    """Load and extract the autoencoder module from the EMA model."""
    cfg = Experiment.get_config()
    autoencoder = get_autoencoder()
    model = EMAModel(autoencoder, name=cfg.autoencoder.model.name, device=cfg.user.device)
    model.load_state(-1)
    module = model.averaged_module
    module.eval()
    assert isinstance(module, BaseVAE)
    return module


def load_extract_flow_module(cfg_flow: FlowExperimentConfig) -> FlowMatching:
    """Load and extract the flow module from the EMA model."""
    cfg = Experiment.get_config()
    module = get_flow_module(cfg_flow).eval()
    model = EMAModel(module, name=cfg_flow.model.name, device=cfg.user.device)
    model.load_state(-1)
    ema_module = model.averaged_module
    assert isinstance(ema_module, FlowMatching)
    return ema_module


def load_extract_cond_flow_module(cfg_flow: FlowExperimentConfig, autoencoder: BaseVAE) -> CondFlowMatching:
    """Load and extract the flow module from the EMA model."""
    cfg = Experiment.get_config()
    module = get_flow_module(cfg_flow, autoencoder=autoencoder).eval()
    model = EMAModel(module, name=cfg_flow.model.name, device=cfg.user.device)
    model.load_state(-1)
    ema_module = model.averaged_module
    assert isinstance(ema_module, CondFlowMatching)
    return ema_module


def load_extract_classifier_module() -> Model:
    """Load and extract the classifier module."""
    cfg = Experiment.get_config()
    module = get_classifier().eval()
    model = Model(module, name=cfg.classifier.model.name, device=cfg.user.device)
    model.load_state()
    return model
