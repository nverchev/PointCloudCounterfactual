"""Defines and registers the configuration for the experiment."""
import enum
import functools
import pathlib
import os
from typing import Optional, Self, Type, Annotated, Callable, TypeAlias

import dotenv
import dry_torch
import hydra
import numpy as np
from pydantic import model_validator, Field
from pydantic.dataclasses import dataclass
import torch

dotenv.load_dotenv()

DATASET_DIR = pathlib.Path(os.getenv('DATASET_DIR', ''))
EXPERIMENT_DIR = pathlib.Path(os.getenv('EXPERIMENT_DIR', './'))
METADATA_DIR = pathlib.Path(os.getenv('METADATA_DIR', './'))

PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]
ActClass: TypeAlias = Callable[[], torch.nn.Module]

ACT: ActClass = functools.partial(torch.nn.LeakyReLU, negative_slope=0.2)


def _get_activation_cls(act_name: str) -> ActClass:
    try:
        return getattr(torch.nn.modules.activation, act_name)
    except AttributeError:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.')


def _get_optim_cls(optimizer_name: str) -> Type[torch.optim.Optimizer]:
    try:
        return getattr(torch.optim, optimizer_name)
    except AttributeError:
        raise ValueError(f'Input opt_name "{optimizer_name}" is not the name of a pytorch optimizer.')


class Datasets(enum.StrEnum):
    """Dataset names."""
    ModelNet = enum.auto()
    ShapenetAtlas = enum.auto()


class Encoders(enum.StrEnum):
    """Encoder names."""
    LDGCNN = enum.auto()
    DGCNN = enum.auto()


class Decoders(enum.StrEnum):
    """Encoder names."""
    PCGen = enum.auto()


class WEncoders(enum.StrEnum):
    """Encoder names for the W-autoencoder."""
    Convolution = enum.auto()


class WDecoders(enum.StrEnum):
    """Decoder names for the W-autoencoder."""
    Convolution = enum.auto()
    Linear = enum.auto()


class ModelHead(enum.StrEnum):
    """Model type."""
    AE = enum.auto()
    VQVAE = enum.auto()


class Schedulers(enum.StrEnum):
    """Scheduler names."""
    Constant = enum.auto()
    Cosine = enum.auto()
    Exponential = enum.auto()


class ReconLosses(enum.StrEnum):
    """Loss names."""
    Chamfer = enum.auto()
    ChamferEMD = enum.auto()


@dataclass
class ExperimentSettings:
    """Global settings for the experiment."""
    main_name: str
    """Name of the main experiment"""
    child_name: str
    """Name of the child experiment"""
    final: bool
    """uses val dataset for training and test dataset for testing, otherwise test on val"""
    seed: Optional[int]
    """torch/numpy seed. Default: no seed"""

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return f"{self.child_name}{'_final' if self.final else ''}"

    def __post_init__(self):
        self._set_seed()

    def _set_seed(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(self.seed)
        return


@dataclass
class DatasetConfig:
    """Config for the dataset."""

    name: Datasets
    """dataset name"""
    settings: dict
    """settings specific for the dataset"""


@dataclass
class DataConfig:
    """Config for pre-processing the data."""

    dataset: DatasetConfig
    """dataset configuration"""
    input_points: StrictlyPositiveInt
    """(maximum) input points"""
    translation: bool
    """random translating training inputs"""
    rotation: bool
    """random rotating training inputs"""
    resample: bool
    """two different samplings for input and reference"""
    k: StrictlyPositiveInt
    """number of neighbours of a point (counting the point itself)"""


@dataclass
class WEncoderConfig:
    """Configuration for the VAE encoder."""

    architecture: WEncoders
    """name of the architecture"""
    hidden_dims_conv: list[StrictlyPositiveInt]
    """hidden dimensions for the convolution part of the WEncoder"""
    hidden_dims_lin: list[StrictlyPositiveInt]
    """hidden dimensions for the linear part of the WEncoder"""
    n_pseudo_inputs: StrictlyPositiveInt
    """number of pseudo_inputs for the VAMP loss"""
    dropout: list[PositiveFloat]
    """dropout probability"""
    act_name: str = ''
    """name of the Pytorch activation (see https://pytorch.org/docs/stable/nn.html)"""

    def __post_init__(self):
        self.act_cls = _get_activation_cls(self.act_name) if self.act_name else ACT

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.hidden_dims_lin) != len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims_lin), len(self.dropout)))
        return self


@dataclass
class EncoderConfig:
    """Configuration for the encoder."""

    architecture: Encoders
    """name of the architecture"""
    k: StrictlyPositiveInt
    """number of neighbours of a point (counting the point itself)"""
    hidden_dims: list[StrictlyPositiveInt]
    """hidden dimensions"""
    w_encoder: WEncoderConfig
    """config for the word encoder"""
    act_name: str = ''
    """name of the Pytorch activation (see https://pytorch.org/docs/stable/nn.html)"""

    def __post_init__(self):
        self.act_cls = _get_activation_cls(self.act_name) if self.act_name else ACT


@dataclass
class WDecoderConfig:
    """Configuration for the VAE decoder."""

    architecture: WDecoders
    """name of the architecture"""
    expand: StrictlyPositiveInt
    """feature expansion before final output"""
    hidden_dims: list[StrictlyPositiveInt]
    """hidden dimensions"""
    dropout: list[PositiveFloat]
    """dropout probability"""
    act_name: str = ''
    """name of the Pytorch activation (see https://pytorch.org/docs/stable/nn.html)"""

    def __post_init__(self):
        self.act_cls = _get_activation_cls(self.act_name) if self.act_name else ACT

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.hidden_dims) + int(self.architecture == WDecoders.Linear) != len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims), len(self.dropout)))
        return self


@dataclass
class DecoderConfig:
    """Configuration for the decoder."""

    architecture: Decoders
    """name of the architecture"""
    sample_dim: StrictlyPositiveInt
    """dimensions of the sampling sphere"""
    n_components: StrictlyPositiveInt
    """number of components of PCGen"""
    hidden_dims_map: list[StrictlyPositiveInt]
    """ Hidden dimensions  for mapping the initial sampling"""
    hidden_dims_conv: list[StrictlyPositiveInt]
    """hidden channels for each component"""
    w_decoder: WDecoderConfig
    """config for the word decoder"""
    tau: PositiveFloat
    """coefficient for Gumbel Softmax activation"""
    filtering: bool
    """filter applied to the output (see https://www.mdpi.com/1424-8220/24/5/1414/review_report)"""
    act_name: str = ''
    """name of the Pytorch activation (see https://pytorch.org/docs/stable/nn.html)"""

    def __post_init__(self):
        self.act_cls = _get_activation_cls(self.act_name) if self.act_name else ACT


@dataclass
class AEConfig:
    """Configuration for the autoencoder."""

    head: ModelHead
    """Simple encoding AE or the VQVAE generative model"""
    encoder: EncoderConfig
    """encoder name"""
    decoder: DecoderConfig
    """decoder model"""
    book_size: StrictlyPositiveInt
    """dictionary size"""
    embedding_dim: StrictlyPositiveInt
    """code length"""
    w_dim: StrictlyPositiveInt
    """codeword length"""
    z_dim: StrictlyPositiveInt
    """continuous latent space dim"""
    vq_ema_update: bool
    """EMA update on quantized codes"""
    vq_noise: PositiveFloat
    """noise when redistributing the codes"""
    double_encoding: bool
    """reconstructs samples based on the retrieved codes"""
    name: str
    """name of the model"""
    output_points: StrictlyPositiveInt
    """points generated in training, 0 for input number of points"""

    @property
    def num_codes(self) -> int:
        """Number of codes"""
        return self.w_dim // self.embedding_dim


@dataclass
class ClassifierConfig:
    """Configuration for the classifier."""

    k: StrictlyPositiveInt
    """number of neighbours of a point (counting the point itself)"""
    hidden_dims: list[StrictlyPositiveInt]
    """hidden dimensions"""
    emb_dim: StrictlyPositiveInt
    """dimension of the final convolution output dims."""
    mlp_dims: list[StrictlyPositiveInt]
    """dimensions for the MLP layers."""
    dropout: list[PositiveFloat]
    """dimensions for the MLP layers."""
    act_name: str = ''
    """name of the Pytorch activation (see https://pytorch.org/docs/stable/nn.html)."""
    name: str = ''
    """name of the model."""

    def __post_init__(self):
        self.act_cls = _get_activation_cls(self.act_name) if self.act_name else ACT

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.mlp_dims) != len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims), len(self.dropout)))
        return self


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    function: Schedulers
    """scheduler name"""
    settings: dict
    """default settings for the scheduler"""


@dataclass
class LearningConfig:
    """Configuration for the learning scheme."""

    optimizer_name: str
    """name of the Pytorch optimizer (see https://pytorch.org/docs/stable/optim.html)"""
    learning_rate: PositiveFloat
    """learning rate or dictionary of learning rate for the model parameters"""
    settings: dict
    """default settings for the optimizer"""
    scheduler: SchedulerConfig
    """scheduler configuration for the learning rate decay"""

    def __post_init__(self):
        self.optimizer_cls = _get_optim_cls(self.optimizer_name)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    metric_name: str = 'Criterion'
    """name of the metric to monitor"""

    patience: int = 10
    """number of epochs to wait before stopping"""


@dataclass
class TrainingConfig:
    """Configuration for the training."""

    batch_size: StrictlyPositiveInt
    """batch size"""
    diagnose_every: StrictlyPositiveInt
    """number of points between diagnostics (PCGen rearranges the discrete space)"""
    learn: LearningConfig
    """learning config for the training"""
    epochs: StrictlyPositiveInt
    """number of epochs for the training"""
    load_checkpoint: int = -1
    """load checkpoint if available. Default: last one"""
    early_stopping: Optional[EarlyStoppingConfig] = None
    """Configuration for early stopping"""


@dataclass
class ObjectiveAEConfig:
    """Configuration for the loss and the metrics."""

    recon_loss: ReconLosses
    """reconstruction loss denomination"""
    c_embedding: PositiveFloat
    """coefficient for the embedding loss"""


@dataclass
class ObjectiveWAEConfig:
    """Configuration for the loss and the metrics."""
    c_kld: PositiveFloat
    """Kullback-Leibler Divergence coefficient"""


@dataclass
class PlottingOptions:
    """Options for plotting and visualization."""

    training: bool
    """visualize learning curves"""
    interactive: bool
    """3D plot when using pyvista plotting"""
    refresh: StrictlyPositiveInt
    """epochs before refreshing the learning curves"""
    indices_to_reconstruct: list[PositiveInt]
    """indices of the evaluation test to reconstruct"""


@dataclass
class PathSpecs:
    """Path specifications."""

    exp_par_dir: pathlib.Path = EXPERIMENT_DIR
    """directory for containing the experiment. Default takes path specified in .env file"""
    data_dir: pathlib.Path = DATASET_DIR
    """directory for models. Default takes path specified in .env file"""
    metadata_dir: pathlib.Path = METADATA_DIR
    """directory for the metadata of the dataset. Default takes path specified in .env file"""


@dataclass
class GenerationOptions:
    """Options for generation of point clouds."""

    batch_size: StrictlyPositiveInt
    """number of samples to generate"""
    bias_dim: PositiveInt
    """add a bias to latent dim"""
    bias_value: float
    """value of the bias in latent dim"""


@dataclass
class UserSettings:
    """Options relative to the user preferences."""
    checkpoint_every: StrictlyPositiveInt
    """number of epochs between checkpoints"""
    cuda: bool
    """run on cuda:0"""
    generate: GenerationOptions
    """options for generation of point clouds"""
    path: PathSpecs
    """specifications for paths that override .env settings"""
    plot: PlottingOptions
    """options for plotting and visualization"""
    validate: StrictlyPositiveInt
    """epochs before validation"""

    @property
    def device(self) -> torch.device:
        """The device where to run the model."""
        return torch.device('cuda:0' if self.cuda else 'cpu')


@dataclass
class Config:
    """Shared options for training, evaluating and random generation."""

    exp: ExperimentSettings
    data: DataConfig
    train: TrainingConfig
    user: UserSettings


class ParentExperiment(dry_torch.ParentExperiment[None, Config]):
    pass


@dataclass
class ConfigTrainClassifier(Config):
    """Options for training the classifier."""
    classifier: ClassifierConfig


class ExperimentClassifier(dry_torch.ChildExperiment[ConfigTrainClassifier]):
    """Subclass for current config."""
    pass


@dataclass
class ConfigTrainAE(Config):
    """Options for training the autoencoder."""
    autoencoder: AEConfig
    objective: ObjectiveAEConfig


class ExperimentAE(dry_torch.ChildExperiment[ConfigTrainAE]):
    """Subclass for current config."""
    pass


@dataclass
class ConfigTrainWAE(Config):
    """Options for training the autoencoder."""
    autoencoder: AEConfig
    objective = ObjectiveWAEConfig


class ExperimentWAE(dry_torch.ChildExperiment[ConfigTrainWAE]):
    """Subclass for current config."""
    pass


cs = hydra.core.config_store.ConfigStore.instance()  # type: ignore
cs.store(name='config_classifier', node=ConfigTrainClassifier)
cs.store(name='config_ae', node=ConfigTrainAE)
cs.store(name='config_wae', node=ConfigTrainWAE)
