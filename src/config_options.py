"""Defines and registers the configuration for the experiment."""

import enum
import functools
import pathlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import field
from typing import Optional, Self, Annotated, Callable, TypeAlias, cast

import dotenv
import drytorch
import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
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


class ConfigPath(enum.StrEnum):
    """Configuration paths relative to the project root."""
    CONFIG_ALL = 'config_all'
    TUNE_AUTOENCODER = 'tune_autoencoder'
    TUNE_W_AUTOENCODER = 'tune_w_autoencoder'

    @classmethod
    def get_folder(cls) -> str:
        """Return folder_name."""
        return 'hydra_conf'

    def get_path(self) -> pathlib.Path:
        """Return folder path."""
        return pathlib.Path(__file__).parent.parent / self.get_folder() / self

    def absolute(self) -> str:
        """Absolute path to folder"""
        return str(self.get_path().absolute().resolve())

    def relative(self) -> str:
        """Relative path to folder"""
        return f'../{self.get_folder()}/{self}'


def _get_activation_cls(act_name: str) -> ActClass:
    try:
        return getattr(torch.nn.modules.activation, act_name)
    except AttributeError:
        raise ValueError(f'Input act_name "{act_name}" is not the name of a pytorch activation.')


def _get_optim_cls(optimizer_name: str) -> type[torch.optim.Optimizer]:
    try:
        return getattr(torch.optim, optimizer_name)
    except AttributeError:
        raise ValueError(f'Input opt_name "{optimizer_name}" is not the name of a pytorch optimizer.')


class Datasets(enum.StrEnum):
    """Dataset names."""
    ModelNet = enum.auto()
    ShapenetFlow = enum.auto()


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
    Transformers = enum.auto()


class WDecoders(enum.StrEnum):
    """Decoder names for the W-autoencoder."""
    Convolution = enum.auto()
    Linear = enum.auto()
    TransformerCross = enum.auto()


class ModelHead(enum.StrEnum):
    """Model type."""
    AE = enum.auto()
    VQVAE = enum.auto()
    CounterfactualVQVAE = enum.auto()

class GradOp(enum.StrEnum):
    """Gradient operation names."""
    GradParamNormalizer = enum.auto()
    GradZScoreNormalizer = enum.auto()
    GradNormClipper = enum.auto()
    GradValueClipper = enum.auto()
    HistClipper = enum.auto()
    ParamHistClipper = enum.auto()
    NoOp = enum.auto()

class ClipCriterion(enum.StrEnum):
    """Clipping criterion names."""
    ZStat = enum.auto()
    EMA = enum.auto()


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
class DatasetConfig:
    """Configuration for the dataset.

    Attributes:
        name (Datasets): The name of the dataset
        n_classes (PositiveInt): The number of classes in the dataset, 0 if no class information is available
        settings (dict): A dictionary containing dataset-specific settings
    """

    name: Datasets
    n_classes: PositiveInt
    settings: dict


@dataclass
class DataConfig:
    """Configuration for pre-processing the data.

    Attributes:
        dataset (DatasetConfig): The dataset configuration
        n_input_points (StrictlyPositiveInt): The (maximum) number of input points
        n_target_points (StrictlyPositiveInt): The (maximum) number of target points
        translation (bool): Whether to apply random translation to training inputs and reference
        rotation (bool): Whether to apply random rotation to training inputs and reference
        jitter_sigma (PositiveFloat): The variance for random perturbation of training inputs
        jitter_clip (PositiveFloat): The threshold value for random perturbation of training inputs
        resample (bool): Whether to use two different samplings for input and reference
        k (StrictlyPositiveInt): The number of neighbors of a point (counting the point itself)
    """

    dataset: DatasetConfig
    n_input_points: StrictlyPositiveInt
    n_target_points: StrictlyPositiveInt
    translation: bool
    rotation: bool
    jitter_sigma: PositiveFloat
    jitter_clip: PositiveFloat
    resample: bool
    k: StrictlyPositiveInt


@dataclass
class WEncoderConfig:
    """Configuration for the W-Autoencoder's encoder.

    Attributes:
        architecture (WEncoders): The name of the architecture
        hidden_dims_conv (tuple[StrictlyPositiveInt]): Hidden dimensions for the convolutional part
        hidden_dims_lin(tuple[StrictlyPositiveInt]): Hidden dimensions for the linear part
        n_heads (StrictlyPositiveInt): Number of attention heads for self-attention
        proj_dim (StrictlyPositiveInt): Number of dimensions in expanded embedding
        dropout (tuple[PositiveFloat]): Dropout probabilities for the linear layers
        cf_temperature (int): Temperature for the probabilities (closer to zero means closer to samplings)
        act_name (str): The name of the PyTorch activation function (e.g., 'ReLU', 'LeakyReLU')
        gumbel (bool): Whether to use Gumbel Softmax to add noise to the codes
    """

    architecture: WEncoders
    hidden_dims_conv: tuple[StrictlyPositiveInt, ...]
    hidden_dims_lin: tuple[StrictlyPositiveInt, ...]
    n_heads: StrictlyPositiveInt
    proj_dim: StrictlyPositiveInt
    dropout: tuple[PositiveFloat, ...]
    cf_temperature: int
    act_name: str = ''
    gumbel: bool = True

    @property
    def act_cls(self) -> ActClass:
        """The activation class."""
        return _get_activation_cls(self.act_name) if self.act_name else ACT
    
    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.hidden_dims_lin) > len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims_lin), len(self.dropout)))
        return self


@dataclass
class EncoderConfig:
    """Configuration for the encoder.

    Attributes:
        architecture (Encoders): The name of the architecture
        k (StrictlyPositiveInt): The number of neighbors for a point (counting the point itself)
        hidden_dims (tuple[StrictlyPositiveInt]): Hidden dimensions for the encoder
        w_encoder (WEncoderConfig): Configuration for the word encoder
        act_name (str): The name of the PyTorch activation function
    """

    architecture: Encoders
    k: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    w_encoder: WEncoderConfig
    act_name: str = ''

    @property
    def act_cls(self) -> ActClass:
        """The activation class."""
        return _get_activation_cls(self.act_name) if self.act_name else ACT


@dataclass
class WDecoderConfig:
    """Configuration for the W-Autoencoder's decoder.

    Attributes:
        architecture (WDecoders): The name of the architecture
        n_heads (StrictlyPositiveInt): Number of attention heads for self-attention
        proj_dim (StrictlyPositiveInt): Number of dimensions in expanded embedding
        hidden_dims (tuple[StrictlyPositiveInt]): Hidden dimensions for the decoder
        dropout (tuple[PositiveFloat]): Dropout probabilities
        act_name (str): The name of the PyTorch activation function
    """

    architecture: WDecoders
    n_heads: StrictlyPositiveInt
    proj_dim: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    dropout: tuple[PositiveFloat, ...]
    act_name: str = ''

    @property
    def act_cls(self):
        """The activation class."""
        return _get_activation_cls(self.act_name) if self.act_name else ACT

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.hidden_dims) > len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims), len(self.dropout)))
        return self


@dataclass
class PosteriorDecoderConfig:
    """Configuration for the decoder of the latent space prior.

    Attributes:
        hidden_dims (tuple[StrictlyPositiveInt]): Hidden dimensions
        n_heads (StrictlyPositiveInt): Number of attention heads for self-attention
        dropout (tuple[PositiveFloat]): Dropout probabilities
        act_name (str): The name of the PyTorch activation function
    """

    hidden_dims: tuple[StrictlyPositiveInt, ...]
    n_heads: StrictlyPositiveInt
    dropout: tuple[PositiveFloat, ...]
    act_name: str = ''

    @property
    def act_cls(self):
        """The activation class."""
        return _get_activation_cls(self.act_name) if self.act_name else ACT

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.hidden_dims) > len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims), len(self.dropout)))
        return self


@dataclass
class DecoderConfig:
    """Configuration for the decoder.

    Attributes:
        architecture (Decoders): The name of the architecture
        sample_dim (StrictlyPositiveInt): Dimensions of the sampling sphere
        n_components (StrictlyPositiveInt): Number of components for PCGen
        hidden_dims_map (tuple[StrictlyPositiveInt]): Hidden dimensions for mapping the initial sampling
        hidden_dims_conv (tuple[StrictlyPositiveInt]): Hidden channels for each component
        w_decoder (WDecoderConfig): Configuration for the word decoder
        posterior_decoder (PosteriorDecoderConfig): Configuration for the posterior distribution of z2
        tau (PositiveFloat): Coefficient for Gumbel Softmax activation
        filtering (bool): Whether to apply filtering to the output
        act_name (str): The name of the PyTorch activation function
    """

    architecture: Decoders
    sample_dim: StrictlyPositiveInt
    n_components: StrictlyPositiveInt
    hidden_dims_map: tuple[StrictlyPositiveInt, ...]
    hidden_dims_conv: tuple[StrictlyPositiveInt, ...]
    w_decoder: WDecoderConfig
    posterior_decoder: PosteriorDecoderConfig
    tau: PositiveFloat
    filtering: bool
    act_name: str = ''

    @property
    def act_cls(self):
        """The activation class."""
        return _get_activation_cls(self.act_name) if self.act_name else ACT


@dataclass
class AEConfig:
    """Configuration for the autoencoder.

    Attributes:
        head (ModelHead): The type of model head (e.g., AE, VQVAE, CounterfactualVQVAE)
        encoder (EncoderConfig): The encoder configuration
        decoder (DecoderConfig): The decoder configuration
        book_size (StrictlyPositiveInt): The size of the dictionary for VQ-VAE
        embedding_dim (StrictlyPositiveInt): The length of the code embedding
        w_dim (StrictlyPositiveInt): The codeword length
        z1_dim (StrictlyPositiveInt): The continuous latent space dimension
        z2_dim (StrictlyPositiveInt): The continuous latent space dimension for counterfactual manipulation
        vq_ema_update (bool): Whether to use EMA update on quantized codes
        vq_noise (PositiveFloat): Quantity of noise to add when redistributing the codes
        name (str): The name of the model
        training_output_points (StrictlyPositiveInt): The number of points generated during training (0: same as input)
        n_pseudo_inputs (PositiveInt): The number of pseudo-inputs for the VAMP loss
    """

    head: ModelHead
    encoder: EncoderConfig
    decoder: DecoderConfig
    book_size: StrictlyPositiveInt
    embedding_dim: StrictlyPositiveInt
    w_dim: StrictlyPositiveInt
    z1_dim: StrictlyPositiveInt
    z2_dim: StrictlyPositiveInt
    vq_ema_update: bool
    vq_noise: PositiveFloat
    name: str
    training_output_points: StrictlyPositiveInt

    n_pseudo_inputs: PositiveInt

    @property
    def n_codes(self) -> int:
        """The number of codes."""
        return self.w_dim // self.embedding_dim

    @property
    def hidden_features(self) -> int:
        """The number of codes."""
        return self.encoder.w_encoder.hidden_dims_conv[-1]

@dataclass
class ClassifierConfig:
    """Configuration for the classifier.

    Attributes:
        k (StrictlyPositiveInt): The number of neighbors for a point (counting the point itself)
        hidden_dims (tuple[StrictlyPositiveInt]): Hidden dimensions for the convolutional part
        emb_dim (StrictlyPositiveInt): The dimension of the final convolution output
        mlp_dims (tuple[StrictlyPositiveInt]): Dimensions for the MLP layers
        dropout (tuple[PositiveFloat]): Dropout probabilities for the MLP layers
        out_classes (StrictlyPositiveInt): The number of output classes for the classifier
        act_name (str): The name of the PyTorch activation function
        name (str): The name of the model
    """

    k: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    emb_dim: StrictlyPositiveInt
    mlp_dims: tuple[StrictlyPositiveInt, ...]
    dropout: tuple[PositiveFloat, ...]
    out_classes: StrictlyPositiveInt
    act_name: str = ''
    name: str = ''

    @property
    def act_cls(self):
        """The activation class."""
        return _get_activation_cls(self.act_name) if self.act_name else ACT

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.mlp_dims) > len(self.dropout):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.hidden_dims), len(self.dropout)))
        return self


@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler.

    Attributes:
        function (Schedulers): The name of the scheduler function
        restart_interval (PositiveInt): The number of epochs between restarts
        restart_fraction (PositiveInt): The fraction of the base learning rate when restarting
        warmup_steps (int): The number of initial epochs with linearly increasing learning rate
        settings (dict): A dictionary containing default settings for the scheduler
    """

    function: Schedulers
    restart_interval: PositiveInt
    restart_fraction: PositiveInt
    warmup_steps: PositiveInt
    settings: dict


@dataclass
class LearningConfig:
    """Configuration for the learning scheme.

    Attributes:
        optimizer_name (str): The name of the PyTorch optimizer (e.g., 'Adam', 'SGD')
        learning_rate (PositiveFloat): The learning rate or a dictionary of learning rates for model parameters
        grad_op (GradOp | None): The gradient operation to be applied before the optimizer
        clip_criterion (ClipCriterion): The criterion for gradient clipping (only used for some gradient operations)
        opt_settings (dict): A dictionary containing default settings for the optimizer
        scheduler (SchedulerConfig): The scheduler configuration for learning rate decay
    """

    optimizer_name: str
    learning_rate: PositiveFloat
    grad_op: GradOp | None
    clip_criterion: ClipCriterion
    opt_settings: dict
    scheduler: SchedulerConfig

    @property
    def optimizer_cls(self) -> type[torch.optim.Optimizer]:
        """The optimizer class."""
        return _get_optim_cls(self.optimizer_name)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping during training.

    Attributes:
        active (bool): Whether to use the early stopping strategy
        window (int): The number of last metrics to average
        patience (int): The number of epochs to wait before stopping
    """

    active: bool
    window: int = 1
    patience: int = 10


@dataclass
class TrainingConfig:
    """Configuration for the training process.

    Attributes:
        batch_size (StrictlyPositiveInt): The batch size for training
        learn (LearningConfig): The learning configuration for training
        n_epochs (StrictlyPositiveInt): The total number of epochs for training
        early_stopping (EarlyStoppingConfig): The configuration for early stopping
    """
    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig


@dataclass
class ObjectiveAEConfig:
    """Configuration for the Autoencoder's loss and metrics.

    Attributes:
        n_inference_output_points (StrictlyPositiveInt): The number of inference points for evaluation
        recon_loss (ReconLosses): The denomination of the reconstruction loss
        c_embedding (PositiveFloat): The coefficient for the embedding loss
    """
    n_inference_output_points: StrictlyPositiveInt
    recon_loss: ReconLosses
    c_embedding: PositiveFloat


@dataclass
class ObjectiveWAEConfig:
    """Configuration for the W-Autoencoder's loss and metrics.

    Attributes:
        vamp (bool): Whether to use the Variational Mixture of Posteriors (VAMP) loss
        c_kld1 (PositiveFloat): The Kullback-Leibler Divergence coefficient for the first latent variable
        c_kld2 (PositiveFloat): The Kullback-Leibler Divergence coefficient for the second latent variable
        c_counterfactual (PositiveFloat): The coefficient for the Counterfactual loss
    """
    vamp: bool
    c_kld1: PositiveFloat
    c_kld2: PositiveFloat
    c_counterfactual: PositiveFloat


@dataclass
class PlottingOptions:
    """Options for plotting and visualization.

    Attributes:
        interactive (bool): Whether to enable 3D plotting when using pyvista
        indices_to_reconstruct (list[PositiveInt]): A list with indices of evaluation test samples to reconstruct
        double_encoding (bool): Whether to reconstruct samples based on the retrieved codes
    """

    interactive: bool
    indices_to_reconstruct: list[PositiveInt]
    double_encoding: bool


@dataclass
class PathSpecs:
    """Path specifications for directories.

    Attributes:
        exp_par_dir (pathlib.Path): The directory for containing the experiment. Default is the path in the .env file
        data_dir (pathlib.Path): The directory for datasets. Default is the path specified in the .env file
        metadata_dir (pathlib.Path): The directory for dataset metadata. Default is the path specified in the .env file
    """

    exp_par_dir: pathlib.Path = EXPERIMENT_DIR
    data_dir: pathlib.Path = DATASET_DIR
    metadata_dir: pathlib.Path = METADATA_DIR


@dataclass
class GenerationOptions:
    """Options for the generation of point clouds.

    Attributes:
        batch_size (StrictlyPositiveInt): The number of samples to generate in a batch
        bias_dim (PositiveInt): The dimension to add a bias to in the latent space
        bias_value (float): The value of the bias to add in the latent dimension
    """

    batch_size: StrictlyPositiveInt
    bias_dim: PositiveInt
    bias_value: float


@dataclass
class UserSettings:
    """User-specific options and preferences.

    Attributes:
        cuda (bool): Whether to run computations on `cuda:0`
        n_workers: The number of workers for data loading
        n_parallel_training_processes: The number of parallel training processes
        generate (GenerationOptions): Options for generating a point cloud
        path (PathSpecs): Specifications for paths that override .env settings
        plot (PlottingOptions): Options for plotting and visualization
        seed (Optional[int]): The seed for PyTorch/NumPy randomness. If None, no seed is set
        checkpoint_every (PositiveInt): The number of epochs between saving checkpoints
        n_generated_output_points (int): The number of points to generate during inference
        load_checkpoint (int): The checkpoint to load if available. Default is the last one (-1)
        counterfactual_value (PositiveFloat): The value for counterfactual strength
    """

    cuda: bool
    n_workers: PositiveInt
    n_parallel_training_processes: PositiveInt
    generate: GenerationOptions
    path: PathSpecs
    plot: PlottingOptions
    seed: Optional[int]
    checkpoint_every: PositiveInt
    n_generated_output_points: int
    load_checkpoint: int = -1
    counterfactual_value: PositiveFloat = 1.

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

    @property
    def device(self) -> torch.device:
        """The device where the model should run."""
        return torch.device('cuda:0' if self.cuda else 'cpu')


@dataclass
class ConfigTrain:
    """Specifications for the current experiment.

    Attributes:
        train (TrainingConfig): Training options
        name (str): The name of the child experiment
    """
    train: TrainingConfig
    name: str


@dataclass
class ConfigTrainClassifier(ConfigTrain):
    """Configuration for training the classifier.

    Attributes:
        train (TrainingConfig): Training options
        name (str): The name of the child experiment
        architecture (ClassifierConfig): The classifier architecture configuration.
    """
    architecture: ClassifierConfig


@dataclass
class ConfigTrainAE(ConfigTrain):
    """Configuration for training the autoencoder.

    Attributes:
        train (TrainingConfig): Training options
        name (str): The name of the child experiment
        architecture (AEConfig): The autoencoder architecture configuration
        objective (ObjectiveAEConfig): The autoencoder objective (loss and metrics) configuration
        diagnose_every (StrictlyPositiveInt): The number of points between diagnostics (rearranging the discrete space)
    """
    architecture: AEConfig
    objective: ObjectiveAEConfig
    diagnose_every: StrictlyPositiveInt


@dataclass
class ConfigTrainWAE(ConfigTrain):
    """Configuration for training the W-autoencoder.

    Attributes:
        train (TrainingConfig): Training options
        name (str): The name of the child experiment
        objective (ObjectiveWAEConfig): The W-autoencoder objective (loss and metrics) configuration
    """
    objective: ObjectiveWAEConfig


@dataclass
class ConfigAll:
    """Root configuration for all experiment settings.

    Attributes:
        variation (str): The name for the experiment
        final (bool): If True, it uses the validation dataset for training and the test dataset for testing
        classifier (ConfigTrainClassifier): The configuration for training the classifier
        autoencoder (ConfigTrainAE): The configuration for training the autoencoder
        w_autoencoder (ConfigTrainWAE): The configuration for training the W-autoencoder
        user (UserSettings): User-specific settings
        data (DataConfig): Data pre-processing configuration
    """
    variation: str
    final: bool
    version: str
    classifier: ConfigTrainClassifier
    autoencoder: ConfigTrainAE
    w_autoencoder: ConfigTrainWAE
    user: UserSettings
    data: DataConfig
    variant: str = 'default'
    tags: list[str] = field(default_factory=list)
    _lens: ConfigTrain | None = None

    @property
    def name(self) -> str:
        """The full name of the experiment, indicating if metrics are calculated on the test dataset."""
        out = f'{self.variation}_final' if self.final else self.variation
        return out[:255]

    @property
    def project(self) -> str:
        """Project name for wandb logging."""
        return 'PointCloudCounterfactual' + str(self.version)

    @property
    def lens(self) -> ConfigTrain:
        """The section of the configuration that is currently focused on."""
        if self._lens is None:
            raise ValueError('lens not set.')
        return self._lens

    @contextmanager
    def focus(self, section: ConfigTrain) -> Iterator[Self]:
        """Context manager that focuses on a specific config section."""
        if self._lens is not None:
            raise ValueError('lens already set.')

        self._lens = section
        try:
            yield self
        finally:
            self._lens = None

class Experiment(drytorch.Experiment[ConfigAll]):
    """Specifications for the current experiment."""
    pass


def get_config_all(overrides: Optional[list[str]] = None) -> ConfigAll:
    """Get hydra configuration without starting a run."""
    GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path=ConfigPath.CONFIG_ALL.relative()):
        dict_cfg = hydra.compose(config_name='defaults', overrides=overrides)
        cfg = cast(ConfigAll, OmegaConf.to_object(dict_cfg))

        if overrides is not None:
            update_exp_name(cfg, overrides)
        return cfg

def get_current_hydra_dir() -> pathlib.Path:
    """Get the path to the current hydra run."""
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    return pathlib.Path(hydra_config.runtime.output_dir)


def hydra_main(func: Callable[[ConfigAll], None]) -> Callable[[], None]:
    """Start hydra run and Converts dict_cfg to ConfigAll."""

    @hydra.main(version_base=None, config_path=str(ConfigPath.CONFIG_ALL.absolute()), config_name='defaults')
    @functools.wraps(func)
    def wrapper(dict_cfg: DictConfig) -> None:
        """Convert configuration to the stored object"""
        cfg = cast(ConfigAll, OmegaConf.to_object(dict_cfg))
        overrides = HydraConfig.get().overrides.task
        update_exp_name(cfg, overrides)
        drytorch.init_trackers(mode='hydra')
        return func(cfg)

    return wrapper.__call__


def update_exp_name(cfg: ConfigAll, overrides: list[str]) -> None:
    """Adds the overrides to the name for the experiment."""
    overrides = [override for override in overrides
                 if override.split('.')[0] != 'user' and
                 override.split('=')[0] not in ('final', 'variation')]
    cfg.variation = '_'.join([cfg.variation] + overrides).replace('/', '_')
    cfg.tags = overrides
    return


cs = hydra.core.config_store.ConfigStore.instance()  # type: ignore
cs.store(name='config_all', node=ConfigAll)
