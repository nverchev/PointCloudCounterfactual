"""Specification for the configuration files."""

import dataclasses
import pathlib

from contextlib import contextmanager
from typing import Any, Annotated, Self
from collections.abc import Iterator

import torch

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.config.environment import EnvSettings, VERSION
from src.config.torch import ActClass, get_activation_cls, get_optim_cls, set_seed, DEFAULT_ACT
from src.config.options import (
    Datasets,
    Encoders,
    Decoders,
    WEncoders,
    WDecoders,
    ModelHead,
    GradOp,
    ClipCriterion,
    Schedulers,
    ReconLosses,
)

PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


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
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)


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
        gumbel (bool): Whether to use Gumbel Softmax to add noise to the codes
        act_name (str): The name of the PyTorch activation function (e.g., 'ReLU', 'LeakyReLU')
        act_cls (ActClass): The activation class
    """

    architecture: WEncoders
    hidden_dims_conv: tuple[StrictlyPositiveInt, ...]
    hidden_dims_lin: tuple[StrictlyPositiveInt, ...]
    n_heads: StrictlyPositiveInt
    proj_dim: StrictlyPositiveInt
    dropout: tuple[PositiveFloat, ...]
    cf_temperature: int
    gumbel: bool = True
    act_name: str = ''
    act_cls: ActClass = DEFAULT_ACT

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        if self.act_name:
            self.act_cls = get_activation_cls(self.act_name)

        return

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
        act_cls (ActClass): The activation class
    """

    architecture: Encoders
    k: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    w_encoder: WEncoderConfig
    act_name: str = ''
    act_cls: ActClass = DEFAULT_ACT

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        if self.act_name:
            self.act_cls = get_activation_cls(self.act_name)

        return


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
        act_cls (ActClass): The activation class
    """

    architecture: WDecoders
    n_heads: StrictlyPositiveInt
    proj_dim: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    dropout: tuple[PositiveFloat, ...]
    act_name: str = ''
    act_cls: ActClass = DEFAULT_ACT

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        if self.act_name:
            self.act_cls = get_activation_cls(self.act_name)

        return

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
        act_cls (ActClass): The activation class
    """

    hidden_dims: tuple[StrictlyPositiveInt, ...]
    n_heads: StrictlyPositiveInt
    dropout: tuple[PositiveFloat, ...]
    act_name: str = ''
    act_cls: ActClass = DEFAULT_ACT

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        if self.act_name:
            self.act_cls = get_activation_cls(self.act_name)

        return

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
        act_cls (ActClass): The activation class
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
    act_cls: ActClass = DEFAULT_ACT

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        if self.act_name:
            self.act_cls = get_activation_cls(self.act_name)

        return


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
        name (str): The name of the model
        act_name (str): The name of the PyTorch activation function
        act_cls (ActClass): The activation class
    """

    k: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    emb_dim: StrictlyPositiveInt
    mlp_dims: tuple[StrictlyPositiveInt, ...]
    dropout: tuple[PositiveFloat, ...]
    out_classes: StrictlyPositiveInt
    name: str = ''
    act_name: str = ''
    act_cls: ActClass = DEFAULT_ACT

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        if self.act_name:
            self.act_cls = get_activation_cls(self.act_name)

        return

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
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)


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
        optimizer_cls (type[torch.optim.Optimizer]): The optimizer class
    """

    optimizer_name: str
    learning_rate: PositiveFloat
    grad_op: GradOp | None
    clip_criterion: ClipCriterion
    scheduler: SchedulerConfig
    opt_settings: dict[str, Any] = dataclasses.field(default_factory=dict)
    optimizer_cls: type[torch.optim.Optimizer] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Resolve optimizer class from name."""
        self.optimizer_cls = get_optim_cls(self.optimizer_name)
        return


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
        batch_size (StrictlyPositiveInt): The batch size for training of all the processes combined
        learn (LearningConfig): The learning configuration for training
        n_epochs (StrictlyPositiveInt): The total number of epochs for training
        early_stopping (EarlyStoppingConfig): The configuration for early stopping
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig
    _n_subprocesses: PositiveInt

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if self._n_subprocesses and self.batch_size % self._n_subprocesses != 0:
            msg = 'Global batch size {} not divisible by number of devices {}.'
            raise ValueError(msg.format(self.batch_size, self._n_subprocesses))

        return self

    @property
    def batch_size_per_device(self) -> int:
        """The batch size per device."""
        if self._n_subprocesses == 0:
            return self.batch_size

        return self.batch_size // self._n_subprocesses


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
        sample_indices (list[PositiveInt]): A list with indices of the evaluation samples to plot
    """

    interactive: bool
    sample_indices: list[PositiveInt]


@dataclass
class PathSpecs:
    """Path specifications for directories.

    Attributes:
        root_exp_dir (pathlib.Path): The directory for containing the experiment. Default is the path in the .env file
        data_dir (pathlib.Path): The directory for datasets. Default is the path specified in the .env file
        metadata_dir (pathlib.Path): The directory for dataset metadata. Default is the path specified in the .env file
    """

    _env = EnvSettings()
    root_exp_dir: pathlib.Path = _env.root_exp_dir
    data_dir: pathlib.Path = _env.dataset_dir
    metadata_dir: pathlib.Path = _env.metadata_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """The full path for the version directory."""
        return self.root_exp_dir / f'v{VERSION}'


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
class TrackerList:
    """List of trackers to use for logging.

    Attributes:
        wandb: use the Wandb tracker
        hydra: use the HydraLink tracker
        csv: use the CSVDumper tracker
        tensorboard: use the TensorBoard tracker
        sqlalchemy: use the SQLAlchemyConnection tracker
    """

    wandb: bool
    hydra: bool
    csv: bool
    tensorboard: bool
    sqlalchemy: bool


@dataclasses.dataclass
class HydraSettings:
    """Subset of the current hydra settings."""

    output_dir: pathlib.Path = dataclasses.field(init=False)
    job_logging: DictConfig = dataclasses.field(init=False)


@dataclass
class UserSettings:
    """User-specific options and preferences.

    Attributes:
        cpu (bool): Whether to run computations on `cpu`, otherwise defaults to local accelerator
        n_workers: The number of workers for data loading
        n_subprocesses: The number of subprocesses for training
        generate (GenerationOptions): Options for generating a point cloud
        plot (PlottingOptions): Options for plotting and visualization
        seed (Optional[int]): The seed for PyTorch/NumPy randomness. If None, no seed is set
        checkpoint_every (PositiveInt): The number of epochs between saving checkpoints
        n_generated_output_points (int): The number of points to generate during inference
        load_checkpoint (int): The checkpoint to load if available. Default is the last one (-1)
        counterfactual_value (PositiveFloat): Value associated with counterfactual change (0 no change, 1 full change)
        hydra (HydraSettings): Subset of the current hydra settings
        path (PathSpecs): Specifications for paths that override .env settings
    """

    cpu: bool
    n_workers: PositiveInt
    n_subprocesses: PositiveInt
    generate: GenerationOptions
    trackers: TrackerList
    plot: PlottingOptions
    seed: int | None
    checkpoint_every: PositiveInt
    n_generated_output_points: int
    load_checkpoint: int = -1
    counterfactual_value: PositiveFloat = 1.0
    hydra = HydraSettings()
    path = PathSpecs()

    def __post_init__(self):
        if self.seed is not None:
            set_seed(self.seed)

        return

    @property
    def device(self) -> torch.device | None:
        """The device where the model should run (None for default)."""
        return torch.device('cpu') if self.cpu else None


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
    classifier: ConfigTrainClassifier
    autoencoder: ConfigTrainAE
    w_autoencoder: ConfigTrainWAE
    user: UserSettings
    data: DataConfig
    tags: list[str] = dataclasses.field(default_factory=list)
    version = f'v{VERSION}'
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
