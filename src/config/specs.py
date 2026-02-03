"""Specification for the configuration files."""

import dataclasses
import pathlib

from typing import Any, Annotated, Self

import torch

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig

from src.config.environment import EnvSettings, VERSION
from src.config.torch import get_activation_cls, get_norm_cls, get_optim_cls, set_seed, DEFAULT_ACT, DEFAULT_NORM
from src.config.options import (
    Datasets,
    Encoders,
    Decoders,
    WEncoders,
    WDecoders,
    AutoEncoders,
    GradOp,
    ClipCriterion,
    Schedulers,
    ReconLosses,
    WConditionalEncoders,
    Classifiers,
    Diffusion,
    DiffusionSchedulers,
)

PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


@dataclass
class DatasetConfig:
    """Specification for the dataset.

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
    """Specification for pre-processing the data.

    Attributes:
        dataset (DatasetConfig): The dataset configuration
        n_input_points (StrictlyPositiveInt): The (maximum) number of input points
        n_target_points (StrictlyPositiveInt): The (maximum) number of target points
        translate (bool): Whether to apply random translation to training inputs and targets
        rotate (bool): Whether to apply random rotation to training inputs and reference
        jitter_sigma (PositiveFloat): The variance for random perturbation of training inputs
        jitter_clip (PositiveFloat): The threshold value for random perturbation of training inputs
        resample (bool): Whether to use two different samplings for input and reference
        sample_with_replacement: Whether to sample with replacement when resampling the input points
    """

    dataset: DatasetConfig
    n_input_points: StrictlyPositiveInt
    n_target_points: StrictlyPositiveInt
    translate: bool
    rotate: bool
    jitter_sigma: PositiveFloat
    jitter_clip: PositiveFloat
    resample: bool
    sample_with_replacement: bool


@dataclass(kw_only=True)
class ArchitectureConfig:
    """Base specification for an architecture with optional convolutional and transformer parts.

    Attributes:
        conv_dims (tuple[StrictlyPositiveInt]): Hidden dimensions for the convolutional part
        norm_cls (NormClass): The normalization class to be applied to the convolutional
        n_heads (StrictlyPositiveInt): Number of attention heads for self-attention
        proj_dim (StrictlyPositiveInt): Number of dimensions in expanded embedding for self-attention
        act_name (str): The name of the PyTorch activation class (e.g., 'ReLU', 'LeakyReLU')
        n_transformer_layers: Number of transformer layers
        act_cls (ActClass): The activation class to be applied both to the convolutional and transformer parts
    """

    conv_dims: tuple[StrictlyPositiveInt, ...] = dataclasses.field(default_factory=tuple)
    conv_norm_name: str = ''
    n_heads: StrictlyPositiveInt = 1
    proj_dim: StrictlyPositiveInt = 1
    feedforward_dim: StrictlyPositiveInt = 1024
    n_transformer_layers: PositiveInt = 0
    transformer_dropout: float = 0.1
    act_name: str = ''

    def __post_init__(self) -> None:
        """Resolve activation class from name."""
        self.act_cls = get_activation_cls(self.act_name) if self.act_name else DEFAULT_ACT
        self.norm_cls = get_norm_cls(self.conv_norm_name) if self.conv_norm_name else DEFAULT_NORM
        return


@dataclass
class EncoderConfig(ArchitectureConfig):
    """Specification for the encoder.

    Attributes:
        class_name (Encoders): The name of the encoder class
        n_neighbors (StrictlyPositiveInt): The number of neighbors for edge convolution (counting the point itself)
    """

    class_name: Encoders
    n_neighbors: StrictlyPositiveInt


@dataclass
class DecoderConfig(ArchitectureConfig):
    """Specification for the decoder.

    Attributes:
        class_name (Decoders): The name of the decoder class
        sample_dim (StrictlyPositiveInt): Dimensions of the sampling sphere
        n_components (StrictlyPositiveInt): Number of components for PCGen
        map_dims (tuple[StrictlyPositiveInt]): Hidden dimensions for mapping the initial sampling
        tau (PositiveFloat): Coefficient for Gumbel Softmax activation
        filter (bool): Whether to apply filtering to the output
    """

    class_name: Decoders
    sample_dim: StrictlyPositiveInt
    n_components: StrictlyPositiveInt
    map_dims: tuple[StrictlyPositiveInt, ...]
    tau: PositiveFloat
    filter: bool


@dataclass
class WEncoderConfig(ArchitectureConfig):
    """Specification for the W-Autoencoder's encoder.

    Attributes:
        class_name (WEncoders): The name of the w-encoder class
    """

    class_name: WEncoders


@dataclass
class WDecoderConfig(ArchitectureConfig):
    """Specification for the W-Autoencoder's decoder.

    Attributes:
        class_name (WDecoders): The name of the w-decoder class
    """

    class_name: WDecoders


@dataclass
class WConditionalEncoder(ArchitectureConfig):
    """Specification for the encoding of z2 given the classifier probabilities.

    Attributes:
        class_name (WConditionalEncoders): The name of the encoder class
    """

    class_name: WConditionalEncoders


@dataclass
class AutoEncoderConfig:
    """Specification for the autoencoder.

    Attributes:
        name (str): The name of the autoencoder model
        class_name (AutoEncoders): The name of the autoencoder class
        encoder (EncoderConfig): The encoder configuration
        decoder (DecoderConfig): The decoder configuration
        book_size (StrictlyPositiveInt): The size of the dictionary for VQ-VAE
        embedding_dim (StrictlyPositiveInt): The length of the code embedding
        n_codes (StrictlyPositiveInt): The number of codes
        vq_noise (PositiveFloat): Quantity of noise to add when redistributing the codes
        codebook_momentum (PositiveFloat): Momentum parameter for the codebook update
        w_dim (StrictlyPositiveInt): The codeword length
    """

    name: str
    class_name: AutoEncoders
    encoder: EncoderConfig
    decoder: DecoderConfig
    book_size: StrictlyPositiveInt
    embedding_dim: StrictlyPositiveInt
    n_codes: StrictlyPositiveInt
    vq_noise: PositiveFloat
    codebook_momentum: PositiveFloat

    def __post_init__(self) -> None:
        """Calculate the word dimension."""
        self.w_dim = self.embedding_dim * self.n_codes
        return


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion model.

    Attributes:
        class_name (Diffusion): The type of model head
        decoder (Decoders): The network used for the diffusion process
        beta_start (PositiveFloat): The starting beta value
        beta_end (PositiveFloat): The ending beta value
        n_timesteps (StrictlyPositiveInt): The number of timesteps
        schedule_type (str): The type of schedule (linear, cosine)
        name (str): The name of the model
    """

    class_name: Diffusion
    decoder: DecoderConfig
    beta_start: PositiveFloat
    beta_end: PositiveFloat
    n_timesteps: StrictlyPositiveInt
    schedule_type: DiffusionSchedulers
    name: str


@dataclass
class WAutoEncoderConfig:
    """Specification for the autoencoder.

    Attributes:
        name: name foe the w-autoencoder model
        w_decoder (WDecoderConfig): Configuration for the word decoder
        w_encoder (WEncoderConfig): Configuration for the word encoder
        conditional_w_encoder (WConditionalEncoder): Configuration for the conditional w-encoder
        z1_dim (StrictlyPositiveInt): The continuous latent space dimension
        z2_dim (StrictlyPositiveInt): The continuous latent space dimension for counterfactual manipulation
        n_pseudo_inputs (PositiveInt): The number of pseudo-inputs for the VAMP loss (0 for no pseudo inputs)
        cf_temperature (float): Temperature for the probabilities (closer to zero means closer to samplings)
    """

    name: str
    w_decoder: WDecoderConfig
    w_encoder: WEncoderConfig
    conditional_w_encoder: WConditionalEncoder
    z1_dim: StrictlyPositiveInt
    z2_dim: StrictlyPositiveInt
    cf_temperature: float
    n_pseudo_inputs: PositiveInt


@dataclass
@dataclass
class ClassifierConfig(ArchitectureConfig):
    """Specification for the classifier.

    Attributes:
        name (str): The name of the classifier model
        class_name (Classifiers): The name of the classifier class
        n_neighbors (StrictlyPositiveInt): The number of neighbors for edge convolution (counting the point itself)
        feature_dim (StrictlyPositiveInt): The dimension of the extracted features from the convolutional part
        mlp_dims (tuple[StrictlyPositiveInt]): Hidden dimensions for the MLP part
        dropout_rates (tuple[PositiveFloat]): Dropout rates for the MLP part
    """

    name: str
    class_name: Classifiers
    n_neighbors: StrictlyPositiveInt
    feature_dim: StrictlyPositiveInt
    mlp_dims: tuple[StrictlyPositiveInt, ...] = dataclasses.field(default_factory=tuple)
    dropout_rates: tuple[PositiveFloat, ...] = dataclasses.field(default_factory=tuple)

    @model_validator(mode='after')
    def _check_length_dropout(self) -> Self:
        if len(self.mlp_dims) > len(self.dropout_rates):
            msg = 'Number of hidden dimensions {} and dropouts {} not compatible.'
            raise ValueError(msg.format(len(self.mlp_dims), len(self.dropout_rates)))

        return self


@dataclass
class SchedulerConfig:
    """Specification for the learning rate scheduler.

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
    """Specification for the learning scheme.

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

    def __post_init__(self) -> None:
        """Resolve optimizer class from name."""
        self.optimizer_cls = get_optim_cls(self.optimizer_name)
        return


@dataclass
class EarlyStoppingConfig:
    """Specification for early stopping during training.

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
    """Specification for the training process.

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
    """Specification for the Autoencoder's loss and metrics.

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
    """Specification for the W-Autoencoder's loss and metrics.

    Attributes:
        c_kld1 (PositiveFloat): The Kullback-Leibler Divergence coefficient for the first latent variable
        c_kld2 (PositiveFloat): The Kullback-Leibler Divergence coefficient for the second latent variable
    """

    c_kld1: PositiveFloat
    c_kld2: PositiveFloat


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
        n_inference_output_points (int): The number of output points during inference for each point cloud
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
    n_inference_output_points: int
    load_checkpoint: int = -1
    counterfactual_value: PositiveFloat = 1.0
    hydra = HydraSettings()
    path = PathSpecs()

    def __post_init__(self):
        """Set seed for PyTorch and NumPy."""
        if self.seed is not None:
            set_seed(self.seed)

        return

    @property
    def device(self) -> torch.device | None:
        """The device where the model should run (None for default)."""
        return torch.device('cpu') if self.cpu else None


@dataclass
class ExperimentConfig:
    """Specification for the current experiment.

    Attributes:
        name (str): The name of the experiment part
        train (TrainingConfig): Training options
        model (Any): The model architecture configuration
        objective (Any): The objective configuration
    """

    name: str
    train: TrainingConfig
    model: Any
    objective: Any


@dataclass
class ClassifierExperimentConfig(ExperimentConfig):
    """Specification for the classifier experimental part.

    Attributes:
        name (str): The name of the experiment part
        train (TrainingConfig): Training options
        model (ClassifierConfig): The classifier architecture configuration.
    """

    model: ClassifierConfig
    objective = None


@dataclass
class AutoEncoderExperimentConfig(ExperimentConfig):
    """Specification for the autoencoder experimental part.

    Attributes:
        name (str): The name of the experiment part
        train (TrainingConfig): Training options
        model (AutoEncoderConfig): The autoencoder architecture configuration
        objective (ObjectiveAEConfig): The autoencoder objective (loss and metrics) configuration
        diagnose_every (StrictlyPositiveInt): The number of points between diagnostics (rearranging the discrete space)
        n_training_output_points (StrictlyPositiveInt): The number of output points during training (0: same as input)

    """

    model: AutoEncoderConfig
    objective: ObjectiveAEConfig
    diagnose_every: StrictlyPositiveInt
    n_training_output_points: StrictlyPositiveInt


@dataclass
class DiffusionExperimentConfig(ExperimentConfig):
    """Specification for the autoencoder experimental part.

    Attributes:
        name (str): The name of the experiment part
        train (TrainingConfig): Training options
        model (AutoEncoderConfig): The autoencoder architecture configuration
        objective (ObjectiveAEConfig): The autoencoder objective (loss and metrics) configuration
        diagnose_every (StrictlyPositiveInt): The number of points between diagnostics (rearranging the discrete space)
        n_training_output_points (StrictlyPositiveInt): The number of output points during training (0: same as input)

    """

    model: DiffusionConfig
    objective: ObjectiveAEConfig
    diagnose_every: StrictlyPositiveInt
    n_training_output_points: StrictlyPositiveInt


@dataclass
class WAutoEncoderExperimentConfig(ExperimentConfig):
    """Specification for the w-autoencoder experimental part.

    Attributes:
        name (str): The name of the experiment part
        train (TrainingConfig): Training options
        objective (ObjectiveWAEConfig): The W-autoencoder objective (loss and metrics) configuration
        model (WAutoEncoderConfig): The W-autoencoder architecture configuration
    """

    model: WAutoEncoderConfig
    objective: ObjectiveWAEConfig


@dataclass
class AllConfig:
    """Root specification for all experiment settings.

    Attributes:
        variation (str): The name for the experiment
        final (bool): If True, it uses the validation dataset for training and the test dataset for testing
        classifier (ClassifierExperimentConfig): The configuration for training the classifier
        autoencoder (AutoEncoderExperimentConfig): The configuration for training the autoencoder
        w_autoencoder (WAutoEncoderExperimentConfig): The configuration for training the W-autoencoder
        user (UserSettings): User-specific settings
        data (DataConfig): Data pre-processing configuration
    """

    classifier: ClassifierExperimentConfig
    autoencoder: AutoEncoderExperimentConfig
    diffusion: DiffusionExperimentConfig
    w_autoencoder: WAutoEncoderExperimentConfig
    user: UserSettings
    data: DataConfig
    variation: str
    final: bool
    tags: list[str] = dataclasses.field(default_factory=list)
    version = f'v{VERSION}'

    @property
    def name(self) -> str:
        """The full name of the experiment, indicating if metrics are calculated on the test dataset."""
        out = f'{self.variation}_final' if self.final else self.variation
        return out[:255]

    @property
    def project(self) -> str:
        """Project name for wandb logging."""
        return 'PointCloudCounterfactual' + str(self.version)
