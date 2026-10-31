"""Specification for the configuration files.

Root config hierarchy::

    ExperimentConfig
    ├── classification: ClassificationConfig        ← the classifier under audit
    │   ├── model: ClassifierModelConfig
    │   │   └── architecture: ClassifierArchitectureConfig
    │   └── train: TrainingConfig
    ├── autoencoding: AutoencodingConfig            ← the outer VQ-VAE, not conditioned
    │   ├── model: AutoencoderModelConfig
    │   │   └── architecture: AutoencoderArchitectureConfig
    │   └── train: AutoencoderTrainingConfig
    ├── w_autoencoding: WAutoencodingConfig         ← the inner VAE, conditioned on the classifier
    │   ├── model: WModelConfig
    │   │   └── architecture: WArchitectureConfig
    │   └── train: WTrainingConfig
    ├── evaluation: EvaluationConfig
    ├── data: DataConfig
    └── user: UserSettings
        ├── trackers: TrackerList
        ├── plot: PlottingSettings
        ├── path: PathSpecs          ← populated from .env
        └── hydra: HydraSettings     ← populated at runtime
"""

import dataclasses
import pathlib

from functools import cached_property
from typing import Annotated, Any, Literal, Self

import torch

from pydantic import BeforeValidator, Field, PlainValidator, model_validator
from pydantic.dataclasses import dataclass

from src.config.environment import PROJECT_NAME, VERSION, EnvSettings, get_current_branch
from src.config.options import ClipCriterion, Datasets, GradOp, Schedulers
from src.config.torch import ActClass, get_activation_cls, get_device, get_optim_cls, set_seed

ActivationClass = Annotated[ActClass, BeforeValidator(get_activation_cls)]
Device = Annotated[torch.device | Literal['auto'], PlainValidator(get_device)]
OptimizerClass = Annotated[type[torch.optim.Optimizer], BeforeValidator(get_optim_cls)]
PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]
StrictlyPositiveFloat = Annotated[float, Field(gt=0)]
UnitIntervalFloat = Annotated[float, Field(ge=0, le=1)]


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierArchitectureConfig:
    """Architecture of the DGCNN classifier.

    Attributes:
        n_neighbours: Neighbours of every point in the edge convolutions.
        hidden_dims: Channels of the edge convolutions.
        embedding_dim: Channels of the point features before the global pooling.
        mlp_dims: Hidden dimensions of the classification head.
        dropout: Dropout probability after the first layer of the classification head.
        act_cls: Activation of the layers, written by its name.
        bn_momentum: Momentum of the batch normalization.
    """

    n_neighbours: StrictlyPositiveInt
    hidden_dims: tuple[StrictlyPositiveInt, ...]
    embedding_dim: StrictlyPositiveInt
    mlp_dims: tuple[StrictlyPositiveInt, ...]
    dropout: UnitIntervalFloat
    act_cls: ActivationClass
    bn_momentum: UnitIntervalFloat


@dataclass(frozen=True)
class ClassifierModelConfig:
    """Model of the classification phase.

    Attributes:
        name: Name of the model.
        architecture: Architecture hyperparameters.
    """

    name: str
    architecture: ClassifierArchitectureConfig


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture of the point encoder.

    Attributes:
        n_neighbours: Neighbours of every point in the edge convolutions.
        h_dims: Channels of the edge convolutions.
        act_cls: Activation of the layers, written by its name.
    """

    n_neighbours: StrictlyPositiveInt
    h_dims: tuple[StrictlyPositiveInt, ...]
    act_cls: ActivationClass


@dataclass(frozen=True)
class DecoderConfig:
    """Architecture of the PCGen decoder.

    Attributes:
        sample_dim: Dimension of the sampling of the generating surface.
        map_h_dims: Hidden dimensions of the map of the sampling.
        conv_h_dims: Channels of the point convolutions of each component.
        n_components: Components of the decoder.
        act_cls: Activation of the point convolutions, written by its name.
        attention_temperature: Temperature of the softmax choosing the component of every point.
        filtering: Whether the graph filter sharpens the decoded cloud.
    """

    sample_dim: StrictlyPositiveInt
    map_h_dims: tuple[StrictlyPositiveInt, ...]
    conv_h_dims: tuple[StrictlyPositiveInt, ...]
    n_components: StrictlyPositiveInt
    act_cls: ActivationClass
    attention_temperature: PositiveFloat
    filtering: bool


@dataclass(frozen=True)
class AutoencoderArchitectureConfig:
    """Architecture of the outer VQ-VAE.

    Attributes:
        encoder: Architecture of the point encoder.
        decoder: Architecture of the PCGen decoder.
        bn_momentum: Momentum of the batch normalization.
    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    bn_momentum: UnitIntervalFloat


@dataclass(frozen=True)
class AutoencoderModelConfig:
    """Model of the autoencoding phase.

    Attributes:
        name: Name of the model.
        architecture: Architecture hyperparameters.
        w_dim: Dimension of the codeword.
        embedding_dim: Dimension of each code.
        book_size: Codes in each codebook.
    """

    name: str
    architecture: AutoencoderArchitectureConfig
    w_dim: StrictlyPositiveInt
    embedding_dim: StrictlyPositiveInt
    book_size: StrictlyPositiveInt

    @model_validator(mode='after')
    def _check_chunks(self) -> Self:
        if self.w_dim % self.embedding_dim:
            msg = f'The codeword of dimension {self.w_dim} is not made of chunks of {self.embedding_dim}.'
            raise ValueError(msg)

        return self

    @property
    def dim_codes(self) -> int:
        """Number of chunks of a codeword, one codebook each."""
        return self.w_dim // self.embedding_dim


@dataclass(frozen=True)
class WArchitectureConfig:
    """Architecture of the inner VAE.

    Attributes:
        encoder_h_dims: Expansion of the channels of each code in the encoder.
        decoder_h_dims: Channels per code of the hidden layers of the decoder.
        prior_h_dims: Hidden dimensions of the conditional prior; empty for a linear layer.
        dropout: Dropout probability.
        act_cls: Activation of the layers, written by its name.
        bn_momentum: Momentum of the batch normalization.
    """

    encoder_h_dims: tuple[StrictlyPositiveInt, ...]
    decoder_h_dims: tuple[StrictlyPositiveInt, ...]
    prior_h_dims: tuple[StrictlyPositiveInt, ...]
    dropout: UnitIntervalFloat
    act_cls: ActivationClass
    bn_momentum: UnitIntervalFloat


@dataclass(frozen=True)
class WModelConfig:
    """Model of the codeword phase.

    Attributes:
        name: Name of the model.
        architecture: Architecture hyperparameters.
        z_u_dim: Dimension of the latent variable unrelated to the classification.
        z_y_dim: Dimension of the latent variable the evaluation of the classifier conditions.
        temperature: Temperature dividing the logits of the classifier before the softmax.
    """

    name: str
    architecture: WArchitectureConfig
    z_u_dim: StrictlyPositiveInt
    z_y_dim: StrictlyPositiveInt
    temperature: StrictlyPositiveFloat


# ---------------------------------------------------------------------------
# Data configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset identity and class filter.

    Attributes:
        name: Name of the dataset.
        n_classes: Number of classes of the evaluations.
        selected_classes: Names of the classes to keep, in the order of their labels; empty keeps all.
    """

    name: Datasets
    n_classes: StrictlyPositiveInt
    selected_classes: tuple[str, ...]

    @model_validator(mode='after')
    def _check_n_classes(self) -> Self:
        if self.selected_classes and len(self.selected_classes) != self.n_classes:
            msg = f'{self.n_classes} classes do not match the selection {self.selected_classes}.'
            raise ValueError(msg)

        return self


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentation of the training clouds.

    Attributes:
        jitter_sigma: Standard deviation of the noise added to the coordinates.
        jitter_clip: Largest absolute value of the noise.
    """

    jitter_sigma: PositiveFloat
    jitter_clip: PositiveFloat


@dataclass(frozen=True)
class DataConfig:
    """Full data configuration.

    Attributes:
        dataset: Dataset selection.
        n_points: Points of each cloud.
        augmentation: Augmentation of the training clouds.
    """

    dataset: DatasetConfig
    n_points: StrictlyPositiveInt
    augmentation: AugmentationConfig


# ---------------------------------------------------------------------------
# Training configs — generic, reuse as-is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """Specification for the learning-rate scheduler.

    Attributes:
        function: Which scheduler to use (see :class:`~src.config.options.Schedulers`).
        restart_interval: Epochs between cosine restarts (0 = no restart).
        restart_fraction: Fraction of base LR applied at each restart.
        warmup_steps: Linear warm-up epochs at the start of training.
        settings: Additional keyword arguments forwarded to the scheduler
            constructor (e.g. ``decay_steps``, ``min_decay``).
    """

    function: Schedulers
    restart_interval: PositiveInt
    restart_fraction: PositiveFloat
    warmup_steps: PositiveInt
    settings: dict[str, Any]


@dataclass(frozen=True)
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer: PyTorch optimizer, written by its name, e.g. ``AdamW``.
        lr: Global learning rate (or per-parameter group base LR).
        scheduler: Learning-rate scheduler configuration.
        grad_op: Optional gradient pre-processing operation applied before
            each optimiser step (see :class:`~src.config.options.GradOp`).
        clip_criterion: Criterion for histogram-based clippers.
        opt_settings: Extra keyword arguments forwarded to the optimiser
            constructor (e.g. ``weight_decay``).
    """

    optimizer: OptimizerClass
    lr: PositiveFloat
    scheduler: SchedulerConfig
    grad_op: GradOp | None
    clip_criterion: ClipCriterion | None
    opt_settings: dict[str, Any]

    @model_validator(mode='after')
    def _check_clip_criterion(self) -> Self:
        if self.clip_criterion is None and self.grad_op in (GradOp.HIST_CLIPPER, GradOp.PARAM_HIST_CLIPPER):
            msg = 'Must specify clip_criterion when using HistClipper or ParamHistClipper.'
            raise ValueError(msg)

        return self


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Specification for early stopping.

    Attributes:
        active: Whether to enable early stopping.
        window: Number of recent validation metrics to average.
        patience: Epochs to wait for improvement before stopping.
    """

    active: bool
    window: int
    patience: int


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Specification for the training loop.

    Attributes:
        batch_size: Batch size.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig


@dataclass(frozen=True)
class AutoencoderLossConfig:
    """Coefficients of the loss of the outer VQ-VAE.

    The distances in the loss are averaged over the points.

    Attributes:
        c_chamfer: Coefficient of the Chamfer distance.
        c_emd: Coefficient of the earth mover distance.
        c_embedding: Coefficient of the distance between the codewords and their codes.
    """

    c_chamfer: PositiveFloat
    c_emd: PositiveFloat
    c_embedding: PositiveFloat


@dataclass(frozen=True, kw_only=True)
class AutoencoderTrainingConfig(TrainingConfig):
    """Specification for the training of the outer VQ-VAE.

    Attributes:
        loss: Coefficients of the loss.
        revive_every: Epochs between two revivals of the unused codes.
        revive_noise: Scale of the noise added to the code an unused one moves onto.
    """

    loss: AutoencoderLossConfig
    revive_every: StrictlyPositiveInt
    revive_noise: PositiveFloat


@dataclass(frozen=True)
class WLossConfig:
    """Coefficients of the loss of the inner VAE.

    Attributes:
        c_kld: Coefficient of the Kullback-Leibler divergence.
    """

    c_kld: PositiveFloat


@dataclass(frozen=True, kw_only=True)
class WTrainingConfig(TrainingConfig):
    """Specification for the training of the inner VAE.

    Attributes:
        loss: Coefficients of the loss.
    """

    loss: WLossConfig


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseConfig:
    """Specification for a single phase.

    Attributes:
        model: Model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    model: ClassifierModelConfig | AutoencoderModelConfig | WModelConfig
    train: TrainingConfig


@dataclass(frozen=True)
class ClassificationConfig(PhaseConfig):
    """Specification for the phase training the classifier under audit.

    Attributes:
        model: Classifier model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    model: ClassifierModelConfig


@dataclass(frozen=True)
class AutoencodingConfig(PhaseConfig):
    """Specification for the phase training the outer VQ-VAE.

    Attributes:
        model: Autoencoder model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    model: AutoencoderModelConfig
    train: AutoencoderTrainingConfig


@dataclass(frozen=True)
class WAutoencodingConfig(PhaseConfig):
    """Specification for the phase training the inner VAE.

    Attributes:
        model: Inner VAE model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    model: WModelConfig
    train: WTrainingConfig


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation of the counterfactuals.

    Attributes:
        strengths: Counterfactual strengths, interpolating from the evaluation to the target class.
        batch_size: Batch size.
    """

    strengths: tuple[UnitIntervalFloat, ...]
    batch_size: StrictlyPositiveInt


# ---------------------------------------------------------------------------
# User / environment settings — generic, reuse as-is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        sqlalchemy: Persists metrics in a SQLite database (requires the ``track_all`` extra).
        plotly: Saves an HTML figure of each metric, reading the database (requires the ``track_all`` extra).
    """

    hydra: bool
    sqlalchemy: bool
    plotly: bool


@dataclasses.dataclass(frozen=True)
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
        config_name: Name of the file in ``configs/experiment`` the configuration starts from.
    """

    output_dir: pathlib.Path = pathlib.Path('outputs')
    job_logging: Any = None
    config_name: str = ''


@dataclass(frozen=True)
class PathSpecs:
    """Filesystem path specifications populated from environment variables.

    Attributes:
        data_dir: Root directory holding the datasets, shared by all versions.
        root_exp_dir: Root directory under which all experiment artefacts are stored.

    Note:
        Paths are loaded from the ``.env`` file via :class:`~src.config.environment.EnvSettings`.
    """

    _env = EnvSettings()
    data_dir: pathlib.Path = _env.data_dir
    root_exp_dir: pathlib.Path = _env.root_exp_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """Version-scoped experiment directory: ``<root_exp_dir>/v<VERSION>``."""
        return self.root_exp_dir / f'v{VERSION}'


@dataclass(frozen=True)
class PlottingSettings:
    """Options for visual inspection.

    Attributes:
        show: Also display the figures (needs a display).
        sample_indices: Clouds of the evaluation partition to render; empty renders the misclassified ones.
    """

    show: bool
    sample_indices: tuple[PositiveInt, ...]


@dataclass(frozen=True)
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        device: Device where the models run: ``auto`` for the accelerator when available, or a name such as
            ``cpu`` or ``cuda:1``.
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        trackers: Active experiment-tracking backends.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        checkpoint_every: Save a model checkpoint every *n* epochs; ``0`` saves only at the end of training.
        load: Checkpoint epoch to resume from:``0`` skip; ``-1`` loads last checkpoint.
        run_id: Run to resume. Defaults to the last run of the experiment.
        plot: Options for visual inspection.
    """

    device: Device
    n_workers: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    load: int
    run_id: str | None
    plot: PlottingSettings
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return


# ---------------------------------------------------------------------------
# Root config — main entry point for the configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        classification: Classification phase configuration.
        autoencoding: Autoencoding phase configuration.
        w_autoencoding: Codeword phase configuration.
        evaluation: Evaluation of the counterfactuals.
        user: User-specific options and runtime preferences.
        data: Data pre-processing and dataset configuration.
        final: If True, uses validation data for training and test data for evaluation.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    classification: ClassificationConfig
    autoencoding: AutoencodingConfig
    w_autoencoding: WAutoencodingConfig
    evaluation: EvaluationConfig
    user: UserSettings
    data: DataConfig
    final: bool
    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in the configuration file."
            raise ValueError(msg) from ve

    @property
    def resolved_variation(self) -> str:
        """variation if set, otherwise the current git branch."""
        return self.variation or self.git_branch

    @property
    def name(self) -> str:
        """Full experiment name embedded into tracker runs."""
        suffix = '_final' if self.final else ''
        return self.resolved_variation[:249] + suffix

    @property
    def par_dir(self) -> pathlib.Path:
        """Directory of the experiments of this configuration: ``<version_dir>/<config_name>``."""
        return self.user.path.version_dir / self.user.hydra.config_name

    @property
    def project(self) -> str:
        """Tracker project name, one per version."""
        return f'{PROJECT_NAME}-{self.version}'
