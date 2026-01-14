import enum


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
