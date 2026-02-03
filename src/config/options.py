"""Defines all the options in the specifications."""

import enum


class Datasets(enum.StrEnum):
    """Dataset classes."""

    ModelNet = enum.auto()
    ShapenetFlow = enum.auto()


class Encoders(enum.StrEnum):
    """Encoder classes."""

    LDGCNN = enum.auto()
    DGCNN = enum.auto()
    Transformer = enum.auto()


class Decoders(enum.StrEnum):
    """Encoder classes."""

    PCGen = enum.auto()


class WEncoders(enum.StrEnum):
    """Encoder classes for the W-autoencoder."""

    Convolutional = enum.auto()
    Transformer = enum.auto()


class WDecoders(enum.StrEnum):
    """Decoder classes for the W-autoencoder."""

    Linear = enum.auto()
    Transformer = enum.auto()


class WConditionalEncoders(enum.StrEnum):
    """Posterior classes."""

    Transformer = enum.auto()


class AutoEncoders(enum.StrEnum):
    """Autoencoder classes."""

    AE = enum.auto()
    VQVAE = enum.auto()
    CounterfactualVQVAE = enum.auto()


class Diffusion(enum.StrEnum):
    """Diffusion classes."""

    Diffusion = enum.auto()


class Classifiers(enum.StrEnum):
    """Classifier classes."""

    DGCNN = enum.auto()


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


class DiffusionSchedulers(enum.StrEnum):
    """Diffusion scheduler names."""

    Linear = enum.auto()
    Cosine = enum.auto()


class ReconLosses(enum.StrEnum):
    """Loss names."""

    Chamfer = enum.auto()
    ChamferEMD = enum.auto()
