"""Defines all the options in the specifications."""

import enum


class Datasets(enum.StrEnum):
    """Dataset classes."""

    ModelNet = enum.auto()
    ShapeNetFlow = enum.auto()


class Encoders(enum.StrEnum):
    """Encoder classes."""

    PointNet = enum.auto()
    LDGCNN = enum.auto()
    DGCNN = enum.auto()


class Decoders(enum.StrEnum):
    """Encoder classes."""

    PCGen = enum.auto()


class LatentEncoders(enum.StrEnum):
    """Encoder classes for the Latent-autoencoder."""

    Convolutional = enum.auto()
    Linear = enum.auto()


class LatentDecoders(enum.StrEnum):
    """Decoder classes for the Latent-autoencoder."""

    Linear = enum.auto()


class ConditionalLatentEncoders(enum.StrEnum):
    """Posterior classes."""

    Linear = enum.auto()


class AutoEncoders(enum.StrEnum):
    """Autoencoder classes."""

    AE = enum.auto()
    VAE = enum.auto()
    CounterfactualVAE = enum.auto()


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


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names."""

    ZStat = enum.auto()
    EMA = enum.auto()


class Schedulers(enum.StrEnum):
    """Scheduler names."""

    Constant = enum.auto()
    Cosine = enum.auto()
    Exponential = enum.auto()
