"""Defines all enumerated options used in configuration files."""

import enum

__all__ = [
    'ClipCriterion',
    'Datasets',
    'GradOp',
    'Schedulers',
]


# ---------------------------------------------------------------------------
# Project Specific: Add choices for your project
# ---------------------------------------------------------------------------
class Datasets(enum.StrEnum):
    """Dataset choices."""

    MODELNET = enum.auto()


# ---------------------------------------------------------------------------
# Generic options available in drytorch
# ---------------------------------------------------------------------------


class GradOp(enum.StrEnum):
    """Gradient operation names."""

    PARAM_NORMALIZER = enum.auto()
    Z_SCORE_NORMALIZER = enum.auto()
    NORM_CLIPPER = enum.auto()
    VALUE_CLIPPER = enum.auto()
    HIST_CLIPPER = enum.auto()
    PARAM_HIST_CLIPPER = enum.auto()


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names for histogram-based gradient clippers."""

    Z_STAT = enum.auto()
    EMA = enum.auto()


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    CONSTANT = enum.auto()
    COSINE = enum.auto()
    EXPONENTIAL = enum.auto()
