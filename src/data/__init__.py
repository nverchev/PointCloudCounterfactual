"""Package containing dataset classes and data types."""

from src.data.dataset import get_dataset, get_datasets
from src.data.modelnet import ModelNet40Dataset
from src.data.processed import (
    WDatasetEncoder,
    WDatasetWithLogits,
    ReconstructedDatasetEncoder,
    BoundaryDataset,
    WDatasetWithLogitsFrozen,
    ReconstructedDatasetWithLogits,
    CounterfactualDatasetEncoder,
)
from src.data.protocols import PointCloudDataset, Partitions
from src.data.shapenet import ShapeNetDatasetFlow
from src.data.structures import Inputs, Outputs, Targets, WInputs, WTargets

IN_CHAN = 3
OUT_CHAN = 3

__all__ = [
    'IN_CHAN',
    'OUT_CHAN',
    'BoundaryDataset',
    'CounterfactualDatasetEncoder',
    'Inputs',
    'ModelNet40Dataset',
    'Outputs',
    'Partitions',
    'PointCloudDataset',
    'ReconstructedDatasetEncoder',
    'ReconstructedDatasetWithLogits',
    'ShapeNetDatasetFlow',
    'Targets',
    'WDatasetEncoder',
    'WDatasetWithLogits',
    'WDatasetWithLogitsFrozen',
    'WInputs',
    'WTargets',
    'get_dataset',
    'get_datasets',
]
