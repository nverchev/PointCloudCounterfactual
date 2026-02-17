"""Package containing dataset classes and data types."""

from src.data.dataset import get_dataset, get_datasets
from src.data.modelnet import ModelNet40Dataset
from src.data.protocols import PointCloudDataset, Partitions
from src.data.shapenet import ShapeNetFlowDataset
from src.data.structures import Inputs, Outputs, Targets

IN_CHAN = 3
OUT_CHAN = 3

__all__ = [
    'IN_CHAN',
    'OUT_CHAN',
    'Inputs',
    'ModelNet40Dataset',
    'Outputs',
    'Partitions',
    'PointCloudDataset',
    'ShapeNetFlowDataset',
    'Targets',
    'Targets',
    'get_dataset',
    'get_datasets',
]
