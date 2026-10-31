"""Selection of the partitions of the active dataset."""

from src.config.experiment import Experiment
from src.config.options import Datasets
from src.data.databuilder import DataBuilder
from src.data.modelnet import ModelNetDataBuilder
from src.data.split import CloudSplit, Partitions

__all__ = [
    'get_databuilder',
    'get_eval_split',
    'get_partitions',
    'get_splits',
]


def get_databuilder() -> DataBuilder:
    """Get the builder of the dataset the configuration selects."""
    name = Experiment.get_config().data.dataset.name
    databuilder_dict: dict[Datasets, type[DataBuilder]] = {Datasets.MODELNET: ModelNetDataBuilder}
    return databuilder_dict[name]()


def get_eval_split() -> CloudSplit:
    """Get the split of the evaluation partition."""
    return get_databuilder().get_split(get_partitions()[1], augment=False)


def get_partitions() -> tuple[Partitions, Partitions]:
    """Get the training and the evaluation partitions of the current experiment.

    Returns:
        ``train`` and ``val``, or ``train_val`` and ``test`` when ``final`` is set.
    """
    final = Experiment.get_config().final
    if final:
        return Partitions.train_val, Partitions.test

    return Partitions.train, Partitions.val


def get_splits() -> tuple[CloudSplit, CloudSplit]:
    """Get the augmented training split and the evaluation split of the current experiment."""
    databuilder = get_databuilder()
    train_partition, eval_partition = get_partitions()
    return databuilder.get_split(train_partition, augment=True), databuilder.get_split(eval_partition, augment=False)
