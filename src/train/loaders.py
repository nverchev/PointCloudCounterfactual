"""Dataloaders for training and testing."""

from drytorch import DataLoader
from src.data import get_datasets
from src.data.processed import EvaluatedDataset
from src.module import BaseClassifier


def get_loaders(batch_size: int, n_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """Get training and test dataloaders."""
    train_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, n_workers=n_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, n_workers=n_workers)
    return train_loader, test_loader


def get_evaluated_loaders(
    classifier: BaseClassifier, batch_size: int, n_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """Get dataloaders for training and testing with classifier evaluation."""
    train_dataset, test_dataset = get_datasets()  # test is validation unless final=True
    processed_train_dataset = EvaluatedDataset(train_dataset, classifier)
    processed_test_dataset = EvaluatedDataset(test_dataset, classifier)
    train_loader = DataLoader(
        dataset=processed_train_dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        dataset=processed_test_dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        pin_memory=False,
    )
    return train_loader, test_loader
