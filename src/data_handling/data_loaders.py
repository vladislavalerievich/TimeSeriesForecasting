import logging
import os
import random
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.dataset_composer import OnTheFlyDatasetGenerator

logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """
    Dataset class for loading pre-generated synthetic time series data from disk.
    """

    def __init__(
        self,
        data_path: str,
        device: Optional[torch.device] = None,
        single_file: bool = True,
    ):
        """
        Initialize the synthetic dataset.

        Args:
            data_path: Path to the dataset file or directory
            device: Device to load data to
            single_file: If True, expect a single .pt file containing all batches.
                        If False, expect a directory with individual batch files.
        """
        self.data_path = data_path
        self.device = device
        self.single_file = single_file
        self.data = []

        self._load_data()

    def _load_data(self):
        """Load data from disk."""
        if self.single_file:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

            logger.info(f"Loading dataset from {self.data_path}")
            self.data = torch.load(
                self.data_path, map_location=self.device, weights_only=False
            )
            logger.info(f"Loaded {len(self.data)} batches from disk")

        else:
            # Load from directory of individual batch files
            if not os.path.isdir(self.data_path):
                raise FileNotFoundError(
                    f"Dataset directory not found: {self.data_path}"
                )

            batch_files = sorted(
                [f for f in os.listdir(self.data_path) if f.endswith(".pt")]
            )
            logger.info(f"Loading {len(batch_files)} batch files from {self.data_path}")

            for batch_file in batch_files:
                batch_path = os.path.join(self.data_path, batch_file)
                batch = torch.load(batch_path, map_location="cpu")
                self.data.append(batch)

            logger.info(f"Loaded {len(self.data)} batches from disk")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> BatchTimeSeriesContainer:
        """Get a batch by index."""
        batch = self.data[idx]
        if self.device is not None:
            batch.to_device(self.device)
        return batch


class SyntheticTrainDataLoader:
    """
    Data loader for training with pre-generated synthetic data.
    Provides shuffling and iteration control for training.
    """

    def __init__(
        self,
        data_path: str,
        num_batches_per_epoch: int,
        device: Optional[torch.device] = None,
        single_file: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize the training data loader.

        Args:
            data_path: Path to the training dataset
            batch_size: Batch size (note: pre-generated batches already have their size)
            num_batches_per_epoch: Number of batches to use per epoch
            device: Device to load data to
            single_file: If True, expect a single .pt file containing all batches
            shuffle: Whether to shuffle batches between epochs
        """
        self.dataset = SyntheticDataset(data_path, device, single_file)
        self.num_batches_per_epoch = num_batches_per_epoch
        self.shuffle = shuffle
        self.device = device

        if len(self.dataset) < num_batches_per_epoch:
            logger.warning(
                f"Requested {num_batches_per_epoch} batches per epoch, "
                f"but only {len(self.dataset)} batches available. "
                f"Will cycle through available batches."
            )

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate through batches for one epoch."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        # Cycle through indices if we need more batches than available
        batch_count = 0
        while batch_count < self.num_batches_per_epoch:
            for idx in indices:
                if batch_count >= self.num_batches_per_epoch:
                    break
                yield self.dataset[idx]
                batch_count += 1

    def __len__(self) -> int:
        return self.num_batches_per_epoch


class SyntheticValidationDataLoader:
    """
    Data loader for validation with pre-generated synthetic data.
    Iterates through all available validation batches.
    """

    def __init__(
        self,
        data_path: str,
        device: Optional[torch.device] = None,
        single_file: bool = True,
    ):
        """
        Initialize the validation data loader.

        Args:
            data_path: Path to the validation dataset
            batch_size: Batch size (for compatibility, pre-generated batches have fixed size)
            device: Device to load data to
            single_file: If True, expect a single .pt file containing all batches
        """
        self.dataset = SyntheticDataset(data_path, device, single_file)
        self.device = device

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate through all validation batches."""
        for i in range(len(self.dataset)):
            yield self.dataset[i]

    def __len__(self) -> int:
        return len(self.dataset)


class OnTheFlySyntheticDataset(IterableDataset):
    """
    PyTorch IterableDataset for on-the-fly generation of synthetic time series data.
    Used primarily for training data where variation across epochs is beneficial.
    """

    def __init__(
        self, generator: OnTheFlyDatasetGenerator, num_batches: Optional[int] = None
    ):
        """
        Initialize the OnTheFlySyntheticDataset.

        Parameters
        ----------
        generator : OnTheFlyDatasetGenerator
            The generator to use for on-the-fly data generation.
        num_batches : int, optional
            Number of batches to generate per epoch. If None, generate indefinitely. (default: None)
        """
        self.generator = generator
        self.num_batches = num_batches

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """
        Return an iterator over the dataset.

        Returns
        -------
        Iterator[BatchTimeSeriesContainer]
            Iterator over batches.
        """
        if self.num_batches is None:
            # Generate indefinitely
            while True:
                yield self.generator.get_batch()
        else:
            # Generate a fixed number of batches
            for _ in range(self.num_batches):
                yield self.generator.get_batch()
