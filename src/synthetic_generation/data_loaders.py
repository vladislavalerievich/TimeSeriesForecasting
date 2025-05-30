import glob
import os
from typing import Callable, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.serialization import add_safe_globals

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.dataset_composer import OnTheFlyDatasetGenerator

# Register BatchTimeSeriesContainer as a safe global for torch.load
add_safe_globals([BatchTimeSeriesContainer])


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for pre-generated synthetic time series data.
    Used primarily for validation data where consistency across epochs is desired.
    """

    def __init__(self, data_path: str, single_file: bool = False):
        """
        Initialize the SyntheticDataset.

        Parameters
        ----------
        data_path : str
            Path to the directory containing batch files or to a single combined file.
        single_file : bool, optional
            If True, load from a single combined file. If False, load from individual batch files.
            (default: False)
        """
        self.data_path = data_path
        self.single_file = single_file

        if self.single_file:
            self.batches = torch.load(data_path)
        else:
            # Find all batch files
            batch_files = sorted(glob.glob(os.path.join(data_path, "batch_*.pt")))
            self.batch_files = batch_files

    def __len__(self) -> int:
        """
        Return the number of batches in the dataset.

        Returns
        -------
        int
            Number of batches.
        """
        if self.single_file:
            return len(self.batches)
        else:
            return len(self.batch_files)

    def __getitem__(self, idx: int) -> BatchTimeSeriesContainer:
        """
        Get the specified batch.

        Parameters
        ----------
        idx : int
            Index of the batch to get.

        Returns
        -------
        BatchTimeSeriesContainer
            The requested batch.
        """
        if self.single_file:
            return self.batches[idx]
        else:
            return torch.load(self.batch_files[idx])


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


class SyntheticTrainDataLoader(DataLoader):
    """
    DataLoader for synthetic training data with on-the-fly generation.
    """

    def __init__(
        self,
        generator: OnTheFlyDatasetGenerator,
        num_batches_per_epoch: Optional[int] = None,
        device: Optional[torch.device] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize the SyntheticTrainDataLoader.

        Parameters
        ----------
        generator : OnTheFlyDatasetGenerator
            The generator to use for on-the-fly data generation.
        num_batches_per_epoch : int, optional
            Number of batches to generate per epoch. If None, generate indefinitely. (default: None)
        device : torch.device, optional
            The device to move batches to (e.g., 'cuda'). If None, don't move. (default: None)
        collate_fn : Callable, optional
            Function to collate individual batches. Not typically needed as batches are already collated. (default: None)
        **kwargs
            Additional arguments to pass to the DataLoader.
        """
        dataset = OnTheFlySyntheticDataset(generator, num_batches_per_epoch)

        # Since batches are pre-collated, we use batch_size=1 and provide identity collate_fn
        kwargs["batch_size"] = 1

        if collate_fn is None:
            kwargs["collate_fn"] = self._default_collate_fn
        else:
            kwargs["collate_fn"] = collate_fn

        super().__init__(dataset, **kwargs)
        self.device = device

    def _default_collate_fn(
        self, batch_list: List[BatchTimeSeriesContainer]
    ) -> BatchTimeSeriesContainer:
        """
        Default collate function that just returns the first (and only) item from the list.
        Since our batches are already pre-collated by the generator, we just need to extract them.

        Parameters
        ----------
        batch_list : List[BatchTimeSeriesContainer]
            List containing a single batch.

        Returns
        -------
        BatchTimeSeriesContainer
            The extracted batch.
        """
        batch = batch_list[0]
        if self.device is not None:
            batch.to_device(self.device)
        return batch


class SyntheticValidationDataLoader(DataLoader):
    """
    DataLoader for pre-generated synthetic validation data.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        single_file: bool = False,
        device: Optional[torch.device] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize the SyntheticValidationDataLoader.

        Parameters
        ----------
        data_path : str
            Path to the directory containing batch files or to a single combined file.
        batch_size : int, optional
            Number of batches to load at once. Typically 1 as batches are already collated. (default: 1)
        single_file : bool, optional
            If True, load from a single combined file. If False, load from individual batch files. (default: False)
        device : torch.device, optional
            The device to move batches to (e.g., 'cuda'). If None, don't move. (default: None)
        collate_fn : Callable, optional
            Function to collate individual batches. Not typically needed as batches are already collated. (default: None)
        **kwargs
            Additional arguments to pass to the DataLoader.
        """
        dataset = SyntheticDataset(data_path, single_file)

        if collate_fn is None:
            kwargs["collate_fn"] = self._default_collate_fn
        else:
            kwargs["collate_fn"] = collate_fn

        super().__init__(dataset, batch_size=batch_size, **kwargs)
        self.device = device

    def _default_collate_fn(
        self, batch_list: List[BatchTimeSeriesContainer]
    ) -> BatchTimeSeriesContainer:
        """
        Default collate function that just returns the first item from the list.
        Since our batches are already pre-collated, we typically use batch_size=1.

        Parameters
        ----------
        batch_list : List[BatchTimeSeriesContainer]
            List containing batch(es).

        Returns
        -------
        BatchTimeSeriesContainer
            The extracted batch.
        """
        batch = batch_list[0]
        if self.device is not None:
            batch.to_device(self.device)
        return batch
