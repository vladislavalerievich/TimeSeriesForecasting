import logging
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from src.data_handling.arrow_data_loaders import create_synthetic_arrow_data_loader
from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.frequency_utils import get_frequency_enum
from src.gift_eval.data import Dataset as GiftEvalDataset
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


class GiftEvalDataLoader:
    """
    Data loader for GIFT-eval datasets, converting them to BatchTimeSeriesContainer format.
    Supports both training and validation modes.
    """

    SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

    MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

    ALL_DATASETS = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
    TERMS = ["short", "medium", "long"]

    def __init__(
        self,
        mode: str = "train",
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
        to_univariate: bool = False,
        max_context_length: Optional[int] = None,
        max_windows: int = 20,
        skip_datasets_with_nans: bool = True,
        datasets_to_use: Optional[List[str]] = None,
    ):
        """
        Initialize GIFT-eval data loader.

        Args:
            mode: Either "train" or "validation"
            batch_size: Number of samples per batch
            device: Device to load data to
            shuffle: Whether to shuffle data
            to_univariate: Whether to convert multivariate data to multiple univariate series
            max_context_length: Optional maximum total window length (context + forecast) to prevent memory issues
            max_windows: Number of windows to use for training/validation
            skip_datasets_with_nans: Whether to skip datasets/series that contain NaN values
            datasets_to_use: Optional list of dataset names to use. If None, uses all available datasets
        """
        # Use specified datasets or all available datasets if none specified
        if datasets_to_use is not None and len(datasets_to_use) > 0:
            # Validate that requested datasets are available
            invalid_datasets = [
                ds for ds in datasets_to_use if ds not in self.ALL_DATASETS
            ]
            if invalid_datasets:
                logger.warning(f"Invalid datasets requested: {invalid_datasets}")
                logger.warning(f"Available datasets: {self.ALL_DATASETS}")
                # Use only valid datasets
                self.dataset_names = [
                    ds for ds in datasets_to_use if ds in self.ALL_DATASETS
                ]
            else:
                self.dataset_names = datasets_to_use
        else:
            self.dataset_names = self.ALL_DATASETS

        # Log dataset selection
        if datasets_to_use is not None and len(datasets_to_use) > 0:
            logger.info(
                f"Using subset of datasets: {len(self.dataset_names)}/{len(self.ALL_DATASETS)} datasets"
            )
            logger.info(f"Selected datasets: {self.dataset_names}")
        else:
            logger.info(
                f"Using all available datasets: {len(self.dataset_names)} datasets"
            )

        self.terms = self.TERMS
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.to_univariate = to_univariate
        self.max_context_length = max_context_length
        self.skip_datasets_with_nans = skip_datasets_with_nans

        # Window configuration based on mode
        self.max_windows = max_windows

        # Load all datasets and prepare data
        self._load_datasets()

        # Create iterator state
        self._current_idx = 0
        self._epoch_data = []
        self._prepare_epoch_data()

    def _load_datasets(self) -> None:
        """Load all specified GIFT-eval datasets."""
        self.datasets = {}
        self.dataset_prediction_lengths = {}

        for dataset_name in self.dataset_names:
            if dataset_name.lower() == "m4":
                max_windows = 1
            else:
                max_windows = self.max_windows
            try:
                # Determine if we need univariate conversion
                # First check with multivariate to see target dimension
                temp_dataset = GiftEvalDataset(
                    name=dataset_name,
                    term=self.terms[0],  # Use first term to check dimensionality
                    to_univariate=False,
                    max_windows=max_windows,
                )

                # Convert to univariate if needed
                to_univariate = self.to_univariate and temp_dataset.target_dim > 1

                # Load datasets for all terms
                for term in self.terms:
                    dataset_key = f"{dataset_name}_{term}"
                    dataset = GiftEvalDataset(
                        name=dataset_name,
                        term=term,
                        to_univariate=to_univariate,
                        max_windows=max_windows,
                    )

                    self.datasets[dataset_key] = dataset
                    self.dataset_prediction_lengths[dataset_key] = (
                        dataset.prediction_length
                    )

                    logger.info(
                        f"Loaded {dataset_key} - prediction_length: {dataset.prediction_length}, "
                        f"frequency: {dataset.freq}, target_dim: {dataset.target_dim}, "
                        f"min_length: {dataset._min_series_length}, windows: {dataset.windows}"
                    )

            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_name}: {str(e)}")
                continue

    def _contains_nan(self, data_entry: dict) -> bool:
        """Check if a data entry contains NaN values."""
        target = data_entry.get("target")
        if target is None:
            return False

        # Convert to numpy array if needed
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)

        return np.isnan(target).any()

    def _convert_to_container(
        self, data_entries: List[dict], prediction_length: int, dataset_freq: str
    ) -> BatchTimeSeriesContainer:
        """Convert a batch of data entries to BatchTimeSeriesContainer format with fixed future length."""
        batch_size = len(data_entries)
        max_history_len = 0

        # First pass: determine max history length after truncation
        for entry in data_entries:
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, -1)

            _, seq_len = target.shape

            # Only consider up to the last (max_context_length) values
            effective_max_context = (
                self.max_context_length
                if self.max_context_length is not None
                else seq_len
            )
            if seq_len > effective_max_context:
                seq_len = effective_max_context

            # History is up to (max_context_length - prediction_length)
            history_len = max(
                0, min(seq_len, effective_max_context) - prediction_length
            )
            max_history_len = max(max_history_len, history_len)

        # Get number of channels from first entry
        first_target = np.asarray(data_entries[0]["target"], dtype=np.float32)
        if first_target.ndim == 1:
            first_target = first_target.reshape(1, -1)
        num_channels = first_target.shape[0]

        # Initialize tensors
        history_values = torch.zeros(
            (batch_size, max_history_len, num_channels), dtype=torch.float32
        )
        future_values = torch.zeros(
            (batch_size, prediction_length, num_channels), dtype=torch.float32
        )
        history_mask = torch.ones((batch_size, max_history_len), dtype=torch.bool)

        # Second pass: fill tensors
        for i, entry in enumerate(data_entries):
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, -1)

            num_dims, seq_len = target.shape

            # Only consider up to the last (max_context_length) values
            effective_max_context = (
                self.max_context_length
                if self.max_context_length is not None
                else seq_len
            )
            if seq_len > effective_max_context:
                target = target[:, -effective_max_context:]
                seq_len = effective_max_context

            # Always extract the last prediction_length as future
            if seq_len < prediction_length + 1:
                # Should not happen due to filtering, but just in case
                raise ValueError(
                    f"Target too short for fixed future: {seq_len} < {prediction_length + 1}"
                )

            # History is everything before the last prediction_length
            history_len = seq_len - prediction_length
            history = target[:, :history_len].T  # [history_len, num_channels]
            future = target[
                :, history_len : history_len + prediction_length
            ].T  # [prediction_length, num_channels]

            # Place into batch tensors (right-aligned)
            history_values[i, -history_len:, :] = torch.from_numpy(history)
            future_values[i, :, :] = torch.from_numpy(future)
            history_mask[i, :-history_len] = False

        # Get start timestamp and frequency
        start_timestamp = data_entries[0]["start"]
        if hasattr(start_timestamp, "to_timestamp"):
            start_numpy = start_timestamp.to_timestamp().to_numpy()
        else:
            start_numpy = pd.Timestamp(start_timestamp).to_numpy()

        # Get frequency enum
        frequency = get_frequency_enum(dataset_freq)

        # Create the container
        return BatchTimeSeriesContainer(
            history_values=history_values,
            future_values=future_values,
            start=start_numpy,
            frequency=frequency,
            history_mask=history_mask if self.mode == "train" else None,
            generator_name=f"gift_eval_{self.mode}",
        )

    def _prepare_epoch_data(self) -> None:
        """Prepare all batches for one epoch."""
        self._epoch_data = []

        for dataset_key, dataset in self.datasets.items():
            try:
                # Get appropriate dataset based on mode
                if self.mode == "train":
                    data = dataset.training_dataset
                else:
                    data = dataset.validation_dataset

                # Collect all valid data entries
                valid_entries = []
                dataset_freq = dataset.freq
                prediction_length = self.dataset_prediction_lengths[dataset_key]

                for entry in data:
                    # Skip if contains NaN and configured to do so
                    if self.skip_datasets_with_nans and self._contains_nan(entry):
                        continue

                    # Check if we have enough data
                    target = np.asarray(entry["target"])
                    if target.ndim == 1:
                        seq_len = len(target)
                    else:
                        seq_len = target.shape[1]

                    # Need at least prediction_length + 1 for training
                    if self.mode == "train" and seq_len < prediction_length + 1:
                        continue

                    valid_entries.append(entry)

                if not valid_entries:
                    logger.warning(f"No valid entries found for {dataset_key}")
                    continue

                # Create batches
                for i in range(0, len(valid_entries), self.batch_size):
                    batch_entries = valid_entries[i : i + self.batch_size]
                    try:
                        batch_container = self._convert_to_container(
                            batch_entries, prediction_length, dataset_freq
                        )
                        self._epoch_data.append((dataset_key, batch_container))
                    except Exception as e:
                        logger.warning(
                            f"Failed to create batch for {dataset_key}: {str(e)}"
                        )
                        continue

            except Exception as e:
                logger.warning(
                    f"Failed to process dataset {dataset_key}: {str(e)}. "
                    f"Dataset may be too short for the required offset."
                )
                continue

        # Shuffle if in training mode
        if self.mode == "train" and self.shuffle:
            random.shuffle(self._epoch_data)

        logger.info(f"Prepared {len(self._epoch_data)} batches for {self.mode} mode")

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate through batches for one epoch."""
        # Reset index at the start of each epoch
        self._current_idx = 0

        # Reshuffle data for each new epoch if in training mode
        if self.mode == "train" and self.shuffle:
            random.shuffle(self._epoch_data)

        return self

    def __next__(self) -> BatchTimeSeriesContainer:
        """Get next batch."""
        if not self._epoch_data:
            raise StopIteration("No valid data available")

        # Check if we've exhausted the epoch
        if self._current_idx >= len(self._epoch_data):
            raise StopIteration

        # Get current batch
        dataset_key, batch = self._epoch_data[self._current_idx]
        self._current_idx += 1

        # Move to device if specified
        if self.device is not None:
            batch.to_device(self.device)

        return batch

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self._epoch_data)


class CyclicGiftEvalDataLoader:
    """
    Wrapper for GiftEvalDataLoader that provides cycling behavior for training.
    This allows training for a fixed number of iterations per epoch, cycling through
    the available data as needed.
    """

    def __init__(self, base_loader: GiftEvalDataLoader, num_iterations_per_epoch: int):
        """
        Initialize the cyclic data loader.

        Args:
            base_loader: The underlying GiftEvalDataLoader
            num_iterations_per_epoch: Number of iterations to run per epoch
        """
        self.base_loader = base_loader
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.dataset_names = base_loader.dataset_names
        self.device = base_loader.device

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate for exactly num_iterations_per_epoch iterations."""
        self._current_iteration = 0
        self._base_iter = iter(self.base_loader)
        return self

    def __next__(self) -> BatchTimeSeriesContainer:
        """Get next batch, cycling through base loader as needed."""
        if self._current_iteration >= self.num_iterations_per_epoch:
            raise StopIteration

        try:
            batch = next(self._base_iter)
        except StopIteration:
            # Restart the base iterator when exhausted
            self._base_iter = iter(self.base_loader)
            batch = next(self._base_iter)

        self._current_iteration += 1
        return batch

    def __len__(self) -> int:
        """Return the configured number of iterations per epoch."""
        return self.num_iterations_per_epoch


class SyntheticArrowTrainDataLoader:
    """
    Training data loader for pre-generated synthetic data stored in Arrow format.
    Provides infinite batch generation with buffering for training.
    """

    def __init__(
        self,
        data_root_dir: str,
        generator_proportions: Dict[str, float],
        batch_size: int = 256,
        future_length_range: Tuple[int, int] = (48, 900),
        device: Optional[torch.device] = None,
        global_seed: int = 42,
        buffer_size: int = 3,
    ):
        """
        Initialize the Arrow-based synthetic training data loader.

        Args:
            data_root_dir: Root directory containing generator subdirectories with dataset.arrow files
            generator_proportions: Dict mapping generator names to their sampling proportions
            batch_size: Number of time series per batch
            future_length_range: Tuple of (min, max) future length to sample
            device: Device to load tensors to
            global_seed: Global random seed for reproducibility
            buffer_size: Number of batches to pre-generate and keep in buffer
        """
        self.buffer_size = buffer_size
        self.batch_counter = 0

        # Create the underlying Arrow data loader
        self.arrow_loader = create_synthetic_arrow_data_loader(
            data_root_dir=data_root_dir,
            generator_proportions=generator_proportions,
            batch_size=batch_size,
            future_length_range=future_length_range,
            device=device,
            global_seed=global_seed,
        )

        # Initialize batch buffer
        self.buffer = []
        self._arrow_iter = iter(self.arrow_loader)

        # Fill the buffer initially
        self._fill_buffer()

        logger.info(
            f"Created SyntheticArrowTrainDataLoader with buffer size {buffer_size} (infinite batches)"
        )

    def _fill_buffer(self) -> None:
        """Fill the batch buffer with pre-generated batches."""
        while len(self.buffer) < self.buffer_size:
            try:
                batch = next(self._arrow_iter)
                self.buffer.append(batch)
            except StopIteration:
                # This shouldn't happen with the Arrow loader as it generates indefinitely
                logger.warning(
                    "Arrow loader stopped iteration unexpectedly during buffer fill"
                )
                break

    def get_batch(self) -> BatchTimeSeriesContainer:
        """
        Get a batch from the buffer and refill as needed.

        Returns:
            BatchTimeSeriesContainer: A batch of time series data.
        """
        if not self.buffer:
            # Buffer is empty, try to fill it
            self._fill_buffer()
            if not self.buffer:
                raise RuntimeError("Unable to generate batches - buffer remains empty")

        # Get a batch from the buffer
        batch = self.buffer.pop(0)
        self.batch_counter += 1

        # Generate a new batch to refill the buffer
        try:
            new_batch = next(self._arrow_iter)
            self.buffer.append(new_batch)
        except StopIteration:
            logger.warning("Arrow loader stopped iteration unexpectedly during refill")

        return batch

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Return an iterator that generates batches infinitely."""
        return self

    def __next__(self) -> BatchTimeSeriesContainer:
        """Get the next batch."""
        return self.get_batch()
