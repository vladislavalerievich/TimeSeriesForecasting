import logging
import os
import random
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

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

    # SHORT_DATASETS = "us_births/D"
    # MED_LONG_DATASETS = "electricity/15T"
    ALL_DATASETS = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
    TERMS = ["short", "medium", "long"]

    def __init__(
        self,
        mode: str = "train",
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
        to_univariate: bool = False,
        num_batches_per_epoch: Optional[int] = None,
        max_context_length: Optional[int] = None,
        training_windows: int = -1,
        evaluation_windows: int = 20,
        skip_datasets_with_nans: bool = True,
    ):
        """
        Initialize GIFT-eval data loader.

        Args:
            mode: Either "train" or "validation"
            batch_size: Number of samples per batch
            device: Device to load data to
            shuffle: Whether to shuffle data
            to_univariate: Whether to convert multivariate data to multiple univariate series
            num_batches_per_epoch: Optional limit on batches per epoch
            max_context_length: Optional maximum total window length (context + forecast) to prevent memory issues
            training_windows: Number of windows to use for training (-1 = all windows)
            evaluation_windows: Number of windows to use for validation/testing
            skip_datasets_with_nans: Whether to skip datasets/series that contain NaN values
        """
        # Use all available datasets and terms
        self.dataset_names = self.ALL_DATASETS
        self.terms = self.TERMS
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.to_univariate = to_univariate
        self.num_batches_per_epoch = num_batches_per_epoch
        self.max_context_length = max_context_length
        self.skip_datasets_with_nans = skip_datasets_with_nans

        # Window configuration based on mode
        self.training_windows = training_windows
        self.evaluation_windows = evaluation_windows
        if mode == "train":
            self.max_windows = training_windows if training_windows > 0 else None
        else:
            self.max_windows = evaluation_windows

        # Track NaN statistics per dataset to avoid repetitive logging
        self.dataset_nan_stats = {}

        # Load all datasets and collect samples
        self.samples = []
        self._load_datasets()

        logger.info(
            f"Loaded {len(self.samples)} samples from {len(self.dataset_names)} datasets across {len(self.terms)} terms in {mode} mode (max_windows: {self.max_windows})"
        )

        # Validate a few samples to ensure data integrity
        self._validate_samples()

    def _load_datasets(self):
        """Load all specified datasets and collect windowed samples across all terms."""
        logger.info(f"Loading datasets: {self.dataset_names}")
        logger.info(f"Using terms: {self.terms}")
        logger.info(f"Mode: {self.mode}, max_windows: {self.max_windows}")

        total_samples_loaded = 0
        successful_datasets = 0

        for ds_name in self.dataset_names:
            for term in self.terms:
                # Filter datasets based on term (matching evaluate_model.py logic)
                if term in ["medium", "long"]:
                    valid_datasets = self.MED_LONG_DATASETS.split()
                    if ds_name not in valid_datasets:
                        logger.info(
                            f"Skipping dataset {ds_name} for term {term} (not in MED_LONG_DATASETS)"
                        )
                        continue

                try:
                    logger.info(f"Loading dataset {ds_name} with term {term}...")

                    # Special case for M4 datasets - temporarily use 1 evaluation window
                    original_evaluation_windows = self.evaluation_windows
                    original_max_windows = self.max_windows
                    if "m4" in ds_name:  # Special case for M4 datasets
                        self.evaluation_windows = 1
                        logger.info(
                            f"M4 dataset detected: using {self.evaluation_windows} evaluation window(s)"
                        )

                    # Determine if we need to convert to univariate
                    temp_dataset = GiftEvalDataset(
                        name=ds_name, term=term, to_univariate=False
                    )
                    to_univariate = (
                        False if temp_dataset.target_dim == 1 else self.to_univariate
                    )

                    # Load dataset with appropriate univariate setting
                    dataset = GiftEvalDataset(
                        name=ds_name, term=term, to_univariate=to_univariate
                    )

                    # Get appropriate data iterator based on mode
                    if self.mode == "train":
                        data_iter = dataset.training_dataset
                    elif self.mode == "validation":
                        data_iter = dataset.validation_dataset
                    else:
                        raise ValueError(
                            f"Invalid mode: {self.mode}. Must be 'train' or 'validation'"
                        )

                    # Generate windowed samples from each time series
                    dataset_samples = []
                    data_entries_processed = 0

                    for data_entry in data_iter:
                        data_entries_processed += 1
                        windowed_samples = self._generate_windowed_samples(
                            data_entry,
                            dataset.prediction_length,
                            dataset.freq,
                            ds_name,
                            term,
                        )
                        dataset_samples.extend(windowed_samples)

                        # Log progress for large datasets
                        if data_entries_processed % 100 == 0:
                            logger.debug(
                                f"Processed {data_entries_processed} data entries for {ds_name}"
                            )

                    if dataset_samples:
                        self.samples.extend(dataset_samples)
                        total_samples_loaded += len(dataset_samples)
                        successful_datasets += 1

                        logger.info(
                            f"✅ Loaded {len(dataset_samples)} windowed samples from {ds_name} "
                            f"(term: {term}, target_dim: {temp_dataset.target_dim}, to_univariate: {to_univariate}, "
                            f"data_entries: {data_entries_processed})"
                        )
                    else:
                        logger.warning(
                            f"❌ No samples generated for {ds_name} (term: {term})"
                        )

                    # Restore original evaluation_windows settings
                    self.evaluation_windows = original_evaluation_windows
                    self.max_windows = original_max_windows

                except Exception as e:
                    logger.error(
                        f"❌ Failed to load dataset {ds_name} with term {term}: {e}"
                    )
                    # Continue with other datasets rather than failing completely
                    continue

        logger.info("Dataset loading summary:")
        logger.info(
            f"  - Successfully loaded: {successful_datasets} dataset/term combinations"
        )
        logger.info(f"  - Total samples: {total_samples_loaded}")
        logger.info(f"  - Expected batches: {len(self.samples) // self.batch_size}")

        # Log NaN statistics summary
        if self.dataset_nan_stats:
            logger.info("📊 NaN Statistics Summary:")
            for dataset_key, stats in self.dataset_nan_stats.items():
                if stats["total_nans"] > 0:
                    overall_nan_percentage = (
                        (stats["total_nans"] / stats["total_elements"] * 100)
                        if stats["total_elements"] > 0
                        else 0
                    )
                    series_with_nans_percentage = (
                        (stats["series_with_nans"] / stats["total_series"] * 100)
                        if stats["total_series"] > 0
                        else 0
                    )

                    if self.skip_datasets_with_nans:
                        logger.info(
                            f"   {dataset_key}: {stats['series_with_nans']}/{stats['total_series']} series ({series_with_nans_percentage:.1f}%) "
                            f"had NaN values, {stats['series_skipped']} series skipped"
                        )
                    else:
                        logger.info(
                            f"   {dataset_key}: {stats['series_with_nans']}/{stats['total_series']} series ({series_with_nans_percentage:.1f}%) "
                            f"have NaN values, overall {stats['total_nans']:,}/{stats['total_elements']:,} elements ({overall_nan_percentage:.1f}%) are NaN"
                        )

        if not self.samples:
            logger.error("No samples loaded! This will cause training to fail.")
            raise ValueError("No valid samples were loaded from any dataset")

    def _generate_windowed_samples(
        self,
        data_entry: dict,
        prediction_length: int,
        frequency: str,
        dataset_name: str,
        term: str,
    ) -> List[dict]:
        """Generate multiple windowed samples from a single time series."""
        target = data_entry["target"]
        start = data_entry["start"]

        # Convert target to numpy array with validation
        try:
            history = np.asarray(target, dtype=np.float32)
        except (ValueError, TypeError) as e:
            logger.error(
                f"Failed to convert target to numpy array for {dataset_name}: {e}"
            )
            return []

        # Validate for NaN/infinite values early and track statistics
        total_elements = history.size
        nan_count = np.isnan(history).sum() if total_elements > 0 else 0
        inf_count = np.isinf(history).sum() if total_elements > 0 else 0

        # Track NaN statistics per dataset to avoid repetitive logging
        dataset_key = f"{dataset_name}_{term}"
        if dataset_key not in self.dataset_nan_stats:
            self.dataset_nan_stats[dataset_key] = {
                "total_series": 0,
                "series_with_nans": 0,
                "series_skipped": 0,
                "total_elements": 0,
                "total_nans": 0,
                "logged": False,
            }

        stats = self.dataset_nan_stats[dataset_key]
        stats["total_series"] += 1
        stats["total_elements"] += total_elements
        stats["total_nans"] += nan_count

        if nan_count > 0:
            stats["series_with_nans"] += 1

            # Check if we should skip this series due to NaN values
            if self.skip_datasets_with_nans:
                stats["series_skipped"] += 1

                # Only log the first occurrence for each dataset
                if not stats["logged"]:
                    nan_percentage = (
                        (nan_count / total_elements * 100) if total_elements > 0 else 0
                    )
                    logger.info(
                        f"🚫 Dataset {dataset_name} ({term}): Skipping series with NaN values"
                    )
                    logger.info(
                        f"   First series: {nan_count:,}/{total_elements:,} NaN values ({nan_percentage:.1f}%)"
                    )
                    logger.info(
                        f"   Shape: {history.shape}, skip_datasets_with_nans=True"
                    )
                    stats["logged"] = True

                # Return empty list to skip this series
                return []
            else:
                # Only log the first occurrence and summary for each dataset
                if not stats["logged"]:
                    nan_percentage = (
                        (nan_count / total_elements * 100) if total_elements > 0 else 0
                    )
                    logger.info(
                        f"📊 Dataset {dataset_name} ({term}): Found NaN values in time series data"
                    )
                    logger.info(
                        f"   First series: {nan_count:,}/{total_elements:,} NaN values ({nan_percentage:.1f}%)"
                    )
                    logger.info(
                        f"   Shape: {history.shape}, replacing NaN values with zeros"
                    )
                    stats["logged"] = True

                # Replace NaN values with zeros
                history = np.nan_to_num(history, nan=0.0)

        if inf_count > 0:
            if self.skip_datasets_with_nans:
                # Also skip series with infinite values if skip_datasets_with_nans is True
                if not stats.get("inf_logged", False):
                    inf_percentage = (
                        (inf_count / total_elements * 100) if total_elements > 0 else 0
                    )
                    logger.info(
                        f"🚫 Dataset {dataset_name} ({term}): Skipping series with infinite values ({inf_percentage:.1f}%)"
                    )
                    stats["inf_logged"] = True

                return []
            else:
                if not stats.get("inf_logged", False):
                    inf_percentage = (
                        (inf_count / total_elements * 100) if total_elements > 0 else 0
                    )
                    logger.warning(
                        f"Dataset {dataset_name} ({term}): Found {inf_count:,} infinite values ({inf_percentage:.1f}%), replacing with zeros"
                    )
                    stats["inf_logged"] = True

                history = np.nan_to_num(history, posinf=0.0, neginf=0.0)

        # Handle dimensionality
        if history.ndim == 1:
            history = history.reshape(1, -1)  # [1, seq_len] for univariate
        elif history.ndim > 2:
            logger.error(
                f"Unexpected history dimensions {history.shape} for {dataset_name}"
            )
            return []

        num_dims, seq_len = history.shape

        # Calculate minimum context length needed
        if "m4" in dataset_name.lower():
            # M4 datasets often have shorter series, use more flexible minimum context
            min_context_length = max(
                prediction_length, 8
            )  # For M4: at least prediction_length or 8 steps
        else:
            min_context_length = max(
                32, prediction_length
            )  # At least 32 steps or prediction length
        window_size = min_context_length + prediction_length

        # Generate sliding windows
        windowed_samples = []
        step_size = max(1, prediction_length // 2)  # Overlap windows by 50%

        window_count = 0
        for start_idx in range(0, max(1, seq_len - window_size + 1), step_size):
            # Check if we've reached the window limit
            if self.max_windows is not None and window_count >= self.max_windows:
                break

            end_idx = min(start_idx + window_size, seq_len)

            # Skip if window is too small
            if end_idx - start_idx < min_context_length + prediction_length:
                continue

            window_data = history[:, start_idx:end_idx]

            # Apply max context length if specified (max_context_length now represents total window size)
            if self.max_context_length is not None:
                max_window_size = self.max_context_length
                if window_data.shape[1] > max_window_size:
                    # Take the most recent data
                    window_data = window_data[:, -max_window_size:]

            # Validate window data
            if np.isnan(window_data).any() or np.isinf(window_data).any():
                logger.warning(
                    f"Skipping window with NaN/inf values for {dataset_name}"
                )
                continue

            # Create sample with windowed data
            sample = {
                "window_data": window_data,
                "dataset_name": dataset_name,
                "prediction_length": prediction_length,
                "frequency": frequency,
                "start": start,
                "original_start_idx": start_idx,
                "term": term,
            }
            windowed_samples.append(sample)
            window_count += 1

        # If no valid windows generated, create at least one from the end of the series
        min_series_length = prediction_length + 10
        if "m4" in dataset_name.lower():
            min_series_length = prediction_length + 2

        if not windowed_samples and seq_len >= min_series_length:
            # Calculate effective max context length (max_context_length now represents total window size)
            effective_max_context = (
                (self.max_context_length - prediction_length)
                if self.max_context_length
                else None
            )
            context_length = min(
                seq_len - prediction_length, effective_max_context or seq_len
            )
            window_data = history[:, -context_length - prediction_length :]

            # Validate the fallback window
            if not (np.isnan(window_data).any() or np.isinf(window_data).any()):
                sample = {
                    "window_data": window_data,
                    "dataset_name": dataset_name,
                    "prediction_length": prediction_length,
                    "frequency": frequency,
                    "start": start,
                    "original_start_idx": max(
                        0, seq_len - context_length - prediction_length
                    ),
                    "term": term,
                }
                windowed_samples.append(sample)
            elif "m4" in dataset_name.lower() and not self.skip_datasets_with_nans:
                # For M4 datasets, try cleaning the fallback window if it has NaN/inf values
                # Note: window_data was created using the updated effective_max_context logic above
                cleaned_window = np.nan_to_num(
                    window_data, nan=0.0, posinf=0.0, neginf=0.0
                )
                sample = {
                    "window_data": cleaned_window,
                    "dataset_name": dataset_name,
                    "prediction_length": prediction_length,
                    "frequency": frequency,
                    "start": start,
                    "original_start_idx": max(
                        0, seq_len - context_length - prediction_length
                    ),
                    "term": term,
                }
                windowed_samples.append(sample)
                logger.debug(
                    f"Used cleaned fallback window for {dataset_name} due to NaN/inf values"
                )

        if not windowed_samples:
            # Add debug logging for M4 datasets to understand why windows aren't generated
            if "m4" in dataset_name.lower():
                logger.warning(
                    f"No valid windows generated for {dataset_name}: "
                    f"seq_len={seq_len}, window_size={window_size}, "
                    f"min_context_length={min_context_length}, prediction_length={prediction_length}, "
                    f"has_nan={np.isnan(history).any()}, has_inf={np.isinf(history).any()}"
                )
            else:
                logger.warning(f"No valid windows generated for {dataset_name}")

        return windowed_samples

    def _convert_sample_to_container(self, sample: dict) -> BatchTimeSeriesContainer:
        """Convert a windowed GIFT-eval sample to BatchTimeSeriesContainer format."""
        # Extract windowed data
        window_data = sample["window_data"]  # [num_dims, window_length]
        prediction_length = sample["prediction_length"]
        start = sample["start"]
        freq_str = sample["frequency"]
        dataset_name = sample.get("dataset_name", "unknown")

        # Validate input data
        if not isinstance(window_data, np.ndarray):
            window_data = np.asarray(window_data, dtype=np.float32)

        # Ensure float32 type for consistency
        if window_data.dtype != np.float32:
            window_data = window_data.astype(np.float32)

        # Data should be clean by now (either NaN series were skipped or NaNs were replaced with zeros)
        # Add safety check for any remaining NaN/inf values only if not skipping datasets with NaNs
        if not self.skip_datasets_with_nans and (
            np.isnan(window_data).any() or np.isinf(window_data).any()
        ):
            logger.debug(
                f"Cleaning remaining NaN/inf values in window_data for dataset {dataset_name}"
            )
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

        num_dims, window_length = window_data.shape

        # Split the window into context and future parts
        context_length = window_length - prediction_length

        if context_length <= 0:
            # If window is too small, use it all as context and create dummy future
            context_data = window_data
            future_data = np.zeros((num_dims, prediction_length), dtype=np.float32)
        else:
            # Split into context and future
            context_data = window_data[:, :context_length]
            future_data = window_data[
                :, context_length : context_length + prediction_length
            ]

        # Final safety check to ensure data is finite before creating tensors
        # Only apply NaN replacement if not in skip mode
        if not self.skip_datasets_with_nans:
            context_data = np.nan_to_num(context_data, nan=0.0, posinf=0.0, neginf=0.0)
            future_data = np.nan_to_num(future_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to torch tensors with proper shape [batch_size=1, seq_len, num_channels]
        context_values = torch.from_numpy(context_data.T.copy()).unsqueeze(
            0
        )  # [1, context_len, num_dims]
        future_values = torch.from_numpy(future_data.T.copy()).unsqueeze(
            0
        )  # [1, pred_len, num_dims]

        # Final tensor validation (should be clean by now)
        if torch.isnan(context_values).any() or torch.isnan(future_values).any():
            if self.skip_datasets_with_nans:
                # This should not happen in skip mode - log as warning
                logger.warning(
                    f"Unexpected NaN values in tensors for dataset {dataset_name} "
                    f"with skip_datasets_with_nans=True. This indicates a bug."
                )
            else:
                logger.debug(f"Final NaN cleanup in tensors for dataset {dataset_name}")
                context_values = torch.nan_to_num(context_values, nan=0.0)
                future_values = torch.nan_to_num(future_values, nan=0.0)

        # Convert frequency
        try:
            frequency = get_frequency_enum(freq_str)
        except Exception as e:
            logger.warning(
                f"Failed to convert frequency {freq_str} for dataset {dataset_name}: {e}"
            )
            # Default to daily frequency
            from src.data_handling.data_containers import Frequency

            frequency = Frequency.D

        # Convert start timestamp
        try:
            if isinstance(start, pd.Period):
                start_timestamp = start.to_timestamp().to_numpy()
            else:
                start_timestamp = pd.to_datetime(start).to_numpy()
        except Exception as e:
            logger.warning(
                f"Failed to convert start timestamp for dataset {dataset_name}: {e}"
            )
            # Use a default timestamp
            start_timestamp = pd.to_datetime("2020-01-01").to_numpy()

        return BatchTimeSeriesContainer(
            history_values=context_values,
            future_values=future_values,
            start=start_timestamp,
            frequency=frequency,
        )

    def _create_batch(self, samples: List[dict]) -> BatchTimeSeriesContainer:
        """Create a batch from multiple samples."""
        if not samples:
            logger.error("Cannot create batch from empty samples list")
            raise ValueError("Empty samples list")

        # Convert each sample to container with error handling
        containers = []
        for i, sample in enumerate(samples):
            try:
                container = self._convert_sample_to_container(sample)
                containers.append(container)
            except Exception as e:
                logger.error(
                    f"Failed to convert sample {i} from dataset {sample.get('dataset_name', 'unknown')}: {e}"
                )
                # Continue with other samples rather than failing the entire batch
                continue

        if not containers:
            logger.error("No valid containers created from samples")
            raise ValueError("No valid containers created")

        # Get batch dimensions
        batch_size = len(containers)
        max_history_len = max(c.history_length for c in containers)
        max_future_len = max(c.future_length for c in containers)

        # Check if all containers have the same number of channels
        channel_counts = [c.num_channels for c in containers]
        if len(set(channel_counts)) > 1:
            logger.warning(
                f"Mixed channel counts in batch: {channel_counts}. Using max channels."
            )
            num_channels = max(channel_counts)
        else:
            num_channels = containers[0].num_channels

        # Create batch tensors with proper initialization
        batch_history = torch.zeros(
            (batch_size, max_history_len, num_channels), dtype=torch.float32
        )
        batch_future = torch.zeros(
            (batch_size, max_future_len, num_channels), dtype=torch.float32
        )

        # Fill batch tensors (pad shorter sequences with zeros)
        for i, container in enumerate(containers):
            hist_len = container.history_length
            fut_len = container.future_length
            container_channels = container.num_channels

            try:
                # Handle potential channel dimension mismatch by padding or truncating
                if container_channels <= num_channels:
                    # Pad channels if needed
                    batch_history[i, :hist_len, :container_channels] = (
                        container.history_values[0]
                    )
                    batch_future[i, :fut_len, :container_channels] = (
                        container.future_values[0]
                    )
                else:
                    # Truncate channels if needed (shouldn't happen with proper univariate conversion)
                    batch_history[i, :hist_len, :num_channels] = (
                        container.history_values[0, :, :num_channels]
                    )
                    batch_future[i, :fut_len, :num_channels] = container.future_values[
                        0, :, :num_channels
                    ]
            except Exception as e:
                logger.error(f"Error filling batch tensors for container {i}: {e}")
                # Fill with zeros as fallback
                batch_history[i] = 0.0
                batch_future[i] = 0.0

        # Validate final batch tensors
        if torch.isnan(batch_history).any() or torch.isnan(batch_future).any():
            logger.warning(
                "NaN values found in final batch tensors, replacing with zeros"
            )
            batch_history = torch.nan_to_num(batch_history, nan=0.0)
            batch_future = torch.nan_to_num(batch_future, nan=0.0)

        # Use first sample's metadata for batch
        first_container = containers[0]

        return BatchTimeSeriesContainer(
            history_values=batch_history,
            future_values=batch_future,
            start=first_container.start,
            frequency=first_container.frequency,
        )

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate through batches."""
        # Shuffle samples if requested
        samples = self.samples.copy()
        if self.shuffle:
            random.shuffle(samples)

        # Create batches
        batch_count = 0
        for i in range(0, len(samples), self.batch_size):
            if (
                self.num_batches_per_epoch is not None
                and batch_count >= self.num_batches_per_epoch
            ):
                break

            batch_samples = samples[i : i + self.batch_size]
            if len(batch_samples) < self.batch_size:
                # Pad last batch by cycling through samples
                while len(batch_samples) < self.batch_size:
                    batch_samples.extend(
                        samples[: self.batch_size - len(batch_samples)]
                    )

            batch = self._create_batch(batch_samples)

            if self.device is not None:
                batch.to_device(self.device)

            yield batch
            batch_count += 1

    def __len__(self) -> int:
        """Return number of batches."""
        if self.num_batches_per_epoch is not None:
            return self.num_batches_per_epoch
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def _validate_samples(self):
        """Validate a subset of loaded samples to ensure data integrity."""
        if not self.samples:
            return

        # Check the first few samples
        num_to_check = min(5, len(self.samples))
        logger.info(f"Validating {num_to_check} samples...")

        for i in range(num_to_check):
            sample = self.samples[i]
            try:
                # Check required keys
                required_keys = [
                    "window_data",
                    "dataset_name",
                    "prediction_length",
                    "frequency",
                    "start",
                    "term",
                ]
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    logger.error(f"Sample {i} missing keys: {missing_keys}")
                    continue

                # Check window_data
                window_data = sample["window_data"]
                if not isinstance(window_data, np.ndarray):
                    logger.error(
                        f"Sample {i}: window_data is not numpy array (type: {type(window_data)})"
                    )
                    continue

                if window_data.size == 0:
                    logger.error(f"Sample {i}: empty window_data")
                    continue

                if np.isnan(window_data).any():
                    logger.warning(f"Sample {i}: window_data contains NaN values")

                if np.isinf(window_data).any():
                    logger.warning(f"Sample {i}: window_data contains infinite values")

                # Check dimensions
                if window_data.ndim != 2:
                    logger.error(
                        f"Sample {i}: window_data has wrong dimensions {window_data.shape}"
                    )
                    continue

                num_dims, window_length = window_data.shape
                prediction_length = sample["prediction_length"]

                if window_length < prediction_length:
                    logger.error(
                        f"Sample {i}: window_length ({window_length}) < prediction_length ({prediction_length})"
                    )
                    continue

                logger.debug(
                    f"Sample {i} validation passed: shape={window_data.shape}, "
                    f"dataset={sample['dataset_name']}, term={sample['term']}"
                )

            except Exception as e:
                logger.error(f"Error validating sample {i}: {e}")

        logger.info("Sample validation completed.")
