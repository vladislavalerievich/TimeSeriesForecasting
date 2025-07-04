import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency
from src.data_handling.frequency_utils import get_frequency_enum

logger = logging.getLogger(__name__)


class ArrowDatasetReader:
    """
    Efficient reader for Arrow dataset files with random access capabilities.
    """

    def __init__(self, dataset_path: str, cache_metadata: bool = True):
        """
        Initialize the Arrow dataset reader.

        Args:
            dataset_path: Path to the Arrow/Parquet dataset file
            cache_metadata: Whether to cache metadata for faster access
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        self.cache_metadata = cache_metadata
        self._parquet_file = None
        self._num_rows = None
        self._metadata = None

        # Initialize connection
        self._initialize()

    def _initialize(self):
        """Initialize the parquet file connection and metadata."""
        try:
            self._parquet_file = pq.ParquetFile(self.dataset_path)
            self._num_rows = self._parquet_file.metadata.num_rows

            if self.cache_metadata:
                # Read a small sample to understand the schema
                sample_table = self._parquet_file.read_row_group(0, columns=None)
                self._metadata = {
                    "schema": sample_table.schema,
                    "num_rows": self._num_rows,
                    "num_row_groups": self._parquet_file.num_row_groups,
                }

            logger.info(
                f"Initialized Arrow reader for {self.dataset_path} with {self._num_rows} rows"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize Arrow reader for {self.dataset_path}: {e}"
            )
            raise

    @property
    def num_rows(self) -> int:
        """Get the total number of rows in the dataset."""
        return self._num_rows

    def get_random_indices(
        self, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate random row indices for sampling.

        Args:
            n_samples: Number of random indices to generate
            rng: Random number generator

        Returns:
            Array of random row indices
        """
        return rng.integers(0, self._num_rows, size=n_samples)

    def read_rows(self, indices: List[int]) -> pa.Table:
        """
        Read specific rows from the dataset.

        Args:
            indices: List of row indices to read

        Returns:
            Arrow Table with the requested rows
        """
        try:
            # Sort indices for more efficient reading
            sorted_indices = sorted(indices)

            # Read the full table and filter by indices
            # Note: For very large datasets, this could be optimized further
            # by reading only relevant row groups
            table = self._parquet_file.read()
            return table.take(sorted_indices)

        except Exception as e:
            logger.error(
                f"Failed to read rows {indices[:5]}... from {self.dataset_path}: {e}"
            )
            raise

    def read_batch(self, batch_size: int, rng: np.random.Generator) -> pa.Table:
        """
        Read a random batch of rows.

        Args:
            batch_size: Number of rows to read
            rng: Random number generator

        Returns:
            Arrow Table with random rows
        """
        indices = self.get_random_indices(batch_size, rng)
        return self.read_rows(indices.tolist())


class SyntheticArrowDataLoader(IterableDataset):
    """
    Data loader for pre-generated synthetic time series data stored in Arrow format.
    Provides on-the-fly batching with configurable generator proportions and
    random history/future splits.
    """

    def __init__(
        self,
        dataset_paths: Dict[str, str],
        generator_proportions: Dict[str, float],
        batch_size: int = 32,
        future_length_range: Tuple[int, int] = (48, 900),
        device: Optional[torch.device] = None,
        global_seed: int = 42,
    ):
        """
        Initialize the synthetic Arrow data loader.

        Args:
            dataset_paths: Dict mapping generator names to their Arrow dataset paths
            generator_proportions: Dict mapping generator names to their sampling proportions
            batch_size: Number of time series per batch
            future_length_range: Tuple of (min, max) future length to sample
            device: Device to load tensors to
            global_seed: Global random seed for reproducibility
        """
        self.dataset_paths = dataset_paths
        self.generator_proportions = generator_proportions
        self.batch_size = batch_size
        self.future_length_range = future_length_range
        self.device = device
        self.global_seed = global_seed

        # Initialize random number generator
        self.rng = np.random.default_rng(global_seed)

        # Validate inputs
        self._validate_inputs()

        # Initialize dataset readers
        self.dataset_readers = {}
        self._initialize_readers()

        # Prepare sampling weights
        self._prepare_sampling_weights()

        logger.info(
            f"Initialized SyntheticArrowDataLoader with {len(self.dataset_readers)} datasets"
        )

    def _validate_inputs(self):
        """Validate input parameters."""
        # Check that all generators in proportions have corresponding paths
        for gen_name in self.generator_proportions:
            if gen_name not in self.dataset_paths:
                raise ValueError(
                    f"Generator {gen_name} in proportions but not in dataset_paths"
                )

        # Check that proportions sum to 1.0
        total_prop = sum(self.generator_proportions.values())
        if not np.isclose(total_prop, 1.0, atol=1e-6):
            raise ValueError(
                f"Generator proportions should sum to 1.0, got {total_prop}"
            )

        # Check future length range
        if self.future_length_range[0] >= self.future_length_range[1]:
            raise ValueError("future_length_range[0] must be < future_length_range[1]")

        if self.future_length_range[0] < 1:
            raise ValueError("future_length_range[0] must be >= 1")

    def _initialize_readers(self):
        """Initialize Arrow dataset readers for each generator."""
        for gen_name, dataset_path in self.dataset_paths.items():
            if (
                gen_name in self.generator_proportions
                and self.generator_proportions[gen_name] > 0
            ):
                try:
                    self.dataset_readers[gen_name] = ArrowDatasetReader(dataset_path)
                    logger.info(
                        f"Loaded {gen_name} dataset with {self.dataset_readers[gen_name].num_rows} rows"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load dataset for {gen_name}: {e}")
                    # Remove from proportions if dataset can't be loaded
                    self.generator_proportions.pop(gen_name, None)

        if not self.dataset_readers:
            raise ValueError("No valid datasets could be loaded")

        # Renormalize proportions after removing invalid datasets
        total_prop = sum(self.generator_proportions.values())
        if total_prop > 0:
            self.generator_proportions = {
                name: prop / total_prop
                for name, prop in self.generator_proportions.items()
                if name in self.dataset_readers
            }

    def _prepare_sampling_weights(self):
        """Prepare sampling weights for generator selection."""
        self.generator_names = list(self.dataset_readers.keys())
        self.generator_weights = np.array(
            [self.generator_proportions[name] for name in self.generator_names]
        )

        logger.info(
            f"Generator sampling weights: {dict(zip(self.generator_names, self.generator_weights))}"
        )

    def _select_generator(self) -> str:
        """Select a generator based on the configured proportions."""
        return self.rng.choice(self.generator_names, p=self.generator_weights)

    def _sample_future_length(self) -> int:
        """Sample a random future length within the specified range."""
        return self.rng.integers(
            self.future_length_range[0], self.future_length_range[1] + 1
        )

    def _convert_arrow_to_container(
        self, arrow_table: pa.Table, future_length: int
    ) -> BatchTimeSeriesContainer:
        """
        Convert Arrow table to BatchTimeSeriesContainer with history/future split.

        Args:
            arrow_table: Arrow table with time series data
            future_length: Number of points to use as future

        Returns:
            BatchTimeSeriesContainer with split data
        """
        # Convert to pandas for easier manipulation
        df = arrow_table.to_pandas()
        batch_size = len(df)

        if batch_size == 0:
            raise ValueError("Empty batch received")

        # Get the first row to determine dimensions
        first_values = np.array(df.iloc[0]["values"])
        total_length = len(first_values)

        # Calculate history length
        history_length = total_length - future_length

        # Initialize tensors - assuming univariate series for now
        num_channels = 1
        history_values = torch.zeros(
            (batch_size, history_length, num_channels), dtype=torch.float32
        )
        future_values = torch.zeros(
            (batch_size, future_length, num_channels), dtype=torch.float32
        )

        # Fill tensors
        for i, row in df.iterrows():
            values = np.array(row["values"], dtype=np.float32)

            # Split into history and future
            history = values[:history_length]
            future = values[history_length : history_length + future_length]

            # Store in tensors
            history_values[i, :, 0] = torch.from_numpy(history)
            future_values[i, :, 0] = torch.from_numpy(future)

        # Get metadata from first row
        first_row = df.iloc[0]
        start_timestamp = pd.Timestamp(first_row["start"]).to_numpy()

        # Convert frequency string to enum
        frequency_str = first_row["frequency"]
        try:
            frequency = get_frequency_enum(frequency_str)
        except:
            # Fallback to daily if frequency not recognized
            frequency = Frequency.D
            logger.warning(f"Unknown frequency {frequency_str}, using daily")

        # Create container
        container = BatchTimeSeriesContainer(
            history_values=history_values,
            future_values=future_values,
            start=start_timestamp,
            frequency=frequency,
            generator_name=f"arrow_synthetic_{first_row.get('generator_type', 'unknown')}",
        )

        # Move to device if specified
        if self.device is not None:
            container.to_device(self.device)

        return container

    def _generate_mixed_batch(
        self, batch_size: int, future_length: int
    ) -> BatchTimeSeriesContainer:
        """
        Generate a mixed batch by sampling from different generators according to proportions.

        Args:
            batch_size: Total number of series in the batch
            future_length: Future length for all series in the batch

        Returns:
            BatchTimeSeriesContainer with mixed data
        """
        # Determine how many samples to get from each generator
        generator_counts = {}
        remaining_samples = batch_size

        for i, gen_name in enumerate(self.generator_names):
            if i == len(self.generator_names) - 1:
                # Last generator gets all remaining samples
                generator_counts[gen_name] = remaining_samples
            else:
                # Sample proportionally
                count = int(batch_size * self.generator_weights[i])
                generator_counts[gen_name] = count
                remaining_samples -= count

        # Collect data from each generator
        all_tables = []

        for gen_name, count in generator_counts.items():
            if count > 0:
                try:
                    reader = self.dataset_readers[gen_name]
                    table = reader.read_batch(count, self.rng)
                    all_tables.append(table)
                except Exception as e:
                    logger.warning(f"Failed to read from {gen_name}: {e}")
                    continue

        if not all_tables:
            raise RuntimeError("Failed to read data from any generator")

        # Combine all tables
        combined_table = pa.concat_tables(all_tables)

        # Convert to container
        return self._convert_arrow_to_container(combined_table, future_length)

    def __iter__(self) -> Iterator[BatchTimeSeriesContainer]:
        """Iterate through batches indefinitely."""
        return self

    def __next__(self) -> BatchTimeSeriesContainer:
        """Generate the next batch."""
        try:
            # Sample future length for this batch
            future_length = self._sample_future_length()

            # Generate mixed batch
            batch = self._generate_mixed_batch(self.batch_size, future_length)

            return batch

        except Exception as e:
            logger.error(f"Error generating batch: {e}")
            raise


def create_synthetic_arrow_data_loader(
    data_root_dir: str,
    generator_proportions: Dict[str, float],
    batch_size: int = 32,
    future_length_range: Tuple[int, int] = (48, 900),
    device: Optional[torch.device] = None,
    global_seed: int = 42,
) -> SyntheticArrowDataLoader:
    """
    Convenience function to create a SyntheticArrowDataLoader.

    Args:
        data_root_dir: Root directory containing generator subdirectories with dataset.arrow files
        generator_proportions: Dict mapping generator names to their sampling proportions
        batch_size: Number of time series per batch
        future_length_range: Tuple of (min, max) future length to sample
        device: Device to load tensors to
        global_seed: Global random seed for reproducibility

    Returns:
        Configured SyntheticArrowDataLoader
    """
    # Construct dataset paths
    dataset_paths = {}
    data_root = Path(data_root_dir)

    for gen_name in generator_proportions:
        dataset_path = data_root / gen_name / "dataset.arrow"
        if dataset_path.exists():
            dataset_paths[gen_name] = str(dataset_path)
        else:
            logger.warning(f"Dataset not found for {gen_name} at {dataset_path}")

    if not dataset_paths:
        raise ValueError(f"No valid datasets found in {data_root_dir}")

    return SyntheticArrowDataLoader(
        dataset_paths=dataset_paths,
        generator_proportions=generator_proportions,
        batch_size=batch_size,
        future_length_range=future_length_range,
        device=device,
        global_seed=global_seed,
    )
