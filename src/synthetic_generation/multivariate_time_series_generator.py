import gc
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.time_features import compute_time_features
from src.synthetic_generation.lmc_synth import LMCSynthGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultivariateTimeSeriesGenerator:
    """
    Generate batches of synthetic multivariate time series data using LMCSynthGenerator.
    Provides functionality to create and save datasets compatible with PyTorch DataLoader.
    """

    def __init__(
        self,
        global_seed: int = 42,
        distribution_type: Literal["uniform", "log_uniform"] = "uniform",
        history_length: Union[int, Tuple[int, int]] = (64, 256),
        target_length: Union[int, Tuple[int, int]] = (32, 256),
        max_target_channels: int = 10,
        num_channels: Union[int, Tuple[int, int]] = (1, 256),
        max_kernels: Union[int, Tuple[int, int]] = (1, 10),
        dirichlet_min: Union[float, Tuple[float, float]] = (0.1, 1.0),
        dirichlet_max: Union[float, Tuple[float, float]] = (1.0, 5.0),
        scale: Union[float, Tuple[float, float]] = (0.5, 2.0),
        weibull_shape: Union[float, Tuple[float, float]] = (1.0, 5.0),
        weibull_scale: Union[int, Tuple[int, int]] = (1, 3),
        periodicities: List[str] = None,
    ):
        """
        Initialize the MultivariateTimeSeriesGenerator.

        Parameters
        ----------
        global_seed : int, optional
            Global random seed for reproducibility (default: 42).
        distribution_type : str, optional
            Type of distribution to use for sampling parameters ("uniform" or "log_uniform", default: "uniform").
        history_length : Union[int, Tuple[int, int]], optional
            Fixed history length or range (min, max) (default: (64, 256)).
        target_length : Union[int, Tuple[int, int]], optional
            Fixed target length or range (min, max) (default: (32, 256)).
        max_target_channels : int, optional
            Maximum number of target channels to randomly select (default: 10).
        num_channels : Union[int, Tuple[int, int]], optional
            Fixed number of channels or range (min, max) (default: (1, 256)).
        max_kernels : Union[int, Tuple[int, int]], optional
            Fixed max_kernels value or range (min, max) (default: (1, 10)).
        dirichlet_min : Union[float, Tuple[float, float]], optional
            Fixed dirichlet_min value or range (min, max) (default: (0.1, 1.0)).
        dirichlet_max : Union[float, Tuple[float, float]], optional
            Fixed dirichlet_max value or range (min, max) (default: (1.0, 5.0)).
        scale : Union[float, Tuple[float, float]], optional
            Fixed scale value or range (min, max) (default: (0.5, 2.0)).
        weibull_shape : Union[float, Tuple[float, float]], optional
            Fixed weibull_shape value or range (min, max) (default: (1.0, 5.0)).
        weibull_scale : Union[int, Tuple[int, int]], optional
            Fixed weibull_scale value or range (min, max) (default: (1, 3)).
        periodicities : List[str], optional
            List of possible periodicities to sample from (default: ["s", "m", "h", "D", "W"]).
        """
        self._validate_input_parameters(
            history_length=history_length,
            target_length=target_length,
            max_target_channels=max_target_channels,
            num_channels=num_channels,
            max_kernels=max_kernels,
            dirichlet_min=dirichlet_min,
            dirichlet_max=dirichlet_max,
            scale=scale,
            weibull_shape=weibull_shape,
            weibull_scale=weibull_scale,
        )

        self.global_seed = global_seed
        self.max_target_channels = max_target_channels
        self.distribution_type = distribution_type

        # Parameter configurations
        self.history_length = history_length
        self.target_length = target_length
        self.num_channels = num_channels
        self.max_kernels = max_kernels
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.scale = scale
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale
        self.periodicities = (
            periodicities if periodicities is not None else ["s", "m", "h", "D", "W"]
        )

        # Set random seeds
        self._set_random_seeds(self.global_seed)

    def _validate_input_parameters(
        self,
        history_length: Union[int, Tuple[int, int]],
        target_length: Union[int, Tuple[int, int]],
        max_target_channels: int,
        num_channels: Union[int, Tuple[int, int]],
        max_kernels: Union[int, Tuple[int, int]],
        dirichlet_min: Union[float, Tuple[float, float]],
        dirichlet_max: Union[float, Tuple[float, float]],
        scale: Union[float, Tuple[float, float]],
        weibull_shape: Union[float, Tuple[float, float]],
        weibull_scale: Union[int, Tuple[int, int]],
    ) -> None:
        """
        Validate input parameters to ensure they have valid values.

        Parameters
        ----------
        history_length : Union[int, Tuple[int, int]]
            Fixed history length or range (min, max).
        target_length : Union[int, Tuple[int, int]]
            Fixed target length or range (min, max).
        max_target_channels : int
            Maximum number of target channels to randomly select.
        num_channels : Union[int, Tuple[int, int]]
            Fixed number of channels or range (min, max).
        max_kernels : Union[int, Tuple[int, int]]
            Fixed max_kernels value or range (min, max).
        dirichlet_min : Union[float, Tuple[float, float]]
            Fixed dirichlet_min value or range (min, max).
        dirichlet_max : Union[float, Tuple[float, float]]
            Fixed dirichlet_max value or range (min, max).
        scale : Union[float, Tuple[float, float]]
            Fixed scale value or range (min, max).
        weibull_shape : Union[float, Tuple[float, float]]
            Fixed weibull_shape value or range (min, max).
        weibull_scale : Union[int, Tuple[int, int]]
            Fixed weibull_scale value or range (min, max).
        """
        # Dictionary of parameters to validate
        tuple_params = {
            "history_length": history_length,
            "target_length": target_length,
            "num_channels": num_channels,
            "max_kernels": max_kernels,
            "dirichlet_min": dirichlet_min,
            "dirichlet_max": dirichlet_max,
            "scale": scale,
            "weibull_shape": weibull_shape,
            "weibull_scale": weibull_scale,
        }

        # Check each tuple parameter to ensure min < max
        for param_name, param_value in tuple_params.items():
            if isinstance(param_value, tuple):
                min_val, max_val = param_value
                if min_val > max_val:
                    raise ValueError(
                        f"For parameter '{param_name}', the minimum value ({min_val}) "
                        f"cannot exceed the maximum value ({max_val})"
                    )

        # Validate max_target_channels
        max_num_channels = (
            num_channels if isinstance(num_channels, int) else max(num_channels)
        )
        if max_target_channels > max_num_channels:
            raise ValueError(
                f"max_target_channels ({max_target_channels}) cannot exceed the maximum "
                f"value of num_channels ({max_num_channels})"
            )

    def _set_random_seeds(self, seed: int) -> None:
        """
        Set random seeds for numpy and torch for reproducibility.

        Parameters
        ----------
        seed : int
            The random seed to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _parse_param_value(
        self,
        param_config: Union[int, float, Tuple[int, float], Tuple[float, float]],
        is_int: bool = True,
    ) -> Union[int, float]:
        """
        Parse a parameter configuration which can be either a fixed value or a range.
        If it's a fixed value, return it directly; if it's a range, sample from it.

        Parameters
        ----------
        param_config : Union[int, float, Tuple[int, float], Tuple[float, float]]
            Parameter configuration, either a fixed value or a (min, max) range.
        is_int : bool, optional
            Whether to return an integer value (default: True).

        Returns
        -------
        Union[int, float]
            The fixed value or a sampled value from the range.
        """
        if isinstance(param_config, (int, float)):
            return int(param_config) if is_int else float(param_config)

        min_val, max_val = param_config
        return self._sample_from_range(min_val, max_val, is_int)

    def _sample_from_range(
        self,
        min_val: Union[int, float],
        max_val: Union[int, float],
        is_int: bool = True,
    ) -> Union[int, float]:
        """
        Sample a value from the specified range using the configured distribution type.

        Parameters
        ----------
        min_val : Union[int, float]
            Minimum value of the range.
        max_val : Union[int, float]
            Maximum value of the range.
        is_int : bool, optional
            Whether to return an integer value (default: True).

        Returns
        -------
        Union[int, float]
            A sampled value from the specified range.
        """
        if min_val == max_val:
            return min_val

        if self.distribution_type == "uniform":
            value = np.random.uniform(min_val, max_val)
        elif self.distribution_type == "log_uniform":
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            value = 10 ** np.random.uniform(log_min, log_max)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return int(value) if is_int else value

    def _split_time_series_data(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        history_length: int,
        target_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        Split time series data into history and target components.

        Parameters
        ----------
        values : np.ndarray
            Time series values with shape (batch_size, total_length, num_channels).
        timestamps : np.ndarray
            Timestamps with shape (batch_size, total_length).
        history_length : int
            Length of the history window.
        target_length : int
            Length of the target window.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
            Tuple containing (history_values, future_values, history_timestamps, target_timestamps).
        """
        # Split values into history and target
        history_values = torch.tensor(
            values[:, :history_length, :], dtype=torch.float32
        )
        future_values = torch.tensor(
            values[:, history_length : history_length + target_length, :],
            dtype=torch.float32,
        )

        # Split timestamps
        history_timestamps = timestamps[:, :history_length]
        target_timestamps = timestamps[
            :, history_length : history_length + target_length
        ]

        return history_values, future_values, history_timestamps, target_timestamps

    def _select_target_channels(
        self,
        batch_size: int,
        num_channels: int,
        max_target_channels: int,
        future_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select target channels.

        Parameters
        ----------
        batch_size : int
            Number of time series in the batch.
        num_channels : int
            Number of channels in each time series.
        max_target_channels : int
            Maximum number of target channels to randomly select.
        future_values : torch.Tensor
            Future values tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (target_values, target_indices).
        """
        # Randomly select the number of target channels (between 1 and max_target_channels)
        num_targets = min(np.random.randint(1, max_target_channels + 1), num_channels)

        # Randomly select target channels indices (same for all series in the batch)
        target_indices = torch.tensor(
            np.random.choice(num_channels, size=num_targets, replace=False),
            dtype=torch.long,
        ).expand(batch_size, -1)

        # Vectorized indexing for target values
        target_values = future_values[
            :, :, target_indices[0]
        ]  # Shape: (batch_size, target_length, num_targets)

        return target_values, target_indices

    def format_to_container(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        history_length: int,
        target_length: int,
        batch_size: int,
        num_channels: int,
        max_target_channels: Optional[int] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Format generated time series data into a TimeSeriesDataContainer.

        Parameters
        ----------
        values : np.ndarray
            Generated time series values with shape (batch_size, total_length, num_channels).
        timestamps : np.ndarray
            Generated timestamps with shape (batch_size, total_length).
        history_length : int
            Length of the history window.
        target_length : int
            Length of the target window.
        batch_size : int
            Number of time series in the batch.
        num_channels : int
            Number of channels in each time series.
        max_target_channels : int, optional
            Maximum number of target channels to randomly select (default: self.max_target_channels).

        Returns
        -------
        TimeSeriesDataContainer
            A container with the formatted time series data.
        """
        if max_target_channels is None:
            max_target_channels = self.max_target_channels

        # Split data into history and target components
        history_values, future_values, history_timestamps, target_timestamps = (
            self._split_time_series_data(
                values, timestamps, history_length, target_length
            )
        )

        # Prepare time features
        history_time_features = compute_time_features(
            history_timestamps, include_subday=True
        )
        target_time_features = compute_time_features(
            target_timestamps, include_subday=True
        )

        # Select target channels
        target_values, target_indices = self._select_target_channels(
            batch_size, num_channels, max_target_channels, future_values
        )

        return BatchTimeSeriesContainer(
            history_values=history_values,
            target_values=target_values,
            target_channels_indices=target_indices,
            history_time_features=history_time_features,
            target_time_features=target_time_features,
            static_features=None,  # Not used for now
            history_mask=None,  # Not used for now
            target_mask=None,  # Not used for now
        )

    def _sample_batch_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for a batch generation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        history_length = self._parse_param_value(self.history_length)
        target_length = self._parse_param_value(self.target_length)
        num_channels = self._parse_param_value(self.num_channels)
        max_kernels = self._parse_param_value(self.max_kernels)
        dirichlet_min = self._parse_param_value(self.dirichlet_min, is_int=False)
        dirichlet_max = self._parse_param_value(self.dirichlet_max, is_int=False)
        scale = self._parse_param_value(self.scale, is_int=False)
        weibull_shape = self._parse_param_value(self.weibull_shape, is_int=False)
        weibull_scale = self._parse_param_value(self.weibull_scale, is_int=True)
        periodicity = np.random.choice(self.periodicities)

        # Ensure dirichlet_min < dirichlet_max
        if dirichlet_min > dirichlet_max:
            dirichlet_min, dirichlet_max = dirichlet_max, dirichlet_min

        return {
            "history_length": history_length,
            "target_length": target_length,
            "num_channels": num_channels,
            "max_kernels": max_kernels,
            "dirichlet_min": dirichlet_min,
            "dirichlet_max": dirichlet_max,
            "scale": scale,
            "weibull_shape": weibull_shape,
            "weibull_scale": weibull_scale,
            "periodicity": periodicity,
        }

    def generate_batch(
        self,
        batch_size: int,
        history_length: int,
        target_length: int,
        num_channels: int,
        max_kernels: int = 5,
        dirichlet_min: float = 0.1,
        dirichlet_max: float = 2.0,
        scale: float = 1.0,
        weibull_shape: float = 2.0,
        weibull_scale: int = 1,
        periodicity: str = "s",
        seed: Optional[int] = None,
        max_target_channels: Optional[int] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        history_length : int
            Length of the history window.
        target_length : int
            Length of the target window.
        num_channels : int
            Number of channels in each time series.
        max_kernels : int, optional
            Maximum number of kernels per latent function (default: 5).
        dirichlet_min : float, optional
            Minimum value for Dirichlet parameter (default: 0.1).
        dirichlet_max : float, optional
            Maximum value for Dirichlet parameter (default: 2.0).
        scale : float, optional
            Scaling factor for Weibull distribution (default: 1.0).
        weibull_shape : float, optional
            Shape parameter for Weibull distribution (default: 2.0).
        weibull_scale : int, optional
            Scale parameter for Weibull distribution (default: 1).
        periodicity : str, optional
            Time step periodicity for timestamps (default: "s").
        seed : int, optional
            Random seed for this batch (default: None).
        max_target_channels : int, optional
            Maximum number of target channels to randomly select (default: self.max_target_channels).

        Returns
        -------
        TimeSeriesDataContainer
            A container with the generated time series data.
        """
        # Set seeds if provided
        if seed is not None:
            self._set_random_seeds(seed)

        total_length = history_length + target_length

        # Initialize LMCSynthGenerator with the specified parameters
        generator = LMCSynthGenerator(
            length=total_length,
            max_kernels=max_kernels,
            num_channels=num_channels,
            dirichlet_min=dirichlet_min,
            dirichlet_max=dirichlet_max,
            scale=scale,
            weibull_shape=weibull_shape,
            weibull_scale=weibull_scale,
        )

        # Generate batch of time series
        batch_values, batch_timestamps = self._generate_time_series_batch(
            generator, batch_size, periodicity, seed
        )

        # Format the data into a TimeSeriesDataContainer
        return self.format_to_container(
            values=batch_values,
            timestamps=batch_timestamps,
            history_length=history_length,
            target_length=target_length,
            batch_size=batch_size,
            num_channels=num_channels,
            max_target_channels=max_target_channels,
        )

    def _generate_time_series_batch(
        self,
        generator: LMCSynthGenerator,
        batch_size: int,
        periodicity: str,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of time series using the LMCSynthGenerator.

        Parameters
        ----------
        generator : LMCSynthGenerator
            Initialized generator instance.
        batch_size : int
            Number of time series to generate.
        periodicity : str
            Time step periodicity for timestamps.
        seed : int, optional
            Random seed to use (default: None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (batch_values, batch_timestamps).
        """
        batch_values = []
        batch_timestamps = []

        for i in range(batch_size):
            # Generate a single time series with a unique seed
            batch_seed = None if seed is None else seed + i
            result = generator.generate_time_series(
                random_seed=batch_seed,
                periodicity=periodicity,
            )
            batch_values.append(result["values"])
            batch_timestamps.append(result["timestamps"])

        # Convert to numpy arrays
        return np.array(batch_values), np.array(batch_timestamps)

    def _generate_batch_worker(
        self, args: Tuple[int, Dict[str, Any]]
    ) -> Tuple[BatchTimeSeriesContainer, int]:
        """
        Helper function for parallel batch generation.

        Parameters
        ----------
        args : Tuple[int, Dict[str, Any]]
            Tuple containing (batch_index, parameters_dict).

        Returns
        -------
        Tuple[TimeSeriesDataContainer, int]
            Tuple containing (data_container, batch_index).
        """
        batch_index, params = args
        seed = self.global_seed + batch_index
        logger.info(f"Generating batch {batch_index} with seed {seed}")
        return self.generate_batch(seed=seed, **params), batch_index

    def generate_dataset(
        self,
        num_batches: int,
        batch_size: int,
        num_cpus: Optional[int] = None,
    ) -> Generator[Tuple[BatchTimeSeriesContainer, int], None, None]:
        """
        Generate a dataset of batches of synthetic multivariate time series.

        Parameters
        ----------
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        num_cpus : int, optional
            Number of parallel CPUs for batch generation. If None, use all available CPUs (default: None).

        Yields
        ------
        Tuple[TimeSeriesDataContainer, int]
            A tuple containing the TimeSeriesDataContainer and the batch index.
        """
        num_cpus = self._determine_cpu_count(num_cpus)
        logger.info(f"Using {num_cpus} CPUs for dataset generation")

        # Prepare batch parameters
        batch_params = self._prepare_batch_parameters(num_batches, batch_size)

        # Generate batches in parallel
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(self._generate_batch_worker, args)
                for args in batch_params
            ]
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"Error in batch generation: {e}")
                    raise

    def _determine_cpu_count(self, num_cpus: Optional[int] = None) -> int:
        """
        Determine the number of CPUs to use for parallel processing.

        Parameters
        ----------
        num_cpus : int, optional
            Requested number of CPUs (default: None).

        Returns
        -------
        int
            Number of CPUs to use.
        """
        if num_cpus is None:
            return min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overload
        return num_cpus

    def _prepare_batch_parameters(
        self, num_batches: int, batch_size: int
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Prepare parameters for each batch generation.

        Parameters
        ----------
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.

        Returns
        -------
        List[Tuple[int, Dict[str, Any]]]
            List of (batch_index, parameters) tuples.
        """
        batch_params = []
        for i in range(num_batches):
            params = self._sample_batch_parameters()
            params["batch_size"] = batch_size
            batch_params.append((i, params))
        return batch_params

    def _save_batch(
        self, batch: BatchTimeSeriesContainer, batch_idx: int, output_dir: str
    ) -> None:
        """
        Save a batch to disk.

        Parameters
        ----------
        batch : TimeSeriesDataContainer
            The batch to save.
        batch_idx : int
            Batch index.
        output_dir : str
            Output directory path.
        """
        batch_path = os.path.join(output_dir, f"batch_{batch_idx:03d}.pt")
        torch.save(batch, batch_path)
        logger.info(f"Saved batch {batch_idx} to {batch_path}")

    def _process_dataset_chunk(
        self,
        chunk_start: int,
        chunk_end: int,
        chunk_size: int,
        batch_size: int,
        num_cpus: int,
        output_dir: str,
        combined_file_path: str,
    ) -> None:
        """
        Process a chunk of batches for large dataset generation.

        Parameters
        ----------
        chunk_start : int
            Starting index of the chunk.
        chunk_end : int
            Ending index of the chunk.
        chunk_size : int
            Size of the chunk.
        batch_size : int
            Number of time series per batch.
        num_cpus : int
            Number of CPUs to use.
        output_dir : str
            Output directory path.
        combined_file_path : str
            Path to the combined dataset file.
        """
        chunk_size_actual = chunk_end - chunk_start
        logger.info(f"Processing batches {chunk_start} to {chunk_end - 1}...")

        # Generate and collect batches for this chunk
        chunk_batches = []
        chunk_indices = []

        with tqdm(
            total=chunk_size_actual,
            desc=f"Generating chunk {(chunk_start // chunk_size) + 1}/{(chunk_end // chunk_size) + 1}",
        ) as pbar:
            for batch, batch_idx in self.generate_dataset(
                num_batches=chunk_size_actual,
                batch_size=batch_size,
                num_cpus=num_cpus,
            ):
                adjusted_batch_idx = batch_idx + chunk_start
                chunk_batches.append(batch)
                chunk_indices.append(adjusted_batch_idx)
                pbar.update(1)

        # Sort this chunk's batches by index
        sorted_indices = np.argsort(chunk_indices)
        sorted_chunk_batches = [chunk_batches[i] for i in sorted_indices]

        # Save or append to the combined file
        self._save_or_append_chunk(
            sorted_chunk_batches, chunk_start, combined_file_path, output_dir
        )

        # Clean up memory
        self._clean_memory([chunk_batches, chunk_indices, sorted_chunk_batches])
        logger.info(f"Completed chunk {chunk_start} to {chunk_end - 1}")

    def _save_or_append_chunk(
        self,
        sorted_chunk_batches: List[BatchTimeSeriesContainer],
        chunk_start: int,
        combined_file_path: str,
        output_dir: str,
    ) -> None:
        """
        Save or append a chunk of batches to the combined dataset file.

        Parameters
        ----------
        sorted_chunk_batches : List[TimeSeriesDataContainer]
            Sorted list of batches in the chunk.
        chunk_start : int
            Starting index of the chunk.
        combined_file_path : str
            Path to the combined dataset file.
        output_dir : str
            Output directory path.
        """
        if chunk_start == 0:
            # First chunk: create new file
            torch.save(sorted_chunk_batches, combined_file_path)
        else:
            # Subsequent chunks: load existing file, append, and save
            try:
                existing_data = torch.load(combined_file_path)
                combined_data = existing_data + sorted_chunk_batches
                torch.save(combined_data, combined_file_path)
            except Exception as e:
                logger.error(f"Error appending to dataset file: {e}")
                chunk_file = os.path.join(output_dir, f"dataset_chunk_{chunk_start}.pt")
                torch.save(sorted_chunk_batches, chunk_file)

    def _clean_memory(self, variables_to_delete: List[Any]) -> None:
        """
        Clean memory by deleting variables and forcing garbage collection.

        Parameters
        ----------
        variables_to_delete : List[Any]
            List of variables to delete.
        """
        for var in variables_to_delete:
            del var
        gc.collect()

    def save_dataset(
        self,
        output_dir: str,
        num_batches: int,
        batch_size: int,
        save_as_single_file: bool = False,
        num_cpus: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Generate and save a dataset to disk using a memory-efficient streaming approach.

        Parameters
        ----------
        output_dir : str
            Directory to save the dataset.
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        save_as_single_file : bool, optional
            If True, save all batches in a single file (default: False).
        num_cpus : int, optional
            Number of parallel CPUs for batch generation (default: None, uses min(cpu_count, 8)).
        chunk_size : int, optional
            Number of batches to process at once in single-file mode (default: 10).
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving dataset to {output_dir} with {num_batches} batches")

        num_cpus = self._determine_cpu_count(num_cpus)
        logger.info(f"Using {num_cpus} CPUs for saving dataset")

        if save_as_single_file:
            self._save_as_single_file(
                output_dir, num_batches, batch_size, num_cpus, chunk_size
            )
        else:
            self._save_as_individual_files(
                output_dir, num_batches, batch_size, num_cpus
            )

        logger.info(f"Dataset saved to {output_dir}")

    def _save_as_single_file(
        self,
        output_dir: str,
        num_batches: int,
        batch_size: int,
        num_cpus: int,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Save dataset as a single file, processing in chunks to manage memory.

        Parameters
        ----------
        output_dir : str
            Directory to save the dataset.
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        num_cpus : int
            Number of CPUs to use for generation.
        chunk_size : int, optional
            Number of batches to process at once (default: min(10, num_batches)).
        """
        # Path for the combined dataset file
        combined_file_path = os.path.join(output_dir, "dataset.pt")

        # Set default chunk size if not provided
        if chunk_size is None:
            chunk_size = min(10, num_batches)

        # Process in chunks to avoid memory issues
        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            self._process_dataset_chunk(
                chunk_start,
                chunk_end,
                chunk_size,
                batch_size,
                num_cpus,
                output_dir,
                combined_file_path,
            )

    def _save_as_individual_files(
        self,
        output_dir: str,
        num_batches: int,
        batch_size: int,
        num_cpus: int,
    ) -> None:
        """
        Save dataset as individual batch files.

        Parameters
        ----------
        output_dir : str
            Directory to save the dataset.
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        num_cpus : int
            Number of CPUs to use for generation.
        """
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = []
            with tqdm(total=num_batches, desc="Generating and saving batches") as pbar:
                for batch, batch_idx in self.generate_dataset(
                    num_batches=num_batches,
                    batch_size=batch_size,
                    num_cpus=num_cpus,
                ):
                    futures.append(
                        executor.submit(self._save_batch, batch, batch_idx, output_dir)
                    )
                    pbar.update(1)

                # Ensure any exceptions are raised
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error saving batch: {e}")
                        raise

        logger.info(f"Dataset saved to {output_dir}")
