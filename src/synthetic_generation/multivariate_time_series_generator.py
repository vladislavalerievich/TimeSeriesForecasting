import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.data_handling.data_containers import TimeSeriesDataContainer
from src.synthetic_generation.lmc_synth import LMCSynthGenerator


class MultivariateTimeSeriesGenerator:
    """
    Generate batches of synthetic multivariate time series data using LMCSynthGenerator.
    Provides functionality to create and save datasets compatible with PyTorch DataLoader.
    """

    def __init__(
        self,
        global_seed: int = 42,
        max_target_channels: int = 10,
        distribution_type: Literal["uniform", "log_uniform"] = "uniform",
        history_length: Union[int, Tuple[int, int]] = (64, 256),
        target_length: Union[int, Tuple[int, int]] = (32, 256),
        num_channels: Union[int, Tuple[int, int]] = (1, 256),
        max_kernels: Union[int, Tuple[int, int]] = (1, 10),
        dirichlet_min: Union[float, Tuple[float, float]] = (0.1, 1.0),
        dirichlet_max: Union[float, Tuple[float, float]] = (1.0, 5.0),
        scale: Union[float, Tuple[float, float]] = (0.5, 2.0),
        weibull_shape: Union[float, Tuple[float, float]] = (1.0, 5.0),
        weibull_scale: Union[float, Tuple[float, float]] = (1, 3),
        periodicities: List[str] = None,
    ):
        """
        Initialize the MultivariateTimeSeriesGenerator.

        Parameters
        ----------
        global_seed : int, optional
            Global random seed for reproducibility (default: 42).
        max_target_channels : int, optional
            Maximum number of target channels to randomly select (default: 10).
        distribution_type : str, optional
            Type of distribution to use for sampling parameters ("uniform" or "log_uniform", default: "uniform").
        history_length : Union[int, Tuple[int, int]], optional
            Fixed history length or range (min, max) (default: (64, 256)).
        target_length : Union[int, Tuple[int, int]], optional
            Fixed target length or range (min, max) (default: (32, 256)).
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

        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)

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

    def format_to_container(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        history_length: int,
        target_length: int,
        batch_size: int,
        num_channels: int,
        max_target_channels: Optional[int] = None,
    ) -> TimeSeriesDataContainer:
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

        # Vectorized timestamp normalization (days since minimum timestamp)
        min_timestamp = timestamps[:, 0].min()
        history_time_features = torch.tensor(
            (history_timestamps - min_timestamp) / np.timedelta64(1, "D"),
            dtype=torch.float32,
        )[:, :, None]  # Shape: (batch_size, history_length, 1)
        target_time_features = torch.tensor(
            (target_timestamps - min_timestamp) / np.timedelta64(1, "D"),
            dtype=torch.float32,
        )[:, :, None]  # Shape: (batch_size, target_length, 1)

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

        return TimeSeriesDataContainer(
            history_values=history_values,
            target_values=target_values,
            target_channels_indices=target_indices,
            history_time_features=history_time_features,
            target_time_features=target_time_features,
            static_features=None,  # Not used for now
            history_mask=None,  # Not used for now
            target_mask=None,  # Not used for now
        )

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
    ) -> TimeSeriesDataContainer:
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
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

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
        batch_values = []
        batch_timestamps = []

        for i in range(batch_size):
            # Generate a single time series with a unique seed
            result = generator.generate_time_series(
                random_seed=seed + i if seed is not None else None,
                periodicity=periodicity,
            )
            batch_values.append(result["values"])  # Shape: (total_length, num_channels)
            batch_timestamps.append(result["timestamps"])

        # Convert to numpy arrays
        batch_values = np.array(batch_values)
        batch_timestamps = np.array(batch_timestamps)

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

    def _generate_batch_worker(self, args):
        """Helper function for parallel batch generation."""
        batch_index, params = args
        seed = self.global_seed + batch_index
        return self.generate_batch(seed=seed, **params), batch_index

    def generate_dataset(
        self,
        num_batches: int,
        batch_size: int,
        num_cpus: Optional[int] = None,
    ) -> Generator[Tuple[TimeSeriesDataContainer, int], None, None]:
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
        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()

        # Prepare batch parameters
        batch_params = []
        for i in range(num_batches):
            # Sample parameters for this batch
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

            params = {
                "batch_size": batch_size,
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
            batch_params.append((i, params))

        # Generate batches in parallel
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            for result in executor.map(self._generate_batch_worker, batch_params):
                yield result

    def save_dataset(
        self,
        output_dir: str,
        num_batches: int,
        batch_size: int,
        save_as_single_file: bool = False,
        num_cpus: Optional[int] = None,
        chunk_size: int = None,  # Process datasets in chunks to save memory
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
            Number of parallel CPUs for batch generation. If None, use all available CPUs (default: None).
        chunk_size : int, optional
            Number of batches to process at once when saving as a single file (default: 100).
            Smaller values use less memory but may be slower.
        """
        os.makedirs(output_dir, exist_ok=True)

        if save_as_single_file:
            # Path for the combined dataset file
            combined_file_path = os.path.join(output_dir, "dataset.pt")

            # Check if chunk_size is provided, otherwise set a default
            if chunk_size is None:
                chunk_size = batch_size * 10

            # For very large datasets, process in chunks to avoid memory issues
            for chunk_start in range(0, num_batches, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_batches)
                chunk_size_actual = chunk_end - chunk_start

                print(f"Processing batches {chunk_start} to {chunk_end - 1}...")

                # Generate and collect batches for this chunk
                chunk_batches = []
                chunk_indices = []

                for batch, batch_idx in tqdm(
                    self.generate_dataset(
                        num_batches=chunk_size_actual,
                        batch_size=batch_size,
                        num_cpus=num_cpus,
                    ),
                    total=chunk_size_actual,
                    desc=f"Generating chunk {chunk_start // chunk_size + 1}/{(num_batches + chunk_size - 1) // chunk_size}",
                ):
                    # Adjust batch index to account for chunking
                    adjusted_batch_idx = batch_idx + chunk_start
                    chunk_batches.append(batch)
                    chunk_indices.append(adjusted_batch_idx)

                    # Option: save individual batches as well
                    if not save_as_single_file:
                        batch_path = os.path.join(
                            output_dir, f"batch_{adjusted_batch_idx:03d}.pt"
                        )
                        torch.save(batch, batch_path)

                # Sort this chunk's batches by index
                sorted_indices = np.argsort(chunk_indices)
                sorted_chunk_batches = [chunk_batches[i] for i in sorted_indices]

                # Save or append to the combined file
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
                        print(f"Error appending to dataset file: {e}")
                        # Save this chunk separately as fallback
                        chunk_file = os.path.join(
                            output_dir, f"dataset_chunk_{chunk_start // chunk_size}.pt"
                        )
                        torch.save(sorted_chunk_batches, chunk_file)

                # Clear memory
                del chunk_batches, chunk_indices, sorted_chunk_batches
                if "combined_data" in locals():
                    del combined_data
                if "existing_data" in locals():
                    del existing_data

                # Force garbage collection to free memory
                import gc

                gc.collect()
        else:
            # Save individual batches without collecting them all in memory
            for batch, batch_idx in tqdm(
                self.generate_dataset(
                    num_batches=num_batches, batch_size=batch_size, num_cpus=num_cpus
                ),
                total=num_batches,
                desc="Generating and saving batches",
            ):
                batch_path = os.path.join(output_dir, f"batch_{batch_idx:03d}.pt")
                torch.save(batch, batch_path)

        print(f"Dataset saved to {output_dir}")
