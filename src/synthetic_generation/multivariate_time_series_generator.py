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
        """
        self.global_seed = global_seed
        self.max_target_channels = max_target_channels
        self.distribution_type = distribution_type
        np.random.seed(self.global_seed)
        torch.manual_seed(self.global_seed)

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

        # Convert timestamps to time features (for now just use raw timestamps)
        history_timestamps = timestamps[:, :history_length]
        target_timestamps = timestamps[
            :, history_length : history_length + target_length
        ]

        # Convert timestamps to float tensor (later will be replaced with sine/cosine encoding)
        # Convert datetime64 to float by subtracting the minimum timestamp
        history_time_features = torch.tensor(
            np.array(
                [
                    (ts - timestamps[:, 0].min()) / np.timedelta64(1, "D")
                    for ts in history_timestamps
                ]
            ),
            dtype=torch.float32,
        ).reshape(batch_size, history_length, 1)

        target_time_features = torch.tensor(
            np.array(
                [
                    (ts - timestamps[:, 0].min()) / np.timedelta64(1, "D")
                    for ts in target_timestamps
                ]
            ),
            dtype=torch.float32,
        ).reshape(batch_size, target_length, 1)

        # Randomly select the number of target channels (between 1 and max_target_channels)
        num_targets = min(np.random.randint(1, max_target_channels + 1), num_channels)

        # Randomly select target channels indices (same indices for all series in the batch)
        target_indices = torch.tensor(
            np.random.choice(num_channels, size=num_targets, replace=False),
            dtype=torch.long,
        ).expand(batch_size, -1)

        # Extract target values based on selected indices
        target_values = torch.zeros(
            (batch_size, target_length, num_targets), dtype=torch.float32
        )
        for i in range(batch_size):
            target_values[i] = future_values[i, :, target_indices[i]]

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
        weibull_scale: float = 1.0,
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
        weibull_scale : float, optional
            Scale parameter for Weibull distribution (default: 1.0).
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
            batch_values.append(result["values"])
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
        min_history_length: int = 64,
        max_history_length: int = 256,
        min_target_length: int = 32,
        max_target_length: int = 256,
        min_num_channels: int = 1,
        max_num_channels: int = 256,
        min_max_kernels: int = 1,
        max_max_kernels: int = 10,
        min_dirichlet_min: float = 0.1,
        max_dirichlet_min: float = 1.0,
        min_dirichlet_max: float = 1.0,
        max_dirichlet_max: float = 5.0,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        min_weibull_shape: float = 1.0,
        max_weibull_shape: float = 5.0,
        min_weibull_scale: float = 0.5,
        max_weibull_scale: float = 2.0,
        periodicities: List[str] = None,
        fixed_history_length: Optional[int] = None,
        fixed_target_length: Optional[int] = None,
        fixed_num_channels: Optional[int] = None,
        fixed_max_kernels: Optional[int] = None,
        fixed_dirichlet_min: Optional[float] = None,
        fixed_dirichlet_max: Optional[float] = None,
        fixed_scale: Optional[float] = None,
        fixed_weibull_shape: Optional[float] = None,
        fixed_weibull_scale: Optional[float] = None,
        fixed_periodicity: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> Generator[Tuple[TimeSeriesDataContainer, int], None, None]:
        """
        Generate a dataset of batches of synthetic multivariate time series.

        Parameters
        ----------
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        min_history_length : int, optional
            Minimum length of history window (default: 64).
        max_history_length : int, optional
            Maximum length of history window (default: 256).
        min_target_length : int, optional
            Minimum length of target window (default: 32).
        max_target_length : int, optional
            Maximum length of target window (default: 256).
        min_num_channels : int, optional
            Minimum number of channels (default: 1).
        max_num_channels : int, optional
            Maximum number of channels (default: 256).
        min_max_kernels : int, optional
            Minimum value for max_kernels parameter (default: 1).
        max_max_kernels : int, optional
            Maximum value for max_kernels parameter (default: 10).
        min_dirichlet_min : float, optional
            Minimum value for dirichlet_min parameter (default: 0.1).
        max_dirichlet_min : float, optional
            Maximum value for dirichlet_min parameter (default: 1.0).
        min_dirichlet_max : float, optional
            Minimum value for dirichlet_max parameter (default: 1.0).
        max_dirichlet_max : float, optional
            Maximum value for dirichlet_max parameter (default: 5.0).
        min_scale : float, optional
            Minimum value for scale parameter (default: 0.5).
        max_scale : float, optional
            Maximum value for scale parameter (default: 2.0).
        min_weibull_shape : float, optional
            Minimum value for weibull_shape parameter (default: 1.0).
        max_weibull_shape : float, optional
            Maximum value for weibull_shape parameter (default: 5.0).
        min_weibull_scale : float, optional
            Minimum value for weibull_scale parameter (default: 0.5).
        max_weibull_scale : float, optional
            Maximum value for weibull_scale parameter (default: 2.0).
        periodicities : List[str], optional
            List of possible periodicities to sample from (default: ["s", "m", "h", "D", "W"]).
        fixed_history_length : int, optional
            If provided, use this fixed history length for all batches (default: None).
        fixed_target_length : int, optional
            If provided, use this fixed target length for all batches (default: None).
        fixed_num_channels : int, optional
            If provided, use this fixed number of channels for all batches (default: None).
        fixed_max_kernels : int, optional
            If provided, use this fixed max_kernels for all batches (default: None).
        fixed_dirichlet_min : float, optional
            If provided, use this fixed dirichlet_min for all batches (default: None).
        fixed_dirichlet_max : float, optional
            If provided, use this fixed dirichlet_max for all batches (default: None).
        fixed_scale : float, optional
            If provided, use this fixed scale for all batches (default: None).
        fixed_weibull_shape : float, optional
            If provided, use this fixed weibull_shape for all batches (default: None).
        fixed_weibull_scale : float, optional
            If provided, use this fixed weibull_scale for all batches (default: None).
        fixed_periodicity : str, optional
            If provided, use this fixed periodicity for all batches (default: None).
        n_jobs : int, optional
            Number of parallel jobs for batch generation. If None, use all available CPUs (default: None).

        Yields
        ------
        Tuple[TimeSeriesDataContainer, int]
            A tuple containing the TimeSeriesDataContainer and the batch index.
        """
        if periodicities is None:
            periodicities = ["s", "m", "h", "D", "W"]

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()

        # Prepare batch parameters
        batch_params = []
        for i in range(num_batches):
            # Sample parameters for this batch
            history_length = fixed_history_length or self._sample_from_range(
                min_history_length, max_history_length
            )
            target_length = fixed_target_length or self._sample_from_range(
                min_target_length, max_target_length
            )
            num_channels = fixed_num_channels or self._sample_from_range(
                min_num_channels, max_num_channels
            )
            max_kernels = fixed_max_kernels or self._sample_from_range(
                min_max_kernels, max_max_kernels
            )
            dirichlet_min = fixed_dirichlet_min or self._sample_from_range(
                min_dirichlet_min, max_dirichlet_min, is_int=False
            )
            dirichlet_max = fixed_dirichlet_max or self._sample_from_range(
                min_dirichlet_max, max_dirichlet_max, is_int=False
            )
            scale = fixed_scale or self._sample_from_range(
                min_scale, max_scale, is_int=False
            )
            weibull_shape = fixed_weibull_shape or self._sample_from_range(
                min_weibull_shape, max_weibull_shape, is_int=False
            )
            weibull_scale = fixed_weibull_scale or self._sample_from_range(
                min_weibull_scale, max_weibull_scale, is_int=False
            )
            periodicity = fixed_periodicity or np.random.choice(periodicities)

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
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(self._generate_batch_worker, batch_params):
                yield result

    def save_dataset(
        self,
        output_dir: str,
        num_batches: int,
        batch_size: int,
        save_as_single_file: bool = False,
        **dataset_params,
    ) -> None:
        """
        Generate and save a dataset to disk.

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
        **dataset_params : dict
            Additional parameters to pass to generate_dataset.
        """
        os.makedirs(output_dir, exist_ok=True)

        if save_as_single_file:
            all_batches = []
            all_indices = []

        for batch, batch_idx in tqdm(
            self.generate_dataset(
                num_batches=num_batches, batch_size=batch_size, **dataset_params
            ),
            total=num_batches,
            desc="Generating and saving batches",
        ):
            if save_as_single_file:
                all_batches.append(batch)
                all_indices.append(batch_idx)
            else:
                # Save individual batch
                batch_path = os.path.join(output_dir, f"batch_{batch_idx:03d}.pt")
                torch.save(batch, batch_path)

        if save_as_single_file:
            # Sort by batch index
            sorted_batches = [b for _, b in sorted(zip(all_indices, all_batches))]
            torch.save(sorted_batches, os.path.join(output_dir, "dataset.pt"))

        print(f"Dataset saved to {output_dir}")


class MultivariateTimeSeriesDataset(torch.utils.data.IterableDataset):
    """
    PyTorch IterableDataset wrapper for MultivariateTimeSeriesGenerator.
    """

    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        global_seed: int = 42,
        **generator_params,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        num_batches : int
            Number of batches to generate.
        batch_size : int
            Number of time series per batch.
        global_seed : int, optional
            Global random seed for reproducibility (default: 42).
        **generator_params : dict
            Additional parameters to pass to MultivariateTimeSeriesGenerator.generate_dataset.
        """
        super().__init__()
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.global_seed = global_seed
        self.generator_params = generator_params

    def __iter__(self):
        """
        Return an iterator over the batches.
        """
        generator = MultivariateTimeSeriesGenerator(global_seed=self.global_seed)
        for batch, _ in generator.generate_dataset(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            **self.generator_params,
        ):
            yield batch
