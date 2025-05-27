from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.base_generator_wrapper import BaseGeneratorWrapper
from src.synthetic_generation.constants import DEFAULT_START_DATE
from src.synthetic_generation.kernel_synth import KernelSynthGenerator


class KernelGeneratorWrapper(BaseGeneratorWrapper):
    """
    Wrapper for KernelSynthGenerator to generate batches of multivariate time series data
    by stacking multiple univariate series.
    """

    def __init__(
        self,
        global_seed: int = 42,
        distribution_type: str = "uniform",
        history_length: Union[int, Tuple[int, int]] = (64, 256),
        target_length: Union[int, Tuple[int, int]] = (32, 256),
        num_channels: Union[int, Tuple[int, int]] = (1, 256),
        max_kernels: Union[int, Tuple[int, int]] = (1, 5),
        periodicities: List[str] = None,
    ):
        """
        Initialize the KernelGeneratorWrapper.

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
        num_channels : Union[int, Tuple[int, int]], optional
            Fixed number of channels or range (min, max) (default: (1, 256)).
        max_kernels : Union[int, Tuple[int, int]], optional
            Fixed max_kernels value or range (min, max) (default: (1, 5)).
        periodicities : List[str], optional
            List of possible periodicities to sample from (default: ["s", "m", "h", "D", "W"]).
        """
        super().__init__(
            global_seed=global_seed,
            distribution_type=distribution_type,
            history_length=history_length,
            target_length=target_length,
            num_channels=num_channels,
            periodicities=periodicities,
        )

        # Kernel-specific parameters
        self.max_kernels = max_kernels

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with KernelSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        # Get common parameters from parent class
        params = super().sample_parameters()

        # Sample Kernel-specific parameters
        max_kernels = self._parse_param_value(self.max_kernels)

        # Add Kernel-specific parameters to the dictionary
        params.update(
            {
                "max_kernels": max_kernels,
            }
        )

        return params

    def _generate_univariate_time_series(
        self,
        generator: KernelSynthGenerator,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a single univariate time series.

        Parameters
        ----------
        generator : KernelSynthGenerator
            Initialized generator instance.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        Dict
            Dictionary containing the generated time series.
        """
        return generator.generate_time_series(random_seed=seed)

    def _generate_multivariate_time_series(
        self,
        num_channels: int,
        length: int,
        max_kernels: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single multivariate time series by stacking multiple univariate series.

        Parameters
        ----------
        num_channels : int
            Number of channels (univariate series) to generate.
        length : int
            Length of each time series.
        max_kernels : int
            Maximum number of kernels for generation.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (values, timestamps).
        """
        # Initialize the generator
        generator = KernelSynthGenerator(
            length=length,
            max_kernels=max_kernels,
        )

        # Generate univariate series for each channel
        values = []
        timestamps = None

        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            result = self._generate_univariate_time_series(generator, channel_seed)
            values.append(result["values"])

            # Use the same timestamps for all channels
            if timestamps is None:
                timestamps = result["timestamps"]

        # Stack the univariate series to form a multivariate series
        values = np.column_stack(values)  # Shape: (length, num_channels)

        return values, timestamps

    def _generate_time_series_batch(
        self,
        batch_size: int,
        num_channels: int,
        length: int,
        max_kernels: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of multivariate time series.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        num_channels : int
            Number of channels per time series.
        length : int
            Length of each time series.
        max_kernels : int
            Maximum number of kernels for generation.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (batch_values, batch_timestamps).
        """
        batch_values = []
        batch_timestamps = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values, timestamps = self._generate_multivariate_time_series(
                num_channels=num_channels,
                length=length,
                max_kernels=max_kernels,
                seed=batch_seed,
            )
            batch_values.append(values)
            batch_timestamps.append(timestamps)

        # Reshape values to (batch_size, length, num_channels)
        batch_values = np.array(batch_values).transpose(0, 1, 2)

        return batch_values, np.array(batch_timestamps)

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series using KernelSynthGenerator.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate.
        seed : int, optional
            Random seed for this batch (default: None).
        params : Dict[str, Any], optional
            Pre-sampled parameters to use. If None, parameters will be sampled.

        Returns
        -------
        BatchTimeSeriesContainer
            A container with the generated time series data.
        """
        # Set seeds if provided
        if seed is not None:
            self._set_random_seeds(seed)

        # Sample parameters if not provided
        if params is None:
            params = self.sample_parameters()

        history_length = params["history_length"]
        target_length = params["target_length"]
        num_channels = params["num_channels"]
        max_kernels = params["max_kernels"]

        total_length = history_length + target_length

        # Generate batch of time series
        batch_values, batch_timestamps = self._generate_time_series_batch(
            batch_size=batch_size,
            num_channels=num_channels,
            length=total_length,
            max_kernels=max_kernels,
            seed=seed,
        )

        # Format the data into a BatchTimeSeriesContainer
        return self.format_to_container(
            values=batch_values,
            timestamps=batch_timestamps,
            history_length=history_length,
            target_length=target_length,
            batch_size=batch_size,
            num_channels=num_channels,
        )
