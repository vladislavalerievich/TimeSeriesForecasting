from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.constants import DEFAULT_START_DATE
from src.synthetic_generation.generator_params import GeneratorParams
from src.synthetic_generation.lmc_synth import LMCSynthGenerator


class LMCGeneratorParams(GeneratorParams):
    max_kernels: int = 5
    dirichlet_min: float = 0.1
    dirichlet_max: float = 2.0
    scale: float = 1.0
    weibull_shape: float = 2.0
    weibull_scale: int = 1


class LMCGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for LMCSynthGenerator to generate batches of multivariate time series data.
    Accepts an LMCGeneratorParams dataclass for configuration.
    """

    def __init__(self, params: LMCGeneratorParams):
        super().__init__(params)
        self.params: LMCGeneratorParams = params

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with LMCSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super().sample_parameters()
        # LMC-specific parameters
        max_kernels = self.params.max_kernels
        dirichlet_min = self.params.dirichlet_min
        dirichlet_max = self.params.dirichlet_max
        scale = self.params.scale
        weibull_shape = self.params.weibull_shape
        weibull_scale = self.params.weibull_scale
        params.update(
            {
                "max_kernels": max_kernels,
                "dirichlet_min": dirichlet_min,
                "dirichlet_max": dirichlet_max,
                "scale": scale,
                "weibull_shape": weibull_shape,
                "weibull_scale": weibull_scale,
            }
        )
        return params

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

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series using LMCSynthGenerator.

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
        dirichlet_min = params["dirichlet_min"]
        dirichlet_max = params["dirichlet_max"]
        scale = params["scale"]
        weibull_shape = params["weibull_shape"]
        weibull_scale = params["weibull_scale"]
        periodicity = params["periodicity"]

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

        # Format the data into a BatchTimeSeriesContainer
        return self.format_to_container(
            values=batch_values,
            timestamps=batch_timestamps,
            history_length=history_length,
            target_length=target_length,
            batch_size=batch_size,
            num_channels=num_channels,
        )
