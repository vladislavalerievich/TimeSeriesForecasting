from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.base_generator_wrapper import BaseGeneratorWrapper
from src.synthetic_generation.constants import DEFAULT_START_DATE
from src.synthetic_generation.lmc_synth import LMCSynthGenerator


class LMCGeneratorWrapper(BaseGeneratorWrapper):
    """
    Wrapper for LMCSynthGenerator to generate batches of multivariate time series data.
    """

    def __init__(
        self,
        global_seed: int = 42,
        distribution_type: str = "uniform",
        history_length: Union[int, Tuple[int, int]] = (64, 256),
        target_length: Union[int, Tuple[int, int]] = (32, 256),
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
        Initialize the LMCGeneratorWrapper.

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
        super().__init__(
            global_seed=global_seed,
            distribution_type=distribution_type,
            history_length=history_length,
            target_length=target_length,
            num_channels=num_channels,
            periodicities=periodicities,
        )

        # LMC-specific parameters
        self.max_kernels = max_kernels
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.scale = scale
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with LMCSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        # Get common parameters from parent class
        params = super().sample_parameters()

        # Sample LMC-specific parameters
        max_kernels = self._parse_param_value(self.max_kernels)
        dirichlet_min = self._parse_param_value(self.dirichlet_min, is_int=False)
        dirichlet_max = self._parse_param_value(self.dirichlet_max, is_int=False)
        scale = self._parse_param_value(self.scale, is_int=False)
        weibull_shape = self._parse_param_value(self.weibull_shape, is_int=False)
        weibull_scale = self._parse_param_value(self.weibull_scale)

        # Ensure dirichlet_min < dirichlet_max
        if dirichlet_min > dirichlet_max:
            dirichlet_min, dirichlet_max = dirichlet_max, dirichlet_min

        # Add LMC-specific parameters to the dictionary
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
