from typing import Any, Dict, Optional

import numpy as np

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import LMCGeneratorParams
from src.synthetic_generation.lmc_synth.lmc_synth import LMCSynthGenerator


class LMCGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for LMCSynthGenerator to generate batches of multivariate time series data.
    Accepts an LMCGeneratorParams dataclass for configuration.
    """

    def __init__(self, params: LMCGeneratorParams):
        super().__init__(params)
        self.params: LMCGeneratorParams = params

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with LMCSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super()._sample_parameters()
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
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate a batch of time series using the LMCSynthGenerator.

        Parameters
        ----------
        generator : LMCSynthGenerator
            Initialized generator instance.
        batch_size : int
            Number of time series to generate.
        seed : int, optional
            Random seed to use (default: None).

        Returns
        -------
        np.ndarray
            Shape: [seq_len, num_channels]
        """
        batch_values = []

        for i in range(batch_size):
            # Generate a single time series with a unique seed
            batch_seed = None if seed is None else seed + i
            result = generator.generate_time_series(
                random_seed=batch_seed,
            )
            batch_values.append(result)

        # Convert to numpy arrays
        return np.array(batch_values)

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
            params = self._sample_parameters()

        # Initialize LMCSynthGenerator with the specified parameters
        generator = LMCSynthGenerator(
            length=params["total_length"],
            max_kernels=params["max_kernels"],
            num_channels=params["num_channels"],
            dirichlet_min=params["dirichlet_min"],
            dirichlet_max=params["dirichlet_max"],
            scale=params["scale"],
            weibull_shape=params["weibull_shape"],
            weibull_scale=params["weibull_scale"],
            random_seed=seed,
        )

        # Generate batch of time series
        batch_values = self._generate_time_series_batch(generator, batch_size, seed)

        return self._format_to_container(
            values=batch_values,
            start=params["start"],
            history_length=params["history_length"],
            future_length=params["future_length"],
            frequency=params["frequency"],
        )
