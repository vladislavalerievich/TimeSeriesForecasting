from typing import Any, Dict, Optional

import numpy as np

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import KernelGeneratorParams
from src.synthetic_generation.kernel_synth import KernelSynthGenerator


class KernelGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for KernelSynthGenerator to generate batches of multivariate time series data
    by stacking multiple univariate series. Accepts a KernelGeneratorParams dataclass for configuration.
    """

    def __init__(self, params: KernelGeneratorParams):
        super().__init__(params)
        self.params: KernelGeneratorParams = params

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with KernelSynthGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super().sample_parameters()
        max_kernels = self.params.max_kernels
        params.update({"max_kernels": max_kernels})
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
    ) -> tuple:
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
        generator = KernelSynthGenerator(
            length=length,
            max_kernels=max_kernels,
        )
        values = []
        timestamps = None
        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            result = self._generate_univariate_time_series(generator, channel_seed)
            values.append(result["values"])
            if timestamps is None:
                timestamps = result["timestamps"]
        values = np.column_stack(values)
        return values, timestamps

    def _generate_time_series_batch(
        self,
        batch_size: int,
        num_channels: int,
        length: int,
        max_kernels: int,
        seed: Optional[int] = None,
    ) -> tuple:
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
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self.sample_parameters()
        history_length = params["history_length"]
        target_length = params["target_length"]
        num_channels = params["num_channels"]
        max_kernels = params["max_kernels"]
        total_length = history_length + target_length
        batch_values, batch_timestamps = self._generate_time_series_batch(
            batch_size=batch_size,
            num_channels=num_channels,
            length=total_length,
            max_kernels=max_kernels,
            seed=seed,
        )
        return self.format_to_container(
            values=batch_values,
            timestamps=batch_timestamps,
            history_length=history_length,
            target_length=target_length,
            batch_size=batch_size,
            num_channels=num_channels,
        )
