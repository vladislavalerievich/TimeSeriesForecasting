from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import SineWaveGeneratorParams
from src.synthetic_generation.sine_waves.sine_wave_generator import SineWaveGenerator


class SineWaveGeneratorWrapper(GeneratorWrapper):
    """
    Wrapper for SineWaveGenerator to generate batches of multivariate time series data
    by stacking multiple univariate sine wave series. Accepts a SineWaveGeneratorParams
    dataclass for configuration.
    """

    def __init__(self, params: SineWaveGeneratorParams):
        super().__init__(params)
        self.params: SineWaveGeneratorParams = params

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameter values for batch generation with SineWaveGenerator.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values.
        """
        params = super()._sample_parameters()

        # Sample sine wave specific parameters
        period_range = self._sample_range_parameter(self.params.period_range)
        amplitude_range = self._sample_range_parameter(self.params.amplitude_range)
        phase_range = self._sample_range_parameter(self.params.phase_range)
        noise_level = self._sample_scalar_parameter(self.params.noise_level)

        params.update(
            {
                "period_range": period_range,
                "amplitude_range": amplitude_range,
                "phase_range": phase_range,
                "noise_level": noise_level,
            }
        )
        return params

    def _sample_range_parameter(self, param_range):
        """Sample a range parameter that could be a fixed tuple or a tuple of ranges."""
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Check if it's a range of ranges: ((min_min, min_max), (max_min, max_max))
            if isinstance(param_range[0], tuple) and isinstance(param_range[1], tuple):
                min_val = self.rng.uniform(param_range[0][0], param_range[0][1])
                max_val = self.rng.uniform(param_range[1][0], param_range[1][1])
                # Ensure min_val <= max_val
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                return (min_val, max_val)
            else:
                # Fixed range
                return param_range
        else:
            raise ValueError(f"Invalid range parameter format: {param_range}")

    def _sample_scalar_parameter(self, param):
        """Sample a scalar parameter that could be a fixed value or a range."""
        if isinstance(param, (int, float)):
            return param
        elif isinstance(param, tuple) and len(param) == 2:
            return self.rng.uniform(param[0], param[1])
        else:
            raise ValueError(f"Invalid scalar parameter format: {param}")

    def _generate_univariate_time_series(
        self,
        generator: SineWaveGenerator,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate a single univariate time series.

        Parameters
        ----------
        generator : SineWaveGenerator
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
        period_range: tuple,
        amplitude_range: tuple,
        phase_range: tuple,
        noise_level: float,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate a single multivariate time series by stacking multiple univariate sine wave series.

        Parameters
        ----------
        num_channels : int
            Number of channels (univariate series) to generate.
        length : int
            Length of the time series.
        period_range : tuple
            Range for sine wave periods.
        amplitude_range : tuple
            Range for sine wave amplitudes.
        phase_range : tuple
            Range for sine wave phases.
        noise_level : float
            Noise level for the sine waves.
        seed : int, optional
            Random seed for generation (default: None).

        Returns
        -------
        np.ndarray
            Shape: [seq_len, num_channels]
        """
        values = []

        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            generator = SineWaveGenerator(
                length=length,
                period_range=period_range,
                amplitude_range=amplitude_range,
                phase_range=phase_range,
                noise_level=noise_level,
                random_seed=channel_seed,
            )
            result = self._generate_univariate_time_series(generator, channel_seed)

            values.append(result)

        values = np.column_stack(values) if num_channels > 1 else np.array(values[0])
        return values

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Generate a batch of synthetic multivariate time series using SineWaveGenerator.

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
            params = self._sample_parameters()

        history_length = params["history_length"]
        future_length = params["future_length"]
        num_channels = params["num_channels"]
        period_range = params["period_range"]
        amplitude_range = params["amplitude_range"]
        phase_range = params["phase_range"]
        noise_level = params["noise_level"]
        frequency = params["frequency"]

        total_length = history_length + future_length
        batch_values = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values = self._generate_multivariate_time_series(
                num_channels=num_channels,
                length=total_length,
                period_range=period_range,
                amplitude_range=amplitude_range,
                phase_range=phase_range,
                noise_level=noise_level,
                seed=batch_seed,
            )
            # Ensure shape for values: (seq_len, num_channels)
            if num_channels == 1:
                values = values.reshape(-1, 1)
            batch_values.append(values)

        batch_values = np.array(batch_values)

        return self._format_to_container(
            values=batch_values,
            start=params["start"],
            history_length=history_length,
            future_length=future_length,
            frequency=frequency,
        )

    def _format_to_container(
        self,
        values: np.ndarray,
        start: np.ndarray,
        history_length: int,
        future_length: int,
        frequency: Optional[Frequency] = None,
    ) -> BatchTimeSeriesContainer:
        """
        Format the generated time series data into a BatchTimeSeriesContainer.

        Parameters
        ----------
        values: np.ndarray
            Shape: [batch_size, seq_len, num_channels]
        start: np.ndarray of np.datetime64
            Shape: [batch_size]
        history_length: int
        future_length: int
        frequency: Optional[Frequency]
            Frequency of the time series. If None, a random frequency is selected.
        """

        return BatchTimeSeriesContainer(
            history_values=torch.tensor(
                values[:, :history_length, :], dtype=torch.float32
            ),
            future_values=torch.tensor(
                values[:, history_length : history_length + future_length, :],
                dtype=torch.float32,
            ),
            start=start,
            frequency=frequency,
        )
