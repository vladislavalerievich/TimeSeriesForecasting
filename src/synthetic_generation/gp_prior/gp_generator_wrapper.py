from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import GPGeneratorParams
from src.synthetic_generation.gp_prior.gp_generator import GPGenerator


class GPGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: GPGeneratorParams):
        super().__init__(params)
        self.params: GPGeneratorParams = params

    def _sample_parameters(self) -> Dict[str, Any]:
        params = super()._sample_parameters()

        # Sample frequency randomly from the frequency enum
        frequency = np.random.choice(list(Frequency))

        params.update(
            {
                "max_kernels": self.params.max_kernels,
                "likelihood_noise_level": self.params.likelihood_noise_level,
                "noise_level": self.params.noise_level,
                "use_original_gp": self.params.use_original_gp,
                "gaussians_periodic": self.params.gaussians_periodic,
                "peak_spike_ratio": self.params.peak_spike_ratio,
                "subfreq_ratio": self.params.subfreq_ratio,
                "periods_per_freq": self.params.periods_per_freq,
                "gaussian_sampling_ratio": self.params.gaussian_sampling_ratio,
                "kernel_periods": self.params.kernel_periods,
                "max_period_ratio": self.params.max_period_ratio,
                "kernel_bank": self.params.kernel_bank,
                "frequency": frequency,
            }
        )
        return params

    def _generate_univariate_time_series(
        self,
        generator: GPGenerator,
        seed: Optional[int] = None,
    ) -> Dict:
        return generator.generate_time_series(random_seed=seed)

    def _generate_multivariate_time_series(
        self,
        num_channels: int,
        history_length: int,
        target_length: int,
        seed: Optional[int] = None,
        **params,
    ) -> tuple:
        history_values = []
        target_values = None
        start = None
        target_index = np.random.randint(0, num_channels) if num_channels > 1 else 0
        total_length = history_length + target_length

        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            generator = GPGenerator(
                GPGeneratorParams(**params),
                length=total_length if i == target_index else history_length,
            )
            result = self._generate_univariate_time_series(generator, channel_seed)
            if i == target_index:
                target_values = result["values"][-target_length:]
                history_values.append(result["values"][:history_length])
                if start is None:
                    start = result["start"]
            else:
                history_values.append(result["values"])

        history_values = (
            np.column_stack(history_values)
            if num_channels > 1
            else np.array(history_values[0])
        )
        target_values = target_values if num_channels > 1 else target_values
        return history_values, target_values, start, target_index

    def _format_to_container(
        self,
        history_values: np.ndarray,
        target_values: np.ndarray,
        target_index: np.ndarray,
        start: np.ndarray,
        frequency: Frequency,
    ) -> BatchTimeSeriesContainer:
        """
        Format the generated time series data into a BatchTimeSeriesContainer.

        Parameters
        ----------
        history_values: np.ndarray
            Shape: [batch_size, history_length, num_channels]
        target_values: np.ndarray
            Shape: [batch_size, target_length] for multivariate, [batch_size, target_length, 1] for univariate
        target_index: np.ndarray
            Shape: [batch_size], index of the target channel for multivariate data
        start: np.ndarray of np.datetime64
            Shape: [batch_size]
        frequency: Frequency
            Frequency of the time series.
        """
        history_values = np.array(history_values)
        target_values = np.array(target_values)

        history_values_tensor = torch.tensor(history_values, dtype=torch.float32)
        target_values_tensor = torch.tensor(target_values, dtype=torch.float32)
        target_index_tensor = torch.tensor(target_index, dtype=torch.long)

        return BatchTimeSeriesContainer(
            history_values=history_values_tensor,
            target_values=target_values_tensor,
            target_index=target_index_tensor,
            start=start,
            frequency=frequency,
        )

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> BatchTimeSeriesContainer:
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters()
        history_length = params["history_length"]
        target_length = params["target_length"]
        num_channels = params["num_channels"]
        total_length = history_length + target_length
        batch_history_values = []
        batch_target_values = []
        batch_start = []
        batch_target_index = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            if num_channels == 1:
                generator = GPGenerator(
                    GPGeneratorParams(**params), length=total_length
                )
                result = self._generate_univariate_time_series(generator, batch_seed)
                history_values = result["values"][:history_length]
                target_values = result["values"][-target_length:]
                start = result["start"]
                target_index = 0
            else:
                history_values, target_values, start, target_index = (
                    self._generate_multivariate_time_series(
                        num_channels=num_channels,
                        history_length=history_length,
                        target_length=target_length,
                        seed=batch_seed,
                        frequency=params["frequency"],
                        max_kernels=params["max_kernels"],
                        likelihood_noise_level=params["likelihood_noise_level"],
                        noise_level=params["noise_level"],
                        use_original_gp=params["use_original_gp"],
                        gaussians_periodic=params["gaussians_periodic"],
                        peak_spike_ratio=params["peak_spike_ratio"],
                        subfreq_ratio=params["subfreq_ratio"],
                        periods_per_freq=params["periods_per_freq"],
                        gaussian_sampling_ratio=params["gaussian_sampling_ratio"],
                        kernel_periods=params["kernel_periods"],
                        max_period_ratio=params["max_period_ratio"],
                        kernel_bank=params["kernel_bank"],
                    )
                )
            # Ensure shape for history_values: (seq_len, num_channels)
            if num_channels == 1:
                history_values = history_values.reshape(-1, 1)
            batch_history_values.append(history_values)
            batch_target_values.append(target_values)
            batch_start.append(start)
            batch_target_index.append(target_index)

        return self._format_to_container(
            history_values=batch_history_values,
            target_values=batch_target_values,
            target_index=np.array(batch_target_index),
            start=np.array(batch_start),
            frequency=params["frequency"],
        )
