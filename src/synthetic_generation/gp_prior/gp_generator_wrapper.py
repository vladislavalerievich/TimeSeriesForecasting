from typing import Any, Dict, Optional

import numpy as np

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import GPGeneratorParams
from src.synthetic_generation.gp_prior.gp_generator import GPGenerator


class GPGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: GPGeneratorParams):
        super().__init__(params)
        self.params: GPGeneratorParams = params

    def _sample_parameters(self) -> Dict[str, Any]:
        params = super()._sample_parameters()

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
        length: int,
        seed: Optional[int] = None,
        **params,
    ) -> tuple:
        values = []

        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            generator = GPGenerator(
                GPGeneratorParams(**params),
                length=length,
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
        if seed is not None:
            self._set_random_seeds(seed)
        if params is None:
            params = self._sample_parameters()

        num_channels = params["num_channels"]

        batch_values = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values = self._generate_multivariate_time_series(
                num_channels=num_channels,
                length=params["total_length"],
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
            # Ensure shape for values: (seq_len, num_channels)
            if num_channels == 1:
                values = values.reshape(-1, 1)

            batch_values.append(values)

        batch_values = np.array(batch_values)

        return self._format_to_container(
            values=batch_values,
            start=params["start"],
            history_length=params["history_length"],
            future_length=params["future_length"],
            frequency=params["frequency"],
        )
