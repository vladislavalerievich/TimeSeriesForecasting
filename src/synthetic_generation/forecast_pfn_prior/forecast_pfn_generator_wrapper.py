from typing import Any, Dict, Optional

import numpy as np

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator import (
    ForecastPFNGenerator,
)
from src.synthetic_generation.generator_params import ForecastPFNGeneratorParams


class ForecastPFNGeneratorWrapper(GeneratorWrapper):
    def __init__(self, params: ForecastPFNGeneratorParams):
        super().__init__(params)
        self.params: ForecastPFNGeneratorParams = params

    def _sample_parameters(self) -> Dict[str, Any]:
        params = super()._sample_parameters()

        params.update(
            {
                "trend_exp": self.params.trend_exp,
                "scale_noise": self.params.scale_noise,
                "harmonic_scale_ratio": self.params.harmonic_scale_ratio,
                "harmonic_rate": self.params.harmonic_rate,
                "period_factor": self.params.period_factor,
                "seasonal_only": self.params.seasonal_only,
                "trend_additional": self.params.trend_additional,
                "transition_ratio": self.params.transition_ratio,
                "random_walk": self.params.random_walk,
                # Univariate augmentation parameters
                "time_warp_prob": self.params.time_warp_prob,
                "time_warp_strength": self.params.time_warp_strength,
                "magnitude_scale_prob": self.params.magnitude_scale_prob,
                "magnitude_scale_range": self.params.magnitude_scale_range,
                "damping_prob": self.params.damping_prob,
                "spike_prob": self.params.spike_prob,
                "pure_spike_prob": self.params.pure_spike_prob,
            }
        )
        return params

    def _generate_univariate_time_series(
        self,
        generator: ForecastPFNGenerator,
        start: Optional[np.datetime64] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        return generator.generate_time_series(start=start, random_seed=seed)

    def _generate_multivariate_time_series(
        self, num_channels: int, length: int, seed: Optional[int] = None, **params
    ) -> np.ndarray:
        values = []
        start_date = params.get("start")

        generator = ForecastPFNGenerator(
            ForecastPFNGeneratorParams(**params),
            length=length,
            random_seed=seed,
        )
        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            result = self._generate_univariate_time_series(
                generator, start_date, channel_seed
            )
            # Extract the actual values from the result dictionary
            values.append(result["values"])

        values = np.column_stack(values) if num_channels > 1 else np.array(values[0])
        return values

    def _apply_augmentations(
        self, batch_values: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply multivariate augmentations to the batch."""
        batch_size = batch_values.shape[0]

        # Apply mixup augmentation if enabled
        if self.rng.random() < params.get("mixup_prob", 0.0):
            mixup_series = self.rng.integers(2, params.get("mixup_series", 4) + 1)
            mixup_indices = self.rng.choice(batch_size, mixup_series, replace=False)
            original_vals = batch_values[mixup_indices, :, :].copy()
            for i, idx in enumerate(mixup_indices):
                mixup_weights = self.rng.random(mixup_series)
                mixup_weights /= np.sum(mixup_weights)
                batch_values[idx, :, :] = np.sum(
                    original_vals * mixup_weights[:, np.newaxis, np.newaxis], axis=0
                )

        return batch_values

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
        frequency = params["frequency"]
        start_date = params["start"]

        batch_values = []

        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values = self._generate_multivariate_time_series(
                num_channels=num_channels,
                length=params["total_length"],
                seed=batch_seed,
                start=start_date,
                frequency=frequency,
                trend_exp=params["trend_exp"],
                scale_noise=params["scale_noise"],
                harmonic_scale_ratio=params["harmonic_scale_ratio"],
                harmonic_rate=params["harmonic_rate"],
                period_factor=params["period_factor"],
                seasonal_only=params["seasonal_only"],
                trend_additional=params["trend_additional"],
                transition_ratio=params["transition_ratio"],
                random_walk=params["random_walk"],
            )
            if num_channels == 1:
                values = values.reshape(-1, 1)
            batch_values.append(values)
        batch_values = np.array(batch_values)

        # Apply multivariate augmentations
        batch_values = self._apply_augmentations(
            batch_values,
            {
                "mixup_prob": self.params.mixup_prob,
                "mixup_series": self.params.mixup_series,
            },
        )

        return self._format_to_container(
            values=batch_values,
            start=start_date,
            history_length=params["history_length"],
            future_length=params["future_length"],
            frequency=frequency,
        )
