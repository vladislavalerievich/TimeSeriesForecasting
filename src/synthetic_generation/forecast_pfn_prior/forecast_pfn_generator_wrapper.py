from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.common.utils import generate_spikes
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
                "frequency": self.params.frequency,
                "trend_exp": self.params.trend_exp,
                "scale_noise": self.params.scale_noise,
                "harmonic_scale_ratio": self.params.harmonic_scale_ratio,
                "harmonic_rate": self.params.harmonic_rate,
                "period_factor": self.params.period_factor,
                "seasonal_only": self.params.seasonal_only,
                "trend_additional": self.params.trend_additional,
                "transition_ratio": self.params.transition_ratio,
                "random_walk": self.params.random_walk,
            }
        )
        return params

    def _generate_univariate_time_series(
        self,
        generator: ForecastPFNGenerator,
        seed: Optional[int] = None,
    ) -> Dict:
        return generator.generate_time_series(random_seed=seed)

    def _generate_multivariate_time_series(
        self, num_channels: int, length: int, seed: Optional[int] = None, **params
    ) -> tuple:
        values = []
        start = None
        generator = ForecastPFNGenerator(
            ForecastPFNGeneratorParams(**params),
            length=length,
            random_seed=seed,
        )
        for i in range(num_channels):
            channel_seed = None if seed is None else seed + i
            result = self._generate_univariate_time_series(generator, channel_seed)
            values.append(result["values"])
            if start is None:
                start = result["start"]
                # Update the generator's start time to ensure all channels in a multivariate series have the same start time
                generator_params = generator.params
                generator = ForecastPFNGenerator(generator_params, length=length)
                generator._generate_series(
                    start=pd.Timestamp(start), random_seed=channel_seed
                )
        values = np.column_stack(values) if num_channels > 1 else np.array(values[0])
        return values, start

    def _generate_damping(
        self, input_size: int, p: list = [0.4, 0.5, 0.1]
    ) -> np.ndarray:
        """Generate damping effect for a time series."""
        spacing = self.rng.choice(["equal", "regular", "random"], p=p)
        t = np.arange(0, input_size, 1).astype(float)

        if spacing == "random":
            num_steps = self.rng.integers(1, 3)
            damping_intervals = np.sort(
                self.rng.choice(t[: -int(input_size * 0.1)], num_steps, replace=False)
            )
            damping_factors = self.rng.uniform(0.1, 2, num_steps + 1)
        elif spacing == "equal":
            num_steps = self.rng.integers(3, 7)
            damping_intervals = np.linspace(0, input_size, num_steps + 2)[1:-1]
            damping_factors = np.array(
                [
                    self.rng.uniform(0.4, 0.8)
                    if (i % 2) == 0
                    else self.rng.uniform(1, 2)
                    for i in range(num_steps + 1)
                ]
            )
        else:
            custom_lengths = self.rng.integers(1, input_size // 2, 2)
            damping_intervals = []
            current_time = 0
            while current_time < input_size:
                for length in custom_lengths:
                    current_time += length
                    if current_time <= input_size:
                        damping_intervals.append(current_time)
                    else:
                        break
            damping_intervals = np.array(damping_intervals)
            num_steps = len(damping_intervals)
            damping_factors = np.array(
                [
                    self.rng.uniform(0.4, 0.8)
                    if (i % 2) == 0
                    else self.rng.uniform(1, 2)
                    for i in range(num_steps + 1)
                ]
            )

        damping = np.piecewise(
            t,
            [t < damping_intervals[0]]
            + [
                (t >= damping_intervals[i]) & (t < damping_intervals[i + 1])
                for i in range(num_steps - 1)
            ]
            + [t >= damping_intervals[-1]],
            damping_factors.tolist(),
        )
        return damping

    def _apply_augmentations(
        self, batch_values: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply mixup, damping, and spike augmentations to the batch."""
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

        # Apply damping and spike augmentations if enabled
        if params.get("damp_and_spike", False):
            damping_ratio = self.rng.uniform(0, params.get("damping_noise_ratio", 0.05))
            spike_ratio = self.rng.uniform(0, params.get("spike_noise_ratio", 0.05))
            damping_indices = self.rng.choice(
                batch_size, int(np.ceil(batch_size * damping_ratio)), replace=False
            )
            spike_indices = self.rng.choice(
                batch_size, int(np.ceil(batch_size * spike_ratio)), replace=False
            )

            for idx in damping_indices:
                damping = self._generate_damping(batch_values.shape[1])
                batch_values[idx, :, :] = (
                    batch_values[idx, :, :] * damping[:, np.newaxis]
                )

            for idx in spike_indices:
                spikes = generate_spikes(batch_values.shape[1])
                if spikes.max() < 0:
                    batch_values[idx, :, :] = (
                        batch_values[idx, :, :] * spikes[:, np.newaxis]
                    )
                else:
                    batch_values[idx, :, :] = (
                        batch_values[idx, :, :] + spikes[:, np.newaxis] + 1
                    )

            if self.rng.random() < params.get("spike_signal_ratio", 0.05):
                spikey_series_ratio = self.rng.uniform(
                    0, params.get("spike_batch_ratio", 0.05)
                )
                spike_replace_indices = self.rng.choice(
                    batch_size,
                    int(np.ceil(batch_size * spikey_series_ratio)),
                    replace=False,
                )
                for idx in spike_replace_indices:
                    spikes = generate_spikes(batch_values.shape[1])
                    batch_values[idx, :, :] = spikes[:, np.newaxis]

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
        history_length = params["history_length"]
        future_length = params["future_length"]
        num_channels = params["num_channels"]

        total_length = history_length + future_length
        batch_values = []
        batch_start = []
        for i in range(batch_size):
            batch_seed = None if seed is None else seed + i * num_channels
            values, start = self._generate_multivariate_time_series(
                num_channels=num_channels,
                length=total_length,
                seed=batch_seed,
                frequency=params["frequency"],
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
            batch_start.append(start)
        batch_values = np.array(batch_values)

        # Apply augmentations if parameters are provided
        batch_values = self._apply_augmentations(
            batch_values,
            {
                "mixup_prob": self.params.mixup_prob,
                "mixup_series": self.params.mixup_series,
                "damp_and_spike": self.params.damp_and_spike,
                "damping_noise_ratio": self.params.damping_noise_ratio,
                "spike_noise_ratio": self.params.spike_noise_ratio,
                "spike_signal_ratio": self.params.spike_signal_ratio,
                "spike_batch_ratio": self.params.spike_batch_ratio,
            },
        )

        return self._format_to_container(
            values=batch_values,
            start=np.array(batch_start),
            history_length=history_length,
            future_length=future_length,
            frequency=params["frequency"],
        )
