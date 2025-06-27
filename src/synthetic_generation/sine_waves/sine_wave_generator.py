from typing import Dict, Optional

import numpy as np

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator


class SineWaveGenerator(AbstractTimeSeriesGenerator):
    """
    Generate synthetic univariate time series using sine waves with configurable parameters.

    Each series is a sine wave with random amplitude, period, phase, and optional noise.
    """

    def __init__(
        self,
        length: int = 1024,
        period_range: tuple = (10, 100),
        amplitude_range: tuple = (0.5, 3.0),
        phase_range: tuple = (0, 2 * np.pi),
        noise_level: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        length : int, optional
            Number of time steps per series (default: 1024).
        period_range : tuple, optional
            (min_period, max_period) for sine wave period (default: (10, 100)).
        amplitude_range : tuple, optional
            (min_amplitude, max_amplitude) for sine wave amplitude (default: (0.5, 3.0)).
        phase_range : tuple, optional
            (min_phase, max_phase) for sine wave phase (default: (0, 2*pi)).
        noise_level : float, optional
            Noise level as a fraction of amplitude (default: 0.1).
        random_seed : int, optional
            Seed for the random number generator.
        """
        self.length = length
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.noise_level = noise_level
        self.rng = np.random.default_rng(random_seed)

    def generate_time_series(
        self, random_seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate a single univariate sine wave time series.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible generation.


        Returns
        -------
        np.ndarray
            Shape: [seq_len]
        """
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        # Sample sine wave parameters
        period = self.rng.uniform(self.period_range[0], self.period_range[1])
        amplitude = self.rng.uniform(self.amplitude_range[0], self.amplitude_range[1])
        phase = self.rng.uniform(self.phase_range[0], self.phase_range[1])

        # Generate time indices
        time_idx = np.arange(self.length)

        # Generate sine wave
        values = amplitude * np.sin(2 * np.pi * time_idx / period + phase)

        # Add noise if specified
        if self.noise_level > 0:
            noise = self.rng.normal(0, amplitude * self.noise_level, size=self.length)
            values += noise

        return values
