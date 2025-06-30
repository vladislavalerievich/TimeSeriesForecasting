from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.data_handling.data_containers import Frequency


@dataclass
class GeneratorParams:
    """Base class for generator parameters."""

    global_seed: int = 42
    distribution_type: str = "uniform"
    total_length: int = 2048
    future_length: Union[int, Tuple[int, int], List[int]] = field(
        default_factory=lambda: (48, 900)
    )
    num_channels: Union[int, Tuple[int, int], List[int]] = field(
        default_factory=lambda: 1  # TODO: revert to (1, 21)
    )
    frequency: Frequency = Frequency.D
    start: Optional[np.datetime64] = None  # If None, will be auto-selected safely

    def update(self, **kwargs):
        """Update parameters from keyword arguments."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __post_init__(self):
        """Validate that future_length doesn't exceed total_length."""
        # Normalize to tuples for consistent comparison
        fut_min, fut_max = self._normalize_range(self.future_length)

        # Ensure that future_length doesn't exceed total_length
        if fut_max > self.total_length:
            raise ValueError(
                f"Maximum future_length ({fut_max}) must be <= total_length ({self.total_length})"
            )

        if fut_min < 1:
            raise ValueError(f"Minimum future_length ({fut_min}) must be >= 1")

    def _normalize_range(
        self, value: Union[int, Tuple[int, int], List[int]]
    ) -> Tuple[int, int]:
        """Convert int or tuple to (min, max) tuple."""
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, list):
            return (min(value), max(value))
        return value

    def get_compatible_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Get compatible min/max ranges for all parameters."""
        fut_min, fut_max = self._normalize_range(self.future_length)
        return {
            "total_length": (self.total_length, self.total_length),
            "history_length": (
                self.total_length - fut_max,
                self.total_length - fut_min,
            ),
            "future_length": (fut_min, fut_max),
            "num_channels": self._normalize_range(self.num_channels),
        }


@dataclass
class LMCGeneratorParams(GeneratorParams):
    """Parameters for the LMCGenerator."""

    max_kernels: int = 5
    dirichlet_min: float = 0.1
    dirichlet_max: float = 2.0
    scale: float = 1.0
    weibull_shape: float = 2.0
    weibull_scale: int = 1


@dataclass
class KernelGeneratorParams(GeneratorParams):
    """Parameters for the KernelSynthGenerator."""

    num_kernels: Union[int, Tuple[int, int]] = 5
    use_gpytorch: bool = False


@dataclass
class GPGeneratorParams(GeneratorParams):
    """
    Parameters for the Gaussian Process (GP) Prior synthetic data generator.
    """

    max_kernels: int = 6
    likelihood_noise_level: float = 0.1
    noise_level: str = "low"  # Options: ["random", "high", "moderate", "low"]
    use_original_gp: bool = False
    gaussians_periodic: bool = True
    peak_spike_ratio: float = 0.1
    subfreq_ratio: float = 0.2
    periods_per_freq: float = 0.5
    gaussian_sampling_ratio: float = 0.2
    max_period_ratio: float = 0.5
    kernel_periods: Tuple[int, ...] = (4, 5, 7, 21, 24, 30, 60, 120)
    kernel_bank: Dict[str, float] = field(
        default_factory=lambda: {
            "matern_kernel": 1.5,
            "linear_kernel": 1.0,
            "periodic_kernel": 5.0,
            "polynomial_kernel": 0.0,
            "spectral_mixture_kernel": 0.0,
        }
    )


@dataclass
class ForecastPFNGeneratorParams(GeneratorParams):
    """Parameters for the ForecastPFNGenerator."""

    trend_exp: bool = True
    scale_noise: Tuple[float, float] = (0.6, 0.3)
    harmonic_scale_ratio: float = 0.5
    harmonic_rate: float = 1.0
    period_factor: float = 1.0
    seasonal_only: bool = False
    trend_additional: bool = False
    transition_ratio: float = (
        1.0  # Probability of applying transition between two series
    )
    random_walk: bool = False
    # Parameters for data augmentation
    mixup_prob: float = 0.0  # Probability of applying mixup augmentation
    mixup_series: int = 4  # Maximum number of series to mix in mixup
    damp_and_spike: bool = False  # Whether to apply damping and spike augmentations
    damping_noise_ratio: float = 0.05  # Probability of applying damping noise
    spike_noise_ratio: float = 0.05  # Probability of applying spike noise
    spike_signal_ratio: float = (
        0.05  # Probability of replacing series with spike-only signal
    )
    spike_batch_ratio: float = 0.05  # Fraction of batch to replace with spike-only signals if spike_signal_ratio is triggered


@dataclass
class SineWaveGeneratorParams(GeneratorParams):
    """Parameters for the SineWaveGenerator."""

    period_range: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (10.0, 100.0)
    amplitude_range: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (0.5, 3.0)
    phase_range: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (0.0, 2.0 * np.pi)
    noise_level: Union[float, Tuple[float, float]] = 0.0
