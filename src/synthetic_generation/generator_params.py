from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

from src.data_handling.data_containers import Frequency


@dataclass
class GeneratorParams:
    """Base class for generator parameters."""

    global_seed: int = 42
    distribution_type: str = "uniform"
    history_length: Union[int, Tuple[int, int]] = (64, 256)
    target_length: Union[int, Tuple[int, int]] = (32, 256)
    num_channels: Union[int, Tuple[int, int]] = (1, 256)

    def update(self, **kwargs):
        """Update parameters from keyword arguments."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


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


@dataclass
class ForecastPFNGeneratorParams(GeneratorParams):
    """Parameters for the ForecastPFNGenerator."""

    frequency: Frequency = Frequency.D
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
class GPGeneratorParams(GeneratorParams):
    """
    Parameters for the Gaussian Process (GP) Prior synthetic data generator.
    """

    frequency: Frequency = Frequency.D
    max_kernels: int = 6
    likelihood_noise_level: float = 0.4
    noise_level: str = "random"  # Options: ["random", "high", "moderate", "low"]
    use_original_gp: bool = False
    gaussians_periodic: bool = True
    peak_spike_ratio: float = 0.1
    subfreq_ratio: float = 0.2
    periods_per_freq: float = 0.5
    gaussian_sampling_ratio: float = 0.2
    max_period_ratio: float = 1.0
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
