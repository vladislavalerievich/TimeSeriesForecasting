import functools
from typing import Dict, Optional

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from src.data_handling.data_containers import Frequency
from src.synthetic_generation.constants import BASE_END_ORD, BASE_START_ORD
from src.synthetic_generation.forecast_pfn_prior.generate_steps_n_spikes import (
    generate_peak_spikes,
)
from src.synthetic_generation.forecast_pfn_prior.utils import (
    create_kernel,
    extract_periodicities,
    random_binary_map,
)
from src.synthetic_generation.generator_params import GPGeneratorParams


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


KERNEL_BANK = {
    0: ("matern_kernel", 3),
    1: ("linear_kernel", 2),
    2: ("rbf_kernel", 2),
    3: ("periodic_kernel", 5),
    4: ("polynomial_kernel", 1),
    5: ("rational_quadratic_kernel", 1),
    6: ("spectral_mixture_kernel", 2),
}

FREQUENCY_MAPPING = {
    Frequency.T5: ("min", "5"),
    Frequency.T10: ("min", "10"),
    Frequency.T15: ("min", "15"),
    Frequency.H: ("H", ""),
    Frequency.D: ("D", ""),
    Frequency.W: ("W", ""),
    Frequency.M: ("MS", ""),
}

KERNEL_PERIODS_BY_FREQ = {
    "min": [5, 15, 30, 60, 120, 240, 360],
    "H": [3, 6, 12, 24, 48, 72, 168],
    "D": [7, 14, 28, 30, 90, 180, 365],
    "W": [2, 4, 8, 12, 24, 52],
    "MS": [3, 4, 6, 12, 24, 36, 60],
}


class GPGenerator:
    def __init__(
        self,
        params: GPGeneratorParams,
        length: int = 1024,
    ):
        self.params = params
        self.length = length
        self.frequency = params.frequency
        self.max_kernels = params.max_kernels
        self.likelihood_noise_level = params.likelihood_noise_level
        self.noise_level = params.noise_level
        self.use_original_gp = params.use_original_gp
        self.gaussians_periodic = params.gaussians_periodic
        self.peak_spike_ratio = params.peak_spike_ratio
        self.subfreq_ratio = params.subfreq_ratio
        self.periods_per_freq = params.periods_per_freq
        self.gaussian_sampling_ratio = params.gaussian_sampling_ratio
        self.kernel_periods = params.kernel_periods
        self.max_period_ratio = params.max_period_ratio
        self.kernel_bank = params.kernel_bank

    def generate_time_series(
        self,
        random_seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Determine kernel_bank and gaussians_periodic
        if self.use_original_gp:
            kernel_bank = KERNEL_BANK
            gaussians_periodic = False
        else:
            # Convert kernel_bank from {str: float} format to {int: (str, float)} format
            kernel_bank = {
                i: (kernel_name, weight)
                for i, (kernel_name, weight) in enumerate(self.kernel_bank.items())
            }
            gaussians_periodic = self.gaussians_periodic

        # Map frequency to freq and subfreq
        freq, subfreq = FREQUENCY_MAPPING.get(self.frequency, ("D", ""))

        # Decide if using exact frequencies
        exact_freqs = np.random.rand() < self.periods_per_freq
        if exact_freqs and freq in KERNEL_PERIODS_BY_FREQ:
            kernel_periods = KERNEL_PERIODS_BY_FREQ[freq]
            if subfreq:
                subfreq_int = int(subfreq)
                kernel_periods = [
                    p // subfreq_int for p in kernel_periods if p >= subfreq_int
                ]
        else:
            kernel_periods = self.kernel_periods

        # Sample number of kernels
        num_kernels = np.random.randint(1, self.max_kernels + 1)
        # Always expect kernel_bank as dict {int: (str, float)}
        kernel_weights = np.array([v[1] for v in kernel_bank.values()])
        kernel_ids = np.random.choice(
            list(kernel_bank.keys()),
            size=num_kernels,
            p=kernel_weights / kernel_weights.sum(),
        )
        kernel_names = [kernel_bank[i][0] for i in kernel_ids]

        # Create composite kernel
        composite_kernel = functools.reduce(
            lambda a, b: random_binary_map(a, b),
            [
                create_kernel(
                    k,
                    self.length,
                    int(self.max_period_ratio * self.length),
                    gaussians_periodic,
                    kernel_periods,
                )
                for k in kernel_names
            ],
        )

        # Set up GP model
        train_x = torch.linspace(0, 1, self.length)
        trend = np.random.choice([True, False])
        mean_module = (
            gpytorch.means.LinearMean(input_size=1)
            if trend
            else gpytorch.means.ConstantMean()
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_covar=torch.diag(
                torch.full_like(train_x, self.likelihood_noise_level**2)
            )
        )
        model = GPModel(train_x, None, likelihood, mean_module, composite_kernel)

        # Determine noise level
        noise = {"high": 1e-1, "moderate": 1e-2, "low": 1e-3}.get(
            self.noise_level,
            np.random.choice([1e-1, 1e-2, 1e-3], p=[0.1, 0.2, 0.7]),
        )

        # Sample from GP prior
        model.eval()
        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.cholesky_jitter(noise),
        ):
            y_sample = model(train_x).sample().numpy()
            # y_sample shape: (self.length,) (should be 1D)

        # Optionally add peak spikes
        if np.random.rand() < self.peak_spike_ratio:
            periodicities = extract_periodicities(composite_kernel, self.length)
            if len(periodicities) > 0:
                p = int(np.round(max(periodicities)))
                spikes_type = np.random.choice(["regular", "patchy"], p=[0.3, 0.7])
                spikes = generate_peak_spikes(self.length, p, spikes_type=spikes_type)
                # y_sample is 1D, so use y_sample[:p].argmax()
                spikes_shift = (
                    p - y_sample[:p].argmax() if p > 0 and p <= len(y_sample) else 0
                )
                spikes = np.roll(spikes, -spikes_shift)
                if spikes.max() < 0:
                    y_sample = y_sample + spikes + 1
                else:
                    y_sample = y_sample * spikes

        # Generate start time
        start = np.datetime64(
            pd.Timestamp.fromordinal(
                int((BASE_START_ORD - BASE_END_ORD) * beta.rvs(5, 1) + BASE_START_ORD)
            )
        )

        return {"start": start, "values": y_sample}
