import functools
from typing import Dict, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import (
    ConstantKernel as GpytorchConstantKernel,
)
from gpytorch.kernels import (
    PeriodicKernel,
    PolynomialKernel,
    RBFKernel,
    RQKernel,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.gaussian_process.kernels import (
    ConstantKernel as SklearnConstantKernel,
)

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.common.constants import DEFAULT_START_DATE


class KernelSynthGenerator(AbstractTimeSeriesGenerator):
    """
    Generate independent synthetic univariate time series using kernel synthesis.

    Each series is sampled from a Gaussian process prior with a random composite kernel.
    """

    def __init__(
        self,
        length: int = 1024,
        max_kernels: int = 5,
        random_seed: Optional[int] = None,
        use_gpytorch: bool = False,
    ):
        """
        Parameters
        ----------
        length : int, optional
            Number of time steps per series (default: 1024).
        max_kernels : int, optional
            Maximum number of base kernels to combine (default: 5).
        random_seed : int, optional
            Seed for the random number generator.
        """
        self.length = length
        self.max_kernels = max_kernels
        self.rng = np.random.default_rng(random_seed)
        self.use_gpytorch = use_gpytorch
        self.kernel_bank = self._initialize_kernel_bank()

    def _initialize_kernel_bank(self):
        if self.use_gpytorch:
            return [
                # Periodic Kernels (equivalent to ExpSineSquared)
                PeriodicKernel(period_length=24 / self.length),
                PeriodicKernel(period_length=48 / self.length),
                PeriodicKernel(period_length=96 / self.length),
                PeriodicKernel(period_length=(24 * 7) / self.length),
                PeriodicKernel(period_length=(48 * 7) / self.length),
                PeriodicKernel(period_length=(96 * 7) / self.length),
                PeriodicKernel(period_length=7 / self.length),
                PeriodicKernel(period_length=14 / self.length),
                PeriodicKernel(period_length=30 / self.length),
                PeriodicKernel(period_length=60 / self.length),
                PeriodicKernel(period_length=365 / self.length),
                PeriodicKernel(period_length=(365 * 2) / self.length),
                PeriodicKernel(period_length=4 / self.length),
                PeriodicKernel(period_length=26 / self.length),
                PeriodicKernel(period_length=52 / self.length),
                PeriodicKernel(period_length=4 / self.length),
                PeriodicKernel(period_length=6 / self.length),
                PeriodicKernel(period_length=12 / self.length),
                PeriodicKernel(period_length=4 / self.length),
                PeriodicKernel(period_length=(4 * 10) / self.length),
                PeriodicKernel(period_length=10 / self.length),
                # Polynomial Kernels (equivalent to DotProduct)
                PolynomialKernel(power=1, offset=0.0),
                PolynomialKernel(power=1, offset=1.0),
                PolynomialKernel(power=1, offset=100.0),
                # RBF Kernels
                RBFKernel(lengthscale=0.1),
                RBFKernel(lengthscale=1.0),
                RBFKernel(lengthscale=10.0),
                # Rational Quadratic Kernels
                RQKernel(alpha=0.1),
                RQKernel(alpha=1.0),
                RQKernel(alpha=10.0),
                # Constant Kernel
                GpytorchConstantKernel(),
            ]
        else:
            return [
                ExpSineSquared(periodicity=24 / self.length),  # H
                ExpSineSquared(periodicity=48 / self.length),  # 0.5H
                ExpSineSquared(periodicity=96 / self.length),  # 0.25H
                ExpSineSquared(periodicity=24 * 7 / self.length),  # H-week
                ExpSineSquared(periodicity=48 * 7 / self.length),  # 0.5H-week
                ExpSineSquared(periodicity=96 * 7 / self.length),  # 0.25H-week
                ExpSineSquared(periodicity=7 / self.length),  # day
                ExpSineSquared(periodicity=14 / self.length),  # 0.5-day
                ExpSineSquared(periodicity=30 / self.length),  # day
                ExpSineSquared(periodicity=60 / self.length),  # 0.5-day
                ExpSineSquared(periodicity=365 / self.length),  # year
                ExpSineSquared(periodicity=365 * 2 / self.length),  # 0.5-year
                ExpSineSquared(periodicity=4 / self.length),  # week
                ExpSineSquared(periodicity=26 / self.length),  # week
                ExpSineSquared(periodicity=52 / self.length),  # week
                ExpSineSquared(periodicity=4 / self.length),  # month
                ExpSineSquared(periodicity=6 / self.length),  # month
                ExpSineSquared(periodicity=12 / self.length),  # month
                ExpSineSquared(periodicity=4 / self.length),  # quarter
                ExpSineSquared(periodicity=4 * 10 / self.length),  # quarter
                ExpSineSquared(periodicity=10 / self.length),  # year
                DotProduct(sigma_0=0.0),
                DotProduct(sigma_0=1.0),
                DotProduct(sigma_0=10.0),
                RBF(length_scale=0.1),
                RBF(length_scale=1.0),
                RBF(length_scale=10.0),
                RationalQuadratic(alpha=0.1),
                RationalQuadratic(alpha=1.0),
                RationalQuadratic(alpha=10.0),
                WhiteKernel(noise_level=0.1),
                WhiteKernel(noise_level=1.0),
                SklearnConstantKernel(),
            ]

    def _random_binary_map(self, a, b):
        """
        Randomly combine two kernels with + or *.
        """
        ops = [lambda x, y: x + y, lambda x, y: x * y]
        return self.rng.choice(ops)(a, b)

    def _sample_from_gp_prior(
        self,
        kernel,
        X: np.ndarray,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw a sample from GP prior using the selected backend.
        """
        if self.use_gpytorch:
            import ipdb; ipdb.set_trace()  # noqa: E702
            return self._sample_from_gpytorch(kernel, X, random_seed)
        else:
            return self._sample_from_sklearn(kernel, X, random_seed)

    def _sample_from_gpytorch(
        self,
        kernel,
        X: np.ndarray,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        if random_seed is not None:
            torch.manual_seed(random_seed)

        X_tensor = torch.from_numpy(X).float().squeeze(-1)
        mean = torch.zeros(X_tensor.shape[0])
        cov = kernel(X_tensor).evaluate()
        # Add a small jitter (noise) for numerical stability
        cov = cov + torch.eye(cov.shape[0]) * 1e-4

        dist = gpytorch.distributions.MultivariateNormal(mean, cov)
        ts = dist.sample()

        return ts.numpy()

    def _sample_from_sklearn(
        self,
        kernel,
        X: np.ndarray,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw a sample from GP prior using GaussianProcessRegressor.
        """
        if X.ndim == 1:
            X = X[:, None]
        gpr = GaussianProcessRegressor(kernel=kernel)
        ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

        return ts.squeeze()

    def generate_time_series(self, random_seed: Optional[int] = None) -> Dict:
        """
        Generate a single independent univariate time series.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible generation.

        Returns
        -------
        dict
            { 'start': np.datetime64, 'values': np.ndarray }
        """
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        X = np.linspace(0, 1, self.length).reshape(-1, 1)

        num_kernels = self.rng.integers(1, self.max_kernels + 1)
        selected = self.rng.choice(self.kernel_bank, num_kernels, replace=True)
        composite = functools.reduce(self._random_binary_map, selected)

        try:
            ts = self._sample_from_gp_prior(composite, X, random_seed=random_seed)
        except (np.linalg.LinAlgError, torch.linalg.LinAlgError) as e:
            import ipdb; ipdb.set_trace()
            new_seed = (random_seed + 1) if random_seed is not None else None
            return self.generate_time_series(new_seed)

        # Create timestamps using the frequency parameter
        start_time = np.datetime64(DEFAULT_START_DATE, "s")

        return {"start": start_time, "values": ts}
