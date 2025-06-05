import functools
from typing import Dict, Optional

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)

from src.synthetic_generation.abstract_classes import AbstractTimeSeriesGenerator
from src.synthetic_generation.constants import DEFAULT_START_DATE


class LMCSynthGenerator(AbstractTimeSeriesGenerator):
    """Generate synthetic multivariate time series data using Latent Multi-Channel Synthesis."""

    def __init__(
        self,
        length: int = 1024,
        max_kernels: int = 5,
        num_channels: int = 10,
        dirichlet_min: float = 0.1,
        dirichlet_max: float = 2.0,
        scale: float = 1.0,
        weibull_shape: float = 2.0,
        weibull_scale: int = 1,
    ):
        """
        Initialize the LMC Synthetic Data Generator.

        Parameters
        ----------
        length : int, optional
            Length of each time series (default: 1024).
        max_kernels : int, optional
            Maximum number of kernels per latent function (default: 5).
        num_channels : int, optional
            Number of channels in the multivariate time series (default: 10).
        dirichlet_min : float, optional
            Minimum value for Dirichlet parameter (default: 0.1).
        dirichlet_max : float, optional
            Maximum value for Dirichlet parameter (default: 2.0).
        scale : float, optional
            Scaling factor for Weibull distribution (default: 1.0).
        weibull_shape : float, optional
            Shape parameter for Weibull distribution (default: 2.0).
        weibull_scale : int, optional
            Scale parameter for Weibull distribution (default: 1).
        """
        self.length = length
        self.max_kernels = max_kernels
        self.num_channels = num_channels
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.scale = scale
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale

        # Initialize kernel bank with adjusted periodicities based on length.
        self.kernel_bank = [
            ExpSineSquared(periodicity=24 / length),  # H
            ExpSineSquared(periodicity=48 / length),  # 0.5H
            ExpSineSquared(periodicity=96 / length),  # 0.25H
            ExpSineSquared(periodicity=24 * 7 / length),  # H
            ExpSineSquared(periodicity=48 * 7 / length),  # 0.5H
            ExpSineSquared(periodicity=96 * 7 / length),  # 0.25H
            ExpSineSquared(periodicity=7 / length),  # D
            ExpSineSquared(periodicity=14 / length),  # 0.5D
            ExpSineSquared(periodicity=30 / length),  # D
            ExpSineSquared(periodicity=60 / length),  # 0.5D
            ExpSineSquared(periodicity=365 / length),  # D
            ExpSineSquared(periodicity=365 * 2 / length),  # 0.5D
            ExpSineSquared(periodicity=4 / length),  # W
            ExpSineSquared(periodicity=26 / length),  # W
            ExpSineSquared(periodicity=52 / length),  # W
            ExpSineSquared(periodicity=4 / length),  # M
            ExpSineSquared(periodicity=6 / length),  # M
            ExpSineSquared(periodicity=12 / length),  # M
            ExpSineSquared(periodicity=4 / length),  # Q
            ExpSineSquared(periodicity=4 * 10 / length),  # Q
            ExpSineSquared(periodicity=10 / length),  # Y
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
            ConstantKernel(),
        ]

    def _random_binary_map(self, a: Kernel, b: Kernel) -> Kernel:
        """
        Applies a random binary operator (+ or *) with equal probability
        on kernels ``a`` and ``b``.

        Parameters
        ----------
        a : Kernel
            A GP kernel
        b : Kernel
            A GP kernel

        Returns
        -------
        Kernel
            The composite kernel `a + b` or `a * b`
        """
        binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
        return np.random.choice(binary_maps)(a, b)

    @staticmethod
    def _sample_from_gp_prior(
        kernel: Kernel,
        X: np.ndarray,
        random_seed: Optional[int] = 42,
        method: str = "eigh",
    ) -> np.ndarray:
        """
        Draw a sample from a GP prior using efficient sampling method.

        Parameters
        ----------
        kernel : Kernel
            The GP covariance kernel
        X : np.ndarray
            The input "time" points
        random_seed : int, optional
            Random seed for sampling
        method : str, optional
            Sampling method for multivariate_normal

        Returns
        -------
        np.ndarray
            A time series sampled from the GP prior
        """
        if X.ndim == 1:
            X = X[:, None]

        assert X.ndim == 2

        cov = kernel(X)
        ts = np.random.default_rng(seed=random_seed).multivariate_normal(
            mean=np.zeros(X.shape[0]), cov=cov, method=method
        )

        return ts

    def generate_time_series(self, random_seed: Optional[int] = 42) -> Dict:
        """
        Generate a single multivariate synthetic time series.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility (default: None).

        Returns
        -------
        dict
            Dictionary containing:
            - 'start': Start timestamp (np.datetime64)
            - 'values': Generated time series (np.ndarray of shape (num_channels, length))
        """
        np.random.seed(random_seed)

        while True:
            X = np.linspace(0, 1, self.length)

            # Sample number of latent functions from Weibull distribution
            latent_num = np.rint(
                np.random.weibull(self.weibull_shape, size=self.length) * self.scale + 1
            )
            latent_num = np.clip(
                latent_num, max(2, self.num_channels // 20), self.num_channels
            )
            latent_num = int(latent_num[0])

            # Sample number of kernels for each latent function
            kernel_numbers = np.random.randint(1, self.max_kernels + 1, size=latent_num)

            # Sample kernels for each latent function
            latent_kernels = [
                functools.reduce(
                    self._random_binary_map,
                    np.random.choice(self.kernel_bank, num_kernels, replace=True),
                )
                for num_kernels in kernel_numbers
            ]

            try:
                # Sample latent functions
                latent_functions = np.array(
                    [
                        self._sample_from_gp_prior(
                            kernel=kernel, X=X, random_seed=random_seed
                        )
                        for kernel in latent_kernels
                    ]
                )

                # Sample Dirichlet parameter
                dirichlet = np.random.uniform(self.dirichlet_min, self.dirichlet_max)

                # Sample weights for combining latent functions
                weights = np.random.dirichlet(
                    dirichlet * np.ones(latent_num), size=self.num_channels
                )

                # Combine latent functions with weights
                ts = np.dot(weights, latent_functions)  # Shape: [num_channels, length]
                ts = ts.T  # Transpose to [length, num_channels]

                start_time = np.datetime64(DEFAULT_START_DATE, "s")

                return {"start": start_time, "values": ts}

            except np.linalg.LinAlgError as err:
                print("Error caught:", err)
                continue
