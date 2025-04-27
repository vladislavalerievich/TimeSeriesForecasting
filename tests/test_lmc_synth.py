import functools
from typing import Optional

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)

from src.synthetic_generation.lmc_synth import LMCSynthGenerator

LENGTH = 1024
KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),  # H
    ExpSineSquared(periodicity=48 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=24 * 7 / LENGTH),  # H
    ExpSineSquared(periodicity=48 * 7 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 * 7 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=7 / LENGTH),  # D
    ExpSineSquared(periodicity=14 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=30 / LENGTH),  # D
    ExpSineSquared(periodicity=60 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=365 / LENGTH),  # D
    ExpSineSquared(periodicity=365 * 2 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=4 / LENGTH),  # W
    ExpSineSquared(periodicity=26 / LENGTH),  # W
    ExpSineSquared(periodicity=52 / LENGTH),  # W
    ExpSineSquared(periodicity=4 / LENGTH),  # M
    ExpSineSquared(periodicity=6 / LENGTH),  # M
    ExpSineSquared(periodicity=12 / LENGTH),  # M
    ExpSineSquared(periodicity=4 / LENGTH),  # Q
    ExpSineSquared(periodicity=4 * 10 / LENGTH),  # Q
    ExpSineSquared(periodicity=10 / LENGTH),  # Y
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


def random_binary_map(a: Kernel, b: Kernel):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    Parameters
    ----------
    a
        A GP kernel.
    b
        A GP kernel.

    Returns
    -------
        The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, X: np.ndarray, random_seed: Optional[int] = None
):
    """
    Draw a sample from a GP prior.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
):
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.
    method, optional
        The sampling method for multivariate_normal, by default `eigh`.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )

    return ts


def generate_time_series(
    dirichlet_min,
    dirichlet_max,
    scale,
    weibull_shape,
    weibul_scale,
    max_kernels: int = 5,
    num_channels: int = 10,
):
    """Generate a synthetic time series from LMC_Synth."""
    np.random.seed(42)

    while True:
        X = np.linspace(0, 1, LENGTH)

        # sample from weibull the number of latent functions
        latent_num = np.rint(np.random.weibull(weibull_shape, weibul_scale) * scale + 1)

        latent_num = np.clip(latent_num, max(2, num_channels // 20), num_channels)
        latent_num = int(latent_num[0])

        # now we will have latent num of latent functions. Each latent function will have a random number of kernels
        kernel_numbers = np.random.randint(1, max_kernels + 1, size=latent_num)
        # now that  we have the number of kernels for each latent function
        # we will sample the kernels for each latent function
        latent_kernels = [
            functools.reduce(
                random_binary_map,
                np.random.choice(KERNEL_BANK, num_kernels, replace=True),
            )
            for num_kernels in kernel_numbers
        ]

        try:
            # now we have the kernels for each latent function
            # we will now sample the latent functions
            latent_functions = np.array(
                [
                    sample_from_gp_prior_efficient(kernel=kernel, X=X, random_seed=42)
                    for kernel in latent_kernels
                ]
            )
            # sample dirichlet parameter between dirichlet_min and dirichlet_max uniformly
            dirichlet = np.random.uniform(dirichlet_min, dirichlet_max)
            # now sample the weights for each latent function
            weights = np.random.dirichlet(
                dirichlet * np.ones(latent_num), size=num_channels
            )
            # now we will combine the latent functions with the weights
            ts = np.dot(weights, latent_functions)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        return {"start": np.datetime64("2000-01-01 00:00", "s"), "target": ts}


# Define test cases as a list of dictionaries
TEST_CASES = [
    {
        "name": "1",
        "length": LENGTH,
        "max_kernels": 5,
        "num_channels": 10,
        "dirichlet_min": 0.1,
        "dirichlet_max": 2.0,
        "scale": 1.0,
        "weibull_shape": 2.0,
        "weibull_scale": 1,
    },
    {
        "name": "2",
        "length": LENGTH,
        "max_kernels": 3,
        "num_channels": 5,
        "dirichlet_min": 0.5,
        "dirichlet_max": 1.5,
        "scale": 0.5,
        "weibull_shape": 1.5,
        "weibull_scale": 2,
    },
    {
        "name": "3",
        "length": LENGTH,
        "max_kernels": 7,
        "num_channels": 20,
        "dirichlet_min": 0.05,
        "dirichlet_max": 3.0,
        "scale": 2.0,
        "weibull_shape": 3.0,
        "weibull_scale": 3,
    },
    {
        "name": "4",
        "length": LENGTH,
        "max_kernels": 1,
        "num_channels": 2,
        "dirichlet_min": 0.01,
        "dirichlet_max": 1.0,
        "scale": 0.1,
        "weibull_shape": 1.0,
        "weibull_scale": 4,
    },
]


@pytest.fixture
def seed():
    """Fixture to set a fixed random seed for reproducibility."""
    return 42


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[case["name"] for case in TEST_CASES],  # Use test case names as test IDs
)
def test_generate_time_series_equivalence(seed, test_case):
    """
    Test that LMCSynthGenerator.generate_time_series produces the same output as the original
    generate_time_series function for various parameter combinations.

    Parameters
    ----------
    seed : int
        Fixed random seed for reproducibility.
    test_case : dict
        Dictionary containing test parameters: length, max_kernels, num_channels,
        dirichlet_min, dirichlet_max, scale, weibull_shape, weibull_scale.
    """
    # Extract parameters from test case
    params = test_case

    # Initialize LMCSynthGenerator with test parameters
    generator = LMCSynthGenerator(
        length=params["length"],
        max_kernels=params["max_kernels"],
        num_channels=params["num_channels"],
        dirichlet_min=params["dirichlet_min"],
        dirichlet_max=params["dirichlet_max"],
        scale=params["scale"],
        weibull_shape=params["weibull_shape"],
        weibull_scale=params["weibull_scale"],
    )

    # Generate time series using LMCSynthGenerator
    new_result = generator.generate_time_series(random_seed=seed, periodicity="s")

    # Generate time series using original function
    original_result = generate_time_series(
        dirichlet_min=params["dirichlet_min"],
        dirichlet_max=params["dirichlet_max"],
        scale=params["scale"],
        weibull_shape=params["weibull_shape"],
        weibul_scale=params["weibull_scale"],
        max_kernels=params["max_kernels"],
        num_channels=params["num_channels"],
    )

    # Compare timestamps/start times
    expected_timestamps = original_result["start"] + np.arange(
        params["length"]
    ) * np.timedelta64(1, "s")
    assert np.array_equal(new_result["timestamps"], expected_timestamps), (
        f"Timestamps do not match for test case: {params['name']}"
    )

    # Compare values/target
    assert np.allclose(new_result["values"], original_result["target"], atol=1e-8), (
        f"Generated time series values do not match for test case: {params['name']}"
    )


def test_generate_time_series_deterministic(seed):
    """
    Test that LMCSynthGenerator.generate_time_series is deterministic with the same seed.
    """
    generator = LMCSynthGenerator(length=1024, max_kernels=5, num_channels=10)

    # Generate two time series with the same seed
    result1 = generator.generate_time_series(random_seed=seed, periodicity="s")
    result2 = generator.generate_time_series(random_seed=seed, periodicity="s")

    # Check that outputs are identical
    assert np.array_equal(result1["timestamps"], result2["timestamps"]), (
        "Timestamps are not deterministic with same seed"
    )
    assert np.allclose(result1["values"], result2["values"], atol=1e-3), (
        "Values are not deterministic with same seed"
    )
