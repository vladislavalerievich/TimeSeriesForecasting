import argparse
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm


@dataclass
class TimeSeriesData:
    history_ts: torch.Tensor  # [batch_size, seq_len, num_ts_features]
    history_values: torch.Tensor  # [batch_size, seq_len, num_features]
    target_ts: torch.Tensor  # [batch_size, pred_len, num_ts_features]
    target_values: torch.Tensor  # [batch_size, pred_len, num_targets]
    task: torch.Tensor  # [batch_size]


class TimeSeriesGenerator:
    """Generate synthetic multivariate time series data using Gaussian Process priors."""

    KERNEL_BANK = [
        ExpSineSquared(periodicity=24 / 1024),  # H
        ExpSineSquared(periodicity=48 / 1024),  # 0.5H
        ExpSineSquared(periodicity=96 / 1024),  # 0.25H
        ExpSineSquared(periodicity=24 * 7 / 1024),  # H
        ExpSineSquared(periodicity=48 * 7 / 1024),  # 0.5H
        ExpSineSquared(periodicity=96 * 7 / 1024),  # 0.25H
        ExpSineSquared(periodicity=7 / 1024),  # D
        ExpSineSquared(periodicity=14 / 1024),  # 0.5D
        ExpSineSquared(periodicity=30 / 1024),  # D
        ExpSineSquared(periodicity=60 / 1024),  # 0.5D
        ExpSineSquared(periodicity=365 / 1024),  # D
        ExpSineSquared(periodicity=365 * 2 / 1024),  # 0.5D
        ExpSineSquared(periodicity=4 / 1024),  # W
        ExpSineSquared(periodicity=26 / 1024),  # W
        ExpSineSquared(periodicity=52 / 1024),  # W
        ExpSineSquared(periodicity=4 / 1024),  # M
        ExpSineSquared(periodicity=6 / 1024),  # M
        ExpSineSquared(periodicity=12 / 1024),  # M
        ExpSineSquared(periodicity=4 / 1024),  # Q
        ExpSineSquared(periodicity=4 * 10 / 1024),  # Q
        ExpSineSquared(periodicity=10 / 1024),  # Y
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

    def __init__(
        self,
        batch_size: int = 16,
        history_len: int = 768,
        target_len: int = 256,
        num_channels: int = 32,
        num_ts_features: int = 4,
        max_kernels: int = 5,
        dirichlet_min: float = 0.1,
        dirichlet_max: float = 5.0,
        scale: float = 0.5,
        weibull_shape: float = 1.5,
        weibull_scale: float = 10.0,
        variable_lengths: bool = True,
        length_variance: float = 0.2,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
    ):
        """
        Initialize the time series generator.

        Parameters
        ----------
        batch_size : int
            Number of time series to generate
        history_len : int
            Length of the history sequence
        target_len : int
            Length of the target/prediction sequence
        num_channels : int
            Number of channels/features in the time series
        num_ts_features : int
            Number of time features (e.g., hour, day, month)
        max_kernels : int
            Maximum number of kernels to use for each latent function
        dirichlet_min : float
            Minimum value for Dirichlet parameter
        dirichlet_max : float
            Maximum value for Dirichlet parameter
        scale : float
            Scale parameter for number of latent functions
        weibull_shape : float
            Shape parameter for Weibull distribution
        weibull_scale : float
            Scale parameter for Weibull distribution
        variable_lengths : bool
            Whether to generate time series with variable lengths
        length_variance : float
            Variance in length as a fraction of the specified length
        seed : int, optional
            Random seed for reproducibility
        config : dict, optional
            Additional configuration parameters
        n_jobs : int
            Number of parallel jobs for generation
        """
        self.batch_size = batch_size
        self.history_len = history_len
        self.target_len = target_len
        self.num_channels = num_channels
        self.num_ts_features = num_ts_features
        self.max_kernels = max_kernels
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.scale = scale
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale
        self.variable_lengths = variable_lengths
        self.length_variance = length_variance
        self.n_jobs = n_jobs

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Override parameters with config if provided
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Initialize the kernel bank with adjusted periodicity
        self.update_kernel_bank()

    def update_kernel_bank(self):
        """Update kernel bank with adjusted periodicities based on total length."""
        total_len = self.history_len + self.target_len
        self.KERNEL_BANK = [
            ExpSineSquared(periodicity=24 / total_len),  # H
            ExpSineSquared(periodicity=48 / total_len),  # 0.5H
            ExpSineSquared(periodicity=96 / total_len),  # 0.25H
            ExpSineSquared(periodicity=24 * 7 / total_len),  # H
            ExpSineSquared(periodicity=48 * 7 / total_len),  # 0.5H
            ExpSineSquared(periodicity=96 * 7 / total_len),  # 0.25H
            ExpSineSquared(periodicity=7 / total_len),  # D
            ExpSineSquared(periodicity=14 / total_len),  # 0.5D
            ExpSineSquared(periodicity=30 / total_len),  # D
            ExpSineSquared(periodicity=60 / total_len),  # 0.5D
            ExpSineSquared(periodicity=365 / total_len),  # D
            ExpSineSquared(periodicity=365 * 2 / total_len),  # 0.5D
            ExpSineSquared(periodicity=4 / total_len),  # W
            ExpSineSquared(periodicity=26 / total_len),  # W
            ExpSineSquared(periodicity=52 / total_len),  # W
            ExpSineSquared(periodicity=4 / total_len),  # M
            ExpSineSquared(periodicity=6 / total_len),  # M
            ExpSineSquared(periodicity=12 / total_len),  # M
            ExpSineSquared(periodicity=4 / total_len),  # Q
            ExpSineSquared(periodicity=4 * 10 / total_len),  # Q
            ExpSineSquared(periodicity=10 / total_len),  # Y
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

    @staticmethod
    def random_binary_map(a: Kernel, b: Kernel) -> Kernel:
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
    def sample_from_gp_prior_efficient(
        kernel: Kernel,
        X: np.ndarray,
        random_seed: Optional[int] = None,
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

    def generate_time_features(self, length: int) -> np.ndarray:
        """
        Generate time features for the time series.

        Parameters
        ----------
        length : int
            Length of the time series

        Returns
        -------
        np.ndarray
            Time features array of shape [length, num_ts_features]
        """
        # Create basic time features (normalized position, sine/cosine for cyclical patterns)
        time_idx = np.arange(length) / length
        features = [time_idx]

        # Add sine and cosine features for daily, weekly, and yearly patterns
        for period in [1, 7, 365]:  # day, week, year
            features.append(np.sin(2 * np.pi * time_idx * period))
            features.append(np.cos(2 * np.pi * time_idx * period))

        # Slice or pad to get the requested number of features
        features = np.column_stack(features)[:, : self.num_ts_features]

        if features.shape[1] < self.num_ts_features:
            # Pad with random noise features if needed
            pad_width = self.num_ts_features - features.shape[1]
            noise = np.random.normal(0, 0.1, (length, pad_width))
            features = np.column_stack((features, noise))

        return features

    def generate_single_series(
        self,
        history_len: int,
        target_len: int,
        num_channels: int,
        seed: Optional[int] = None,
    ):
        """
        Generate a single multivariate time series with corrected weibull sampling.

        Parameters
        ----------
        history_len : int
            Length of the history sequence
        target_len : int
            Length of the target/prediction sequence
        num_channels : int
            Number of channels/features in the time series
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        Dict
            Dictionary containing history values, target values, and time features
        """
        total_len = history_len + target_len

        if seed is not None:
            np.random.seed(seed)

        X = np.linspace(0, 1, total_len)

        # Sample from weibull the number of latent functions
        # Fix: Correctly pass size parameter to weibull distribution
        latent_num = np.rint(
            np.random.weibull(self.weibull_shape, size=1) * self.scale + 1
        )
        latent_num = np.clip(latent_num, max(2, num_channels // 20), num_channels)
        latent_num = int(latent_num[0])

        # Sample random number of kernels for each latent function
        kernel_numbers = np.random.randint(1, self.max_kernels + 1, size=latent_num)

        # Sample kernels for each latent function
        while True:
            try:
                latent_kernels = [
                    functools.reduce(
                        self.random_binary_map,
                        np.random.choice(self.KERNEL_BANK, num_kernels, replace=True),
                    )
                    for num_kernels in kernel_numbers
                ]

                # Sample the latent functions
                latent_functions = np.array(
                    [
                        self.sample_from_gp_prior_efficient(kernel=kernel, X=X)
                        for kernel in latent_kernels
                    ]
                )

                # Sample dirichlet parameter between dirichlet_min and dirichlet_max uniformly
                dirichlet = np.random.uniform(self.dirichlet_min, self.dirichlet_max)

                # Sample weights for each latent function
                weights = np.random.dirichlet(
                    dirichlet * np.ones(latent_num), size=num_channels
                )

                # Combine latent functions with weights
                ts = np.dot(weights, latent_functions)
                break
            except np.linalg.LinAlgError:
                continue

        # Generate time features
        time_features = self.generate_time_features(total_len)

        # Split into history and target
        history_values = ts[:, :history_len]
        target_values = ts[:, history_len:]
        history_time_features = time_features[:history_len]
        target_time_features = time_features[history_len:]

        # Assign a random task ID (0-9) for potential multi-task learning
        task_id = np.random.randint(0, 10)

        return {
            "history_values": history_values,
            "target_values": target_values,
            "history_time_features": history_time_features,
            "target_time_features": target_time_features,
            "task": task_id,
        }

    def generate_batch(self) -> TimeSeriesData:
        """
        Generate a batch of multivariate time series.

        Returns
        -------
        TimeSeriesData
            Batch of synthetic time series data in the TimeSeriesData format
        """
        # Generate series with potentially variable lengths
        series_data = []

        for i in range(self.batch_size):
            # If variable lengths are enabled, adjust history and target lengths
            if self.variable_lengths:
                history_variance = int(self.history_len * self.length_variance)
                target_variance = int(self.target_len * self.length_variance)

                # Ensure lengths are at least 10% of the original lengths
                history_len = max(
                    int(self.history_len * 0.1),
                    np.random.randint(
                        self.history_len - history_variance,
                        self.history_len + history_variance + 1,
                    ),
                )
                target_len = max(
                    int(self.target_len * 0.1),
                    np.random.randint(
                        self.target_len - target_variance,
                        self.target_len + target_variance + 1,
                    ),
                )
            else:
                history_len = self.history_len
                target_len = self.target_len

            # Generate a single series with specified lengths
            series_data.append(
                self.generate_single_series(
                    history_len=history_len,
                    target_len=target_len,
                    num_channels=self.num_channels,
                    seed=None,  # Let each series have different random patterns
                )
            )

        # Find maximum lengths to pad to
        max_history_len = max(
            self.history_len, max(len(s["history_values"][0]) for s in series_data)
        )
        max_target_len = max(
            self.target_len, max(len(s["target_values"][0]) for s in series_data)
        )

        # Prepare tensors for batch
        history_values_batch = []
        target_values_batch = []
        history_ts_batch = []
        target_ts_batch = []
        task_batch = []

        # Process each series
        for series in series_data:
            # Get shapes
            num_channels = series["history_values"].shape[0]
            history_len = series["history_values"].shape[1]
            target_len = series["target_values"].shape[1]

            # Pad if necessary
            if history_len < max_history_len:
                padded_history = np.zeros((num_channels, max_history_len))
                padded_history[:, :history_len] = series["history_values"]
                padded_history_ts = np.zeros((max_history_len, self.num_ts_features))
                padded_history_ts[:history_len] = series["history_time_features"]
            else:
                padded_history = series["history_values"]
                padded_history_ts = series["history_time_features"]

            if target_len < max_target_len:
                padded_target = np.zeros((num_channels, max_target_len))
                padded_target[:, :target_len] = series["target_values"]
                padded_target_ts = np.zeros((max_target_len, self.num_ts_features))
                padded_target_ts[:target_len] = series["target_time_features"]
            else:
                padded_target = series["target_values"]
                padded_target_ts = series["target_time_features"]

            # Transpose to [seq_len, num_channels] and append to batch
            history_values_batch.append(padded_history.T)
            target_values_batch.append(padded_target.T)
            history_ts_batch.append(padded_history_ts)
            target_ts_batch.append(padded_target_ts)
            task_batch.append(series["task"])

        # Convert to torch tensors with shapes [batch_size, seq_len, num_features]
        history_values = torch.tensor(
            np.array(history_values_batch), dtype=torch.float32
        )
        target_values = torch.tensor(np.array(target_values_batch), dtype=torch.float32)
        history_ts = torch.tensor(np.array(history_ts_batch), dtype=torch.float32)
        target_ts = torch.tensor(np.array(target_ts_batch), dtype=torch.float32)
        task = torch.tensor(np.array(task_batch), dtype=torch.long)

        return TimeSeriesData(
            history_values=history_values,
            target_values=target_values,
            history_ts=history_ts,
            target_ts=target_ts,
            task=task,
        )

    def generate_dataset(
        self, num_batches: int = 1, return_raw: bool = False
    ) -> Union[List[TimeSeriesData], TimeSeriesData]:
        """
        Generate multiple batches of time series data.

        Parameters
        ----------
        num_batches : int
            Number of batches to generate
        return_raw : bool
            If True, return a list of TimeSeriesData objects, otherwise concatenate into one batch

        Returns
        -------
        Union[List[TimeSeriesData], TimeSeriesData]
            Dataset of synthetic time series
        """
        if num_batches <= 0:
            raise ValueError("num_batches must be a positive integer")

        if num_batches == 1:
            return self.generate_batch()

        # Generate batches in parallel
        if self.n_jobs != 1:
            batches = Parallel(n_jobs=self.n_jobs)(
                delayed(self.generate_batch)() for _ in tqdm(range(num_batches))
            )
        else:
            batches = [self.generate_batch() for _ in tqdm(range(num_batches))]

        if return_raw:
            return batches

        # Concatenate all batches into one large batch
        history_values = torch.cat([batch.history_values for batch in batches], dim=0)
        target_values = torch.cat([batch.target_values for batch in batches], dim=0)
        history_ts = torch.cat([batch.history_ts for batch in batches], dim=0)
        target_ts = torch.cat([batch.target_ts for batch in batches], dim=0)
        task = torch.cat([batch.task for batch in batches], dim=0)

        return TimeSeriesData(
            history_values=history_values,
            target_values=target_values,
            history_ts=history_ts,
            target_ts=target_ts,
            task=task,
        )


# Example usage with CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic multivariate time series data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of time series to generate"
    )
    parser.add_argument(
        "--history_len", type=int, default=768, help="Length of the history sequence"
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=256,
        help="Length of the target/prediction sequence",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=32,
        help="Number of channels/features in the time series",
    )
    parser.add_argument(
        "--num_ts_features", type=int, default=4, help="Number of time features"
    )
    parser.add_argument(
        "--max_kernels",
        type=int,
        default=5,
        help="Maximum number of kernels per latent function",
    )
    parser.add_argument(
        "--dirichlet_min", type=float, default=0.1, help="Minimum Dirichlet parameter"
    )
    parser.add_argument(
        "--dirichlet_max", type=float, default=5.0, help="Maximum Dirichlet parameter"
    )
    parser.add_argument(
        "--scale", type=float, default=0.5, help="Scale parameter for latent functions"
    )
    parser.add_argument(
        "--weibull_shape", type=float, default=1.5, help="Weibull shape parameter"
    )
    parser.add_argument(
        "--weibull_scale", type=float, default=10.0, help="Weibull scale parameter"
    )
    parser.add_argument(
        "--variable_lengths",
        action="store_true",
        help="Generate variable length time series",
    )
    parser.add_argument(
        "--length_variance",
        type=float,
        default=0.2,
        help="Variance in length as fraction of specified length",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_batches", type=int, default=1, help="Number of batches to generate"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1, help="Number of parallel jobs for generation"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_ts_data.pt", help="Output file path"
    )

    args = parser.parse_args()

    # Create generator with CLI arguments
    generator = TimeSeriesGenerator(
        batch_size=args.batch_size,
        history_len=args.history_len,
        target_len=args.target_len,
        num_channels=args.num_channels,
        num_ts_features=args.num_ts_features,
        max_kernels=args.max_kernels,
        dirichlet_min=args.dirichlet_min,
        dirichlet_max=args.dirichlet_max,
        scale=args.scale,
        weibull_shape=args.weibull_shape,
        weibull_scale=args.weibull_scale,
        variable_lengths=args.variable_lengths,
        length_variance=args.length_variance,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    # Generate dataset
    print(
        f"Generating {args.num_batches} {'batch' if args.num_batches == 1 else 'batches'} of synthetic time series data..."
    )
    dataset = generator.generate_dataset(num_batches=args.num_batches)

    # Save dataset
    torch.save(dataset, args.output)
    print(f"Dataset saved to {args.output}")
    print("Dataset shapes:")
    print(f"  history_values: {dataset.history_values.shape}")
    print(f"  target_values: {dataset.target_values.shape}")
    print(f"  history_ts: {dataset.history_ts.shape}")
    print(f"  target_ts: {dataset.target_ts.shape}")
    print(f"  task: {dataset.task.shape}")
