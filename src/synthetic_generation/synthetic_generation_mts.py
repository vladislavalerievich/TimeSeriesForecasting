import multiprocessing
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.data_handling.time_series_data_structure import TimeSeriesData
from src.synthetic_generation.lmc_synth_old import TimeSeriesGenerator


class MVTimeSeriesDataset(IterableDataset):
    """
    An IterableDataset that generates batches of synthetic multivariate time series data
    on the fly using TimeSeriesGenerator, with one target feature and multiple history features.
    """

    def __init__(
        self, config, mode="train", cpus_available=1, device="cpu", initial_epoch=0
    ):
        self.config = config
        self.batch_size = config["batch_size"]
        self.min_seq_len = config.get("min_seq_len", 64)
        self.max_seq_len = config.get("max_seq_len", 512)
        self.pred_len_fixed = config["pred_len"]
        self.pred_len_min = config.get("pred_len_min", self.pred_len_fixed // 2)
        self.pred_len_sample = config.get("pred_len_sample", False)

        # Configuration for TimeSeriesGenerator
        self.ts_generator_config = config.get("time_series_generator_config", {})
        self.num_features_min = config.get("num_features_min", 8)
        self.num_features_max = config.get("num_features_max", 32)
        self.num_ts_features = config.get("num_ts_features", 4)
        self.device = device

        if mode == "train":
            self.batches_per_iter = (
                int(np.ceil(config["training_rounds"] / cpus_available))
                if cpus_available > 0
                else config["training_rounds"]
            )
        else:  # 'val'
            self.batches_per_iter = (
                int(np.ceil(config["validation_rounds"] / cpus_available))
                if cpus_available > 0
                else config["validation_rounds"]
            )

    def _generate_data_batch(self):
        """Generates a single batch of synthetic multivariate time series data."""
        if self.pred_len_sample:
            pred_len = np.random.randint(self.pred_len_min, self.pred_len_fixed + 1)
        else:
            pred_len = self.pred_len_fixed

        min_hist_len = self.min_seq_len
        max_hist_len = self.max_seq_len - pred_len
        if max_hist_len < min_hist_len:
            max_hist_len = min_hist_len
        history_len = np.random.randint(min_hist_len, max_hist_len + 1)

        # Vary the number of features for diversity
        num_channels = np.random.randint(
            self.num_features_min, self.num_features_max + 1
        )

        # Configure and use TimeSeriesGenerator
        generator_config = {
            "batch_size": self.batch_size,
            "history_len": history_len,
            "target_len": pred_len,
            "num_channels": num_channels,
            "num_ts_features": self.num_ts_features,
            "variable_lengths": False,  # We're already varying lengths at this level
            "seed": random.randint(0, 10000),  # Random seed for each batch
        }

        # Merge with user-provided config, with user config taking precedence
        generator_config.update(self.ts_generator_config)

        # Create generator and generate data
        generator = TimeSeriesGenerator(**generator_config)
        data_batch = generator.generate_batch()

        # Extract only one target feature (first channel)
        # Shape goes from [batch_size, pred_len, num_channels] to [batch_size, pred_len, 1]
        target_values = data_batch.target_values[:, :, :1]

        return TimeSeriesData(
            history_ts=data_batch.history_ts,
            history_values=data_batch.history_values,
            target_ts=data_batch.target_ts,
            target_values=target_values,
            task=data_batch.task,
        )

    def __iter__(self):
        for _ in range(self.batches_per_iter):
            yield self._generate_data_batch()

    def worker_init_fn(self, worker_id):
        seed = (torch.initial_seed() + int(worker_id) + random.randint(0, 10000)) % (
            2**32
        )
        np.random.seed(seed)
        random.seed(seed)

    def collate_fn(self, batch):
        return batch  # Batch is already a TimeSeriesData


def generate_fixed_multivariate_batch(config, batch_size=6) -> TimeSeriesData:
    """
    Generates a fixed batch of multivariate time series data for visualization and testing.

    Args:
        config (dict): Configuration dictionary with key parameters
        batch_size (int): Number of time series to generate

    Returns:
        TimeSeriesData: A dataclass instance with multivariate time series data
    """
    # Determine key parameters from config
    pred_len = config["pred_len"]
    history_len = config["context_len"]
    num_features = config.get("num_features", 16)
    num_ts_features = config.get("num_ts_features", 4)

    # Generate variety in data with different seeds
    half_size = batch_size // 2

    # Configure first generator for more periodic patterns
    generator1_config = {
        "batch_size": half_size,
        "history_len": history_len,
        "target_len": pred_len,
        "num_channels": num_features,
        "num_ts_features": num_ts_features,
        "variable_lengths": False,
        "max_kernels": 3,
        "dirichlet_min": 0.5,
        "dirichlet_max": 2.0,
        "seed": 42,
    }

    # Configure second generator for more complex patterns
    generator2_config = {
        "batch_size": batch_size - half_size,
        "history_len": history_len,
        "target_len": pred_len,
        "num_channels": num_features,
        "num_ts_features": num_ts_features,
        "variable_lengths": False,
        "max_kernels": 5,
        "dirichlet_min": 0.1,
        "dirichlet_max": 5.0,
        "seed": 123,
    }

    # Create generators and generate data
    generator1 = TimeSeriesGenerator(**generator1_config)
    generator2 = TimeSeriesGenerator(**generator2_config)

    batch1 = generator1.generate_batch()
    batch2 = generator2.generate_batch()

    # Extract only one target feature (first channel) for both batches
    target_values1 = batch1.target_values[:, :, :1]
    target_values2 = batch2.target_values[:, :, :1]

    # Combine the batches
    fixed_batch = TimeSeriesData(
        history_ts=torch.cat([batch1.history_ts, batch2.history_ts], dim=0),
        history_values=torch.cat([batch1.history_values, batch2.history_values], dim=0),
        target_ts=torch.cat([batch1.target_ts, batch2.target_ts], dim=0),
        target_values=torch.cat([target_values1, target_values2], dim=0),
        task=torch.cat([batch1.task, batch2.task], dim=0),
    )

    return fixed_batch


def train_val_loader(config, initial_epoch=0, device="cpu"):
    """
    Creates training and validation DataLoaders with multivariate time series data,
    automatically determining and limiting the number of worker processes.
    """
    print("--- Using MVTimeSeriesDataset (Multivariate Time Series) ---")
    import os

    # Auto-detect available CPU cores
    max_available_cpus = multiprocessing.cpu_count()
    os_cpu_count = os.cpu_count()
    # Safe default: cap to 2 or fewer if system has fewer
    cpus_available = min(config.get("cpus_available", 1), max_available_cpus)
    print(
        f"Detected from multiprocessing {max_available_cpus} CPU cores. Using {cpus_available} DataLoader workers."
    )
    print("os_cpu_count = ", os_cpu_count)

    train_dataset = MVTimeSeriesDataset(
        config,
        mode="train",
        cpus_available=cpus_available,
        device=device,
        initial_epoch=initial_epoch,
    )
    val_dataset = MVTimeSeriesDataset(
        config,
        mode="val",
        cpus_available=cpus_available,
        device=device,
        initial_epoch=initial_epoch,
    )

    collate_function = train_dataset.collate_fn
    worker_init = train_dataset.worker_init_fn

    # Shared loader settings
    loader_kwargs = {
        "batch_size": None,
        "shuffle": False,
        "collate_fn": collate_function,
        "worker_init_fn": worker_init,
        "num_workers": cpus_available,
        "prefetch_factor": 15 if cpus_available > 1 else None,
        "persistent_workers": cpus_available > 0,
        "pin_memory": (device != "cpu"),
    }

    train_data_loader = DataLoader(dataset=train_dataset, **loader_kwargs)
    val_data_loader = DataLoader(dataset=val_dataset, **loader_kwargs)

    return train_data_loader, val_data_loader
