import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.data_handling.data_containers import TimeSeriesData
from src.synthetic_generation.sine_wave import generate_sine_batch
from src.synthetic_generation.step import generate_step_batch


class SyntheticDataset(IterableDataset):
    """
    An IterableDataset that generates batches of synthetic data (sine waves or step functions)
    with date features on the fly, matching the format required by the training script.
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

        self.sine_config = config.get("sine_wave_config", None)
        self.step_config = config.get("step_wave_config", None)
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
        """Generates a single batch of synthetic data (sine or step)."""
        if self.pred_len_sample:
            pred_len = np.random.randint(self.pred_len_min, self.pred_len_fixed + 1)
        else:
            pred_len = self.pred_len_fixed

        min_hist_len = self.min_seq_len
        max_hist_len = self.max_seq_len - pred_len
        if max_hist_len < min_hist_len:
            max_hist_len = min_hist_len
        history_len = np.random.randint(min_hist_len, max_hist_len + 1)
        seq_len = history_len + pred_len

        # 50% chance of sine wave, 50% chance of step function
        if np.random.rand() < 0.5:
            return generate_sine_batch(
                batch_size=self.batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                sine_config=self.sine_config,
            )
        else:
            return generate_step_batch(
                batch_size=self.batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                step_config=self.step_config,
            )

    def generate_fixed_batch(self, batch_size):
        """Generate a fixed batch with 50% sine and 50% step functions."""
        assert batch_size % 2 == 0, "Batch size must be even for 50/50 split"
        half_size = batch_size // 2
        pred_len = self.pred_len_fixed
        history_len = self.config["context_len"]
        seq_len = history_len + pred_len

        sine_data = generate_sine_batch(half_size, seq_len, pred_len, self.sine_config)
        step_data = generate_step_batch(half_size, seq_len, pred_len, self.step_config)

        return TimeSeriesData(
            history_ts=torch.cat([sine_data.history_ts, step_data.history_ts], dim=0),
            history_values=torch.cat(
                [sine_data.history_values, step_data.history_values], dim=0
            ),
            target_ts=torch.cat([sine_data.target_ts, step_data.target_ts], dim=0),
            target_values=torch.cat(
                [sine_data.target_values, step_data.target_values], dim=0
            ),
            task=torch.cat([sine_data.task, step_data.task], dim=0),
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


def generate_fixed_synthetic_batch(config, batch_size=6) -> TimeSeriesData:
    """
    Generates a fixed batch with 50% sine waves and 50% step functions.

    Args:
        config (dict): Configuration dictionary with seq_len, pred_len, etc.
        batch_size (int): Desired batch size (must be even for exact 50/50 split).

    Returns:
        TimeSeriesData: A dataclass instance with mixed sine and step data.
    """
    assert batch_size % 2 == 0, "Batch size must be even for 50/50 split"
    half_size = batch_size // 2

    pred_len = config["pred_len"]
    history_len = config["context_len"]
    seq_len = history_len + pred_len

    sine_batch = generate_sine_batch(
        batch_size=half_size,
        seq_len=seq_len,
        pred_len=pred_len,
        sine_config=config.get("sine_wave_config", None),
    )

    step_batch = generate_step_batch(
        batch_size=half_size,
        seq_len=seq_len,
        pred_len=pred_len,
        step_config=config.get("step_wave_config", None),
    )

    fixed_batch = TimeSeriesData(
        history_ts=torch.cat([sine_batch.history_ts, step_batch.history_ts], dim=0),
        history_values=torch.cat(
            [sine_batch.history_values, step_batch.history_values], dim=0
        ),
        target_ts=torch.cat([sine_batch.target_ts, step_batch.target_ts], dim=0),
        target_values=torch.cat(
            [sine_batch.target_values, step_batch.target_values], dim=0
        ),
        task=torch.cat([sine_batch.task, step_batch.task], dim=0),
    )

    return fixed_batch


def train_val_loader(config, cpus_available, initial_epoch=0, device="cpu"):
    """
    Creates training and validation DataLoaders with 50% sine waves and 50% step functions.
    """
    print("--- Using SyntheticDataset (Sine Waves + Step Functions) ---")
    train_dataset = SyntheticDataset(
        config,
        mode="train",
        cpus_available=cpus_available,
        device=device,
        initial_epoch=initial_epoch,
    )
    val_dataset = SyntheticDataset(
        config,
        mode="val",
        cpus_available=cpus_available,
        device=device,
        initial_epoch=initial_epoch,
    )
    collate_function = train_dataset.collate_fn
    worker_init = train_dataset.worker_init_fn

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=collate_function,
        worker_init_fn=worker_init,
        num_workers=cpus_available if cpus_available > 0 else 0,
        prefetch_factor=15 if cpus_available > 1 else None,
        persistent_workers=cpus_available > 0,
        pin_memory=(device != "cpu"),
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=collate_function,
        worker_init_fn=worker_init,
        num_workers=cpus_available if cpus_available > 0 else 0,
        prefetch_factor=15 if cpus_available > 1 else None,
        persistent_workers=cpus_available > 0,
        pin_memory=(device != "cpu"),
    )

    return train_data_loader, val_data_loader
