import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

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

        # Configurations for generators
        self.sine_config = config.get("sine_wave_config", None)
        self.step_config = config.get(
            "step_wave_config", None
        )  # Allow custom step config
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

        # Fix from previous analysis to respect max_seq_len
        min_hist_len = self.min_seq_len
        max_hist_len = self.max_seq_len - pred_len
        if max_hist_len < min_hist_len:
            max_hist_len = min_hist_len
        history_len = np.random.randint(min_hist_len, max_hist_len + 1)
        seq_len = history_len + pred_len

        # 50% chance of sine wave, 50% chance of step function
        if np.random.rand() < 0.5:
            batch = generate_sine_batch(
                batch_size=self.batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                sine_config=self.sine_config,
            )
        else:
            batch = generate_step_batch(
                batch_size=self.batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                step_config=self.step_config,
            )
        return batch

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
        return batch  # Batch is already formatted by generator functions


def generate_mixed_fixed_batch(
    batch_size, seq_len, pred_len, sine_config=None, step_config=None
):
    """
    Generates a fixed batch with 50% sine wave samples and 50% step function samples.

    Args:
        batch_size (int): Total number of samples in the batch (should be even for exact 50/50 split).
        seq_len (int): Total length of each time series sample (history + target).
        pred_len (int): Length of the target sequence to predict.
        sine_config (dict, optional): Configuration for sine wave parameters.
        step_config (dict, optional): Configuration for step function parameters.

    Returns:
        dict: A dictionary containing tensors for the batch, with keys matching the training script:
              ('ts', 'history', 'target_dates', 'target_values', 'task', 'complete_target').
    """
    # Ensure batch_size is even for a clean 50/50 split; adjust if odd
    if batch_size % 2 != 0:
        print(
            f"Warning: batch_size {batch_size} is odd, adjusting to {batch_size + 1} for even split."
        )
        batch_size += 1

    half_batch = batch_size // 2

    # Generate sine wave samples
    sine_batch = generate_sine_batch(
        batch_size=half_batch,
        seq_len=seq_len,
        pred_len=pred_len,
        sine_config=sine_config,
    )

    # Generate step function samples
    step_batch = generate_step_batch(
        batch_size=half_batch,
        seq_len=seq_len,
        pred_len=pred_len,
        step_config=step_config,
    )

    # Combine the two batches
    mixed_batch = {}
    for key in sine_batch.keys():
        # Concatenate along the batch dimension (dim=0)
        mixed_batch[key] = torch.cat((sine_batch[key], step_batch[key]), dim=0)

    # Optional: Shuffle the batch to mix sine and step samples (if order matters for visualization)
    shuffle_indices = torch.randperm(batch_size)
    for key in mixed_batch.keys():
        mixed_batch[key] = mixed_batch[key][shuffle_indices]

    return mixed_batch


def generate_fixed_synthetic_batch(config, batch_size=6):
    """
    Generates a fixed batch with 50% sine waves and 50% step functions.

    Args:
        config (dict): Configuration dictionary with seq_len, pred_len, etc.
        batch_size (int): Desired batch size (must be even for exact 50/50 split).

    Returns:
        dict: Batch with mixed sine and step data.
    """
    assert batch_size % 2 == 0, "Batch size must be even for 50/50 split"
    half_size = batch_size // 2

    # Compute seq_len and pred_len from config
    pred_len = config["pred_len"]  # 128
    history_len = config["context_len"]  # 512
    seq_len = history_len + pred_len  # 640

    # Generate sine batch
    sine_batch = generate_sine_batch(
        batch_size=half_size,
        seq_len=seq_len,
        pred_len=pred_len,
        sine_config=config.get("sine_wave_config", None),
    )

    # Generate step batch
    step_batch = generate_step_batch(
        batch_size=half_size,
        seq_len=seq_len,
        pred_len=pred_len,
        step_config=config.get("step_wave_config", None),
    )

    # Concatenate batches
    fixed_batch = {}
    for key in sine_batch:
        fixed_batch[key] = torch.cat([sine_batch[key], step_batch[key]], dim=0)

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
