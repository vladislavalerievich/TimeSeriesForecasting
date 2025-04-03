import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

# Define reasonable start/end dates for random date generation
# (Using dates relative to current date for broader applicability)
try:
    # Use current date to define a range
    DEFAULT_END_DATE = date.today()
    DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(
        days=20 * 365
    )  # Approx 20 years back
    BASE_START_ORD = DEFAULT_START_DATE.toordinal()
    BASE_END_ORD = DEFAULT_END_DATE.toordinal()
except Exception:
    # Fallback if date operations fail (e.g., in restricted environments)
    BASE_START_ORD = 700000  # Arbitrary reasonable ordinal dates
    BASE_END_ORD = 740000


def generate_sine_batch(batch_size, seq_len, pred_len, sine_config=None):
    """
    Generates a batch of sine wave data with date features.

    Args:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Total length of each time series sample (history + target).
        pred_len (int): Length of the target sequence to predict.
        sine_config (dict, optional): Configuration for sine wave parameters.
                                      Defaults provide reasonable ranges.
                                      Expected keys: 'period_range', 'amp_range', 'phase_range'.

    Returns:
        dict: A dictionary containing tensors for the batch, matching the
              format expected by the training script ('ts', 'history',
              'target_dates', 'target_values', 'task', 'complete_target').
    """
    if sine_config is None:
        sine_config = {
            "period_range": (10, 100),  # Min/max period
            "amp_range": (0.5, 3.0),  # Min/max amplitude
            "phase_range": (0, 2 * np.pi),  # Min/max phase shift
        }

    batch_ts_features = np.zeros(
        (batch_size, seq_len, 7), dtype=np.int64
    )  # 7 date features
    batch_values = np.zeros((batch_size, seq_len), dtype=np.float32)
    # Assuming multipoint prediction based on training loop structure
    task = np.zeros((batch_size, pred_len), dtype=np.int64)

    for i in range(batch_size):
        # Generate random parameters for this sample
        period = np.random.uniform(
            sine_config["period_range"][0], sine_config["period_range"][1]
        )
        amplitude = np.random.uniform(
            sine_config["amp_range"][0], sine_config["amp_range"][1]
        )
        phase = np.random.uniform(
            sine_config["phase_range"][0], sine_config["phase_range"][1]
        )

        # Generate time index and values
        time_idx = np.arange(seq_len)
        values = amplitude * np.sin(2 * np.pi * time_idx / period + phase)
        # Add a small random DC offset for slight variety
        values += np.random.normal(0, amplitude * 0.1)
        batch_values[i, :] = values

        # --- Generate date features ---
        # Choose a random start date within the defined range
        start_ord = np.random.randint(BASE_START_ORD, BASE_END_ORD + 1)
        try:
            start_date = date.fromordinal(start_ord)
        except ValueError:  # Handle potential out-of-range errors for ordinals
            start_date = date.fromordinal(BASE_START_ORD)  # Fallback

        start_timestamp = pd.Timestamp(start_date)
        # Using daily frequency ('D') for simplicity. Change if needed.
        dates = pd.date_range(start=start_timestamp, periods=seq_len, freq="D")

        # Extract date features (consistent with the complex generator)
        ts_features = np.stack(
            [
                dates.year.values,
                dates.month.values,
                dates.day.values,
                dates.dayofweek.values + 1,  # Monday=1, Sunday=7 convention often used
                dates.dayofyear.values,
                dates.hour.values,  # Will be 0 for daily freq
                dates.minute.values,  # Will be 0 for daily freq
            ],
            axis=-1,
        ).astype(np.int64)
        batch_ts_features[i, :, :] = ts_features

    # Combine features and values temporarily for easy splitting
    # Shape: (batch_size, seq_len, 8) where last dim is the value
    combined_samples = np.concatenate(
        [batch_ts_features, batch_values[:, :, np.newaxis]], axis=-1
    )

    # Split into history and target
    history_len = seq_len - pred_len
    # History includes date features [:, :, :7] and values [:, :, 7]
    history_ts_y = combined_samples[:, :history_len, :]
    # Target includes date features [:, :, :7] and values [:, :, 7]
    target_ts = combined_samples[:, history_len:, :]

    # Prepare final batch dictionary with torch tensors
    batch = {
        # History date features: (batch, history_len, 7)
        "ts": torch.from_numpy(history_ts_y[:, :, :7]),
        # History values: (batch, history_len)
        "history": torch.from_numpy(history_ts_y[:, :, 7].astype(np.float32)),
        # Target date features: (batch, pred_len, 7)
        "target_dates": torch.from_numpy(target_ts[:, :, :7]),
        # Target values: (batch, pred_len)
        "target_values": torch.from_numpy(target_ts[:, :, 7].astype(np.float32)),
        # Task identifier: (batch, pred_len) - Set to 0 for simple test
        "task": torch.from_numpy(task),
        # Complete target (features + values): (batch, pred_len, 8) - needed by training loop logic
        "complete_target": torch.from_numpy(target_ts.astype(np.float32)),
    }
    return batch


class SineWaveDataset(IterableDataset):
    """
    An IterableDataset that generates batches of sine wave data with date features
    on the fly, matching the format required by the training script.
    """

    def __init__(
        self, config, mode="train", cpus_available=1, device="cpu", initial_epoch=0
    ):
        """
        Initializes the dataset.

        Args:
            config (dict): Configuration dictionary. Expected keys include:
                           'batch_size', 'min_seq_len', 'max_seq_len',
                           'pred_len', 'pred_len_min' (if pred_len_sample=True),
                           'pred_len_sample', 'training_rounds' (for train mode),
                           'validation_rounds' (for val mode),
                           'sine_wave_config' (optional dict for sine params).
            mode (str): 'train' or 'val', determines number of batches per iter.
            cpus_available (int): Number of CPUs (used for calculating rounds).
            device (str): Target device (used for potential pinning).
            initial_epoch (int): Starting epoch (used for batch counter logic if needed).
        """
        self.config = config
        self.batch_size = config["batch_size"]
        self.min_seq_len = config.get("min_seq_len", 64)  # Provide default if missing
        self.max_seq_len = config.get("max_seq_len", 512)  # Provide default
        self.pred_len_fixed = config["pred_len"]
        self.pred_len_min = config.get(
            "pred_len_min", self.pred_len_fixed // 2
        )  # Default min pred
        self.pred_len_sample = config.get(
            "pred_len_sample", False
        )  # Default to fixed pred len

        self.sine_config = config.get(
            "sine_wave_config", None
        )  # Pass None to use defaults in generate_sine_batch
        self.device = device

        if mode == "train":
            self.batches_per_iter = (
                int(np.ceil(config["training_rounds"] / cpus_available))
                if cpus_available > 0
                else config["training_rounds"]
            )
        else:  # 'val' or other modes
            self.batches_per_iter = (
                int(np.ceil(config["validation_rounds"] / cpus_available))
                if cpus_available > 0
                else config["validation_rounds"]
            )

        # Optional: batch counter if complex logic depended on it (e.g., curriculum)
        # self.batch_counter = initial_epoch * self.batches_per_iter

    def _generate_data_batch(self):
        """Generates a single batch of sine wave data."""
        # Determine prediction length for this batch
        if self.pred_len_sample:
            pred_len = np.random.randint(self.pred_len_min, self.pred_len_fixed + 1)
        else:
            pred_len = self.pred_len_fixed

        # Determine sequence length for this batch
        # Ensure max_seq_len is valid given pred_len
        min_hist_len = self.min_seq_len
        max_hist_len = self.max_seq_len
        if max_hist_len < min_hist_len:
            max_hist_len = min_hist_len  # Sanity check

        history_len = np.random.randint(min_hist_len, max_hist_len + 1)
        seq_len = history_len + pred_len

        # Generate the batch using the helper function
        batch = generate_sine_batch(
            batch_size=self.batch_size,
            seq_len=seq_len,
            pred_len=pred_len,
            sine_config=self.sine_config,
        )
        # Optional: Increment counter if needed
        # self.batch_counter += 1
        return batch

    def __iter__(self):
        """Yields batches."""
        for _ in range(self.batches_per_iter):
            yield self._generate_data_batch()

    def worker_init_fn(self, worker_id):
        """Ensures randomness across workers."""
        # Seed numpy and random for this worker based on torch's initial seed and worker_id
        seed = (torch.initial_seed() + int(worker_id) + random.randint(0, 10000)) % (
            2**32
        )
        np.random.seed(seed)
        random.seed(seed)
        # Note: torch manual seed is usually handled by the DataLoader itself per worker

    def collate_fn(self, batch):
        """
        Custom collate function (optional).
        Since _generate_data_batch already returns the final batch dictionary,
        this might just involve pinning memory if needed.
        """
        # batch is already the fully formed dictionary from _generate_data_batch
        # Pin memory if using multiple GPUs and configured to do so
        # This basic version just returns the batch as is.
        # Add pinning logic here if required, similar to GenerativeDataset*
        return batch


def train_val_loader(config, cpus_available, initial_epoch=0, device="cpu"):
    """
    Creates training and validation DataLoaders.
    Can switch between complex synthetic data and simple sine wave data.
    """
    print("--- Using SineWaveDataset ---")
    # Use the new SineWaveDataset
    train_dataset = SineWaveDataset(
        config,
        mode="train",
        cpus_available=cpus_available,
        device=device,
        initial_epoch=initial_epoch,
    )
    val_dataset = SineWaveDataset(
        config,
        mode="val",
        cpus_available=cpus_available,
        device=device,
        initial_epoch=initial_epoch,
    )
    collate_function = train_dataset.collate_fn  # Use its collate function
    worker_init = train_dataset.worker_init_fn  # Use its worker init

    # Create DataLoaders
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,  # Already handled by IterableDataset returning a full batch
        shuffle=False,  # Not applicable for IterableDataset
        collate_fn=collate_function,  # Pass the chosen collate function
        worker_init_fn=worker_init,  # Pass the chosen worker init
        num_workers=cpus_available
        if cpus_available > 0
        else 0,  # Use 0 workers if only 1 CPU or debugging
        prefetch_factor=15 if cpus_available > 1 else None,  # Sensible prefetch
        persistent_workers=cpus_available > 0,  # Keep workers alive
        pin_memory=(device != "cpu"),  # Pin memory if using GPU
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
