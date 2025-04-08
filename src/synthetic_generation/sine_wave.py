import random
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.synthetic_generation.constants import BASE_END_ORD, BASE_START_ORD


def generate_sine_batch(batch_size, seq_len, pred_len, sine_config=None):
    """
    Generates a batch of sine wave data with date features.

    Note: 'task' is included as a placeholder for multi-task learning; set to zeros for now.
    'complete_target' duplicates 'target_dates' and 'target_values' for compatibility with training script.
    Uses daily frequency ('D')—adjust to 'H' for hourly data if needed.

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
