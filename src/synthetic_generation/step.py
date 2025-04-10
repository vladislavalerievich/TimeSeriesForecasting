from datetime import date

import numpy as np
import pandas as pd
import torch

from src.synthetic_generation.constants import BASE_END_ORD, BASE_START_ORD


def generate_step_batch(batch_size, seq_len, pred_len, step_config=None):
    """
    Generates a batch of periodic step function data with seasonalities and trends,
    with step_config adjusted based on seq_len for consistency.

    Args:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Total length of each time series sample (history + target).
        pred_len (int): Length of the target sequence to predict (unused for generation consistency).
        step_config (dict, optional): Configuration for step function parameters.
                                      Keys: 'num_cycles_range', 'height_range', 'trend_prob', 'trend_max_change'.

    Returns:
        dict: A dictionary containing tensors for the batch, matching the
              format expected by the training script.
    """
    if step_config is None:
        step_config = {
            "num_cycles_range": (5, 10),  # Desired number of cycles in seq_len
            "height_range": (-2.0, 2.0),  # Min/max step height
            "trend_prob": 0.5,  # Probability of adding a trend
            "trend_max_change": 4.0,  # Max total trend change over seq_len
        }

    batch_ts_features = np.zeros((batch_size, seq_len, 7), dtype=np.int64)
    batch_values = np.zeros((batch_size, seq_len), dtype=np.float32)
    task = np.zeros((batch_size, pred_len), dtype=np.int64)

    for i in range(batch_size):
        # Compute period based on seq_len and desired number of cycles
        min_cycles, max_cycles = step_config["num_cycles_range"]
        min_period = max(2, seq_len // max_cycles)  # Ensure at least 2 timesteps/period
        max_period = max(min_period, seq_len // min_cycles)
        period = np.random.randint(min_period, max_period + 1)

        # Number of steps per period (fixed range, e.g., 2-5)
        steps_per_period = np.random.randint(2, 6)
        step_duration = period // steps_per_period
        if step_duration < 1:
            step_duration = 1
            steps_per_period = period  # Adjust to fit period

        # Generate step heights for one period
        step_heights = np.random.uniform(
            step_config["height_range"][0],
            step_config["height_range"][1],
            steps_per_period,
        )

        # Repeat the pattern across seq_len
        time_idx = np.arange(seq_len)
        period_indices = (time_idx // step_duration) % steps_per_period
        values = step_heights[period_indices]

        # Add a linear trend with scaled slope
        if np.random.rand() < step_config["trend_prob"]:
            max_change = step_config["trend_max_change"]
            slope = np.random.uniform(-max_change, max_change) / seq_len
            trend = slope * time_idx
            values += trend

        batch_values[i, :] = values

        # Generate date features (unchanged)
        start_ord = np.random.randint(BASE_START_ORD, BASE_END_ORD + 1)
        try:
            start_date = date.fromordinal(start_ord)
        except ValueError:
            start_date = date.fromordinal(BASE_START_ORD)
        start_timestamp = pd.Timestamp(start_date)
        dates = pd.date_range(start=start_timestamp, periods=seq_len, freq="D")

        ts_features = np.stack(
            [
                dates.year.values,
                dates.month.values,
                dates.day.values,
                dates.dayofweek.values + 1,
                dates.dayofyear.values,
                dates.hour.values,
                dates.minute.values,
            ],
            axis=-1,
        ).astype(np.int64)
        batch_ts_features[i, :, :] = ts_features

    # Split into history and target
    history_len = seq_len - pred_len
    combined_samples = np.concatenate(
        [batch_ts_features, batch_values[:, :, np.newaxis]], axis=-1
    )
    history_ts_y = combined_samples[:, :history_len, :]
    target_ts = combined_samples[:, history_len:, :]

    # Prepare final batch dictionary
    batch = {
        "ts": torch.from_numpy(history_ts_y[:, :, :7]),
        "history": torch.from_numpy(history_ts_y[:, :, 7].astype(np.float32)),
        "target_dates": torch.from_numpy(target_ts[:, :, :7]),
        "target_values": torch.from_numpy(target_ts[:, :, 7].astype(np.float32)),
        "task": torch.from_numpy(task),
        "complete_target": torch.from_numpy(target_ts.astype(np.float32)),
    }
    return batch
