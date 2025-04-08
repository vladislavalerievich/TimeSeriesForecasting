from datetime import date

import numpy as np
import pandas as pd
import torch

from src.synthetic_generation.constants import BASE_END_ORD, BASE_START_ORD


def generate_step_batch(batch_size, seq_len, pred_len, step_config=None, noise=True):
    """
    Generates a batch of simple step function data with date features

    Args:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Total length of each time series sample (history + target).
        pred_len (int): Length of the target sequence to predict.
        step_config (dict, optional): Configuration for step function parameters.
                                      Defaults provide reasonable ranges.
                                      Expected keys: 'num_steps_range', 'height_range'.
        noise (bool): If True, adds Gaussian noise to the generated values. Defaults to False.

    Returns:
        dict: A dictionary containing tensors for the batch, matching the
              format expected by the training script ('ts', 'history',
              'target_dates', 'target_values', 'task', 'complete_target').
    """
    if step_config is None:
        step_config = {
            "num_steps_range": (2, 10),  # Min/max number of steps
            "height_range": (-2.0, 2.0),  # Min/max step height
        }

    batch_ts_features = np.zeros(
        (batch_size, seq_len, 7), dtype=np.int64
    )  # 7 date features
    batch_values = np.zeros((batch_size, seq_len), dtype=np.float32)
    task = np.zeros(
        (batch_size, pred_len), dtype=np.int64
    )  # Placeholder for multi-task

    for i in range(batch_size):
        # Generate random parameters for this sample
        num_steps = np.random.randint(
            step_config["num_steps_range"][0], step_config["num_steps_range"][1] + 1
        )
        # Ensure steps fit within seq_len
        step_points = (
            np.sort(np.random.choice(seq_len - 1, num_steps, replace=False)) + 1
        )
        step_points = np.concatenate(([0], step_points))  # Start at t=0
        step_durations = np.diff(np.concatenate((step_points, [seq_len])))
        step_heights = np.random.uniform(
            step_config["height_range"][0],
            step_config["height_range"][1],
            num_steps + 1,
        )

        # Generate step function values without noise
        values = np.zeros(seq_len, dtype=np.float32)
        for j, (start, duration, height) in enumerate(
            zip(step_points, step_durations, step_heights)
        ):
            values[start : start + duration] = height

        if noise:
            values += np.random.normal(0, 0.01, seq_len)

        batch_values[i, :] = values

        # Generate date features
        start_ord = np.random.randint(BASE_START_ORD, BASE_END_ORD + 1)
        try:
            start_date = date.fromordinal(start_ord)
        except ValueError:
            start_date = date.fromordinal(BASE_START_ORD)  # Fallback
        start_timestamp = pd.Timestamp(start_date)
        dates = pd.date_range(start=start_timestamp, periods=seq_len, freq="D")

        ts_features = np.stack(
            [
                dates.year.values,
                dates.month.values,
                dates.day.values,
                dates.dayofweek.values + 1,
                dates.dayofyear.values,
                dates.hour.values,  # 0 for daily freq
                dates.minute.values,  # 0 for daily freq
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
