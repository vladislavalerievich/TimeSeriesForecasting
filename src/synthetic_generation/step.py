from datetime import date

import numpy as np
import pandas as pd
import torch

from src.data_handling.time_series_data_structure import TimeSeriesData
from src.synthetic_generation.constants import BASE_END_ORD, BASE_START_ORD


def generate_step_batch(
    batch_size, seq_len, pred_len, step_config=None
) -> TimeSeriesData:
    """
    Generates a batch of periodic step function data with seasonalities and trends.

    Args:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Total length of each time series sample (history + target).
        pred_len (int): Length of the target sequence to predict.
        step_config (dict, optional): Configuration for step function parameters.
                                      Keys: 'num_cycles_range', 'height_range', 'trend_prob', 'trend_max_change'.

    Returns:
        TimeSeriesData: A dataclass instance containing the batch tensors.
    """
    if step_config is None:
        step_config = {
            "num_cycles_range": (5, 10),
            "height_range": (-2.0, 2.0),
            "trend_prob": 0.5,
            "trend_max_change": 4.0,
        }

    batch_ts_features = np.zeros((batch_size, seq_len, 7), dtype=np.int64)
    batch_values = np.zeros(
        (batch_size, seq_len, 1), dtype=np.float32
    )  # 3D with 1 feature

    for i in range(batch_size):
        min_cycles, max_cycles = step_config["num_cycles_range"]
        min_period = max(2, seq_len // max_cycles)
        max_period = max(min_period, seq_len // min_cycles)
        period = np.random.randint(min_period, max_period + 1)

        steps_per_period = np.random.randint(2, 6)
        step_duration = period // steps_per_period
        if step_duration < 1:
            step_duration = 1
            steps_per_period = period

        step_heights = np.random.uniform(
            step_config["height_range"][0],
            step_config["height_range"][1],
            steps_per_period,
        )

        time_idx = np.arange(seq_len)
        period_indices = (time_idx // step_duration) % steps_per_period
        values = step_heights[period_indices]

        if np.random.rand() < step_config["trend_prob"]:
            max_change = step_config["trend_max_change"]
            slope = np.random.uniform(-max_change, max_change) / seq_len
            trend = slope * time_idx
            values += trend

        batch_values[i, :, 0] = values  # Assign to feature dimension 0

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
    history_ts = batch_ts_features[:, :history_len, :]
    history_values = batch_values[:, :history_len, :]
    target_ts = batch_ts_features[:, history_len:, :]
    target_values = batch_values[:, history_len:, :]
    task = torch.zeros(batch_size, dtype=torch.int64)

    return TimeSeriesData(
        history_ts=torch.from_numpy(history_ts),
        history_values=torch.from_numpy(history_values),
        target_ts=torch.from_numpy(target_ts),
        target_values=torch.from_numpy(target_values),
        task=task,
    )
