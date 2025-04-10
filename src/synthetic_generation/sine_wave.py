from datetime import date

import numpy as np
import pandas as pd
import torch

from src.data_handling.time_series_data_structure import TimeSeriesData
from src.synthetic_generation.constants import BASE_END_ORD, BASE_START_ORD


def generate_sine_batch(
    batch_size, seq_len, pred_len, sine_config=None
) -> TimeSeriesData:
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
                                      Expected keys: 'period_range', 'amp_range', 'phase_range'.

    Returns:
        TimeSeriesData: A dataclass instance containing the batch tensors.
    """
    if sine_config is None:
        sine_config = {
            "period_range": (10, 100),
            "amp_range": (0.5, 3.0),
            "phase_range": (0, 2 * np.pi),
        }

    batch_ts_features = np.zeros(
        (batch_size, seq_len, 7), dtype=np.int64
    )  # 7 date features
    batch_values = np.zeros(
        (batch_size, seq_len, 1), dtype=np.float32
    )  # 1 feature (univariate)

    for i in range(batch_size):
        period = np.random.uniform(
            sine_config["period_range"][0], sine_config["period_range"][1]
        )
        amplitude = np.random.uniform(
            sine_config["amp_range"][0], sine_config["amp_range"][1]
        )
        phase = np.random.uniform(
            sine_config["phase_range"][0], sine_config["phase_range"][1]
        )

        time_idx = np.arange(seq_len)
        values = amplitude * np.sin(2 * np.pi * time_idx / period + phase)
        values += np.random.normal(0, amplitude * 0.1)
        batch_values[i, :, 0] = values  # Assign to feature dimension

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
    task = torch.zeros(
        batch_size, dtype=torch.int64
    )  # [batch_size] instead of [batch_size, pred_len]

    return TimeSeriesData(
        history_ts=torch.from_numpy(history_ts),
        history_values=torch.from_numpy(history_values),
        target_ts=torch.from_numpy(target_ts),
        target_values=torch.from_numpy(target_values),
        task=task,
    )
