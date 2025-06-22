import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import time_features_from_frequency_str

from src.data_handling.data_containers import Frequency
from src.utils.utils import device

FREQUENCY_STR_MAPPING = {
    Frequency.A: "A",  # Annual
    Frequency.Q: "Q",  # Quarterly
    Frequency.M: "ME",  # Monthly (Month End for pandas compatibility)
    Frequency.W: "W",  # Weekly
    Frequency.D: "D",  # Daily
    Frequency.H: "h",  # Hourly
    Frequency.S: "s",  # Seconds
    Frequency.T1: "1min",  # 1 minute
    Frequency.T5: "5min",  # 5 minutes
    Frequency.T10: "10min",  # 10 minutes
    Frequency.T15: "15min",  # 15 minutes
}

PERIOD_FREQUENCY_STR_MAPPING = {
    Frequency.A: "A",  # Annual
    Frequency.Q: "Q",  # Quarterly
    Frequency.M: "M",  # Monthly
    Frequency.W: "W",  # Weekly
    Frequency.D: "D",  # Daily
    Frequency.H: "h",  # Hourly
    Frequency.S: "s",  # Seconds
    Frequency.T1: "1min",  # 1 minute
    Frequency.T5: "5min",  # 5 minutes
    Frequency.T10: "10min",  # 10 minutes
    Frequency.T15: "15min",  # 15 minutes
}


def compute_batch_time_features(
    start,
    history_length,
    target_length,
    batch_size,
    frequency,
    K_max=6,
):
    """
    Compute time features from start timestamps and frequency using GluonTS.

    Parameters
    ----------
    start : array-like, shape (batch_size,)
        Start timestamps for each batch item.
    history_length : int
        Length of history sequence.
    target_length : int
        Length of target sequence.
    batch_size : int
        Batch size.
    frequency : Frequency
        Frequency of the time series.
    K_max : int, optional
        Maximum number of time features to pad to (default: 6).

    Returns
    -------
    tuple
        (history_time_features, target_time_features) where each is a torch.Tensor
        of shape (batch_size, length, K_max).
    """
    freq_str = FREQUENCY_STR_MAPPING[frequency]
    period_freq_str = PERIOD_FREQUENCY_STR_MAPPING[frequency]
    time_features = time_features_from_frequency_str(freq_str)
    num_features = len(time_features)

    # Generate timestamps and convert to PeriodIndex
    history_indices = []
    future_indices = []
    for i in range(batch_size):
        hist_range = pd.date_range(
            start=start[i], periods=history_length, freq=freq_str
        )
        target_start = hist_range[-1] + pd.tseries.frequencies.to_offset(freq_str)
        targ_range = pd.date_range(
            start=target_start, periods=target_length, freq=freq_str
        )
        history_indices.append(hist_range.to_period(period_freq_str))
        future_indices.append(targ_range.to_period(period_freq_str))

    # Compute features for history
    history_features_list = []
    for idx in history_indices:
        features = [feat(idx) for feat in time_features]
        features = np.stack(features, axis=-1)  # [seq_len, num_features]
        if num_features < K_max:
            padding = np.zeros((history_length, K_max - num_features))
            features = np.concatenate([features, padding], axis=-1)
        history_features_list.append(features)
    history_time_features = np.stack(
        history_features_list, axis=0
    )  # [batch_size, seq_len, K_max]

    # Compute features for target
    target_features_list = []
    for idx in future_indices:
        features = [feat(idx) for feat in time_features]
        features = np.stack(features, axis=-1)  # [pred_len, num_features]
        if num_features < K_max:
            padding = np.zeros((target_length, K_max - num_features))
            features = np.concatenate([features, padding], axis=-1)
        target_features_list.append(features)
    target_time_features = np.stack(
        target_features_list, axis=0
    )  # [batch_size, pred_len, K_max]

    return (
        torch.from_numpy(history_time_features).float().to(device),
        torch.from_numpy(target_time_features).float().to(device),
    )
