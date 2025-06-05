import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset

from src.utils.utils import device


def compute_batch_time_features(
    start,
    history_length,
    target_length,
    batch_size,
    frequency,
    include_subday=False,
):
    """
    Compute time features from start timestamps and frequency.

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
    frequency : str or Frequency
        Frequency of the time series (e.g., 'H' for hourly, 'D' for daily).
    include_subday : bool, optional
        Whether to include hour, minute, second features (default: False).

    Returns
    -------
    tuple
        (history_time_features, target_time_features) where each is a torch.Tensor
        of shape (batch_size, length, n_features).

    Notes
    -----
    Assumption: For multivariate time series, all channels (variables) within a single series
    share the same timestamps and frequency. That is, the time grid is identical for all channels
    in a given series. Time features are therefore computed per series, not per channel.
    """
    # Convert start to numpy array if it's not already
    start = np.asarray(start)

    # Generate timestamps for history and target sequences for each batch item
    history_timestamps = np.zeros((batch_size, history_length), dtype=np.int64)
    target_timestamps = np.zeros((batch_size, target_length), dtype=np.int64)

    freq_value = frequency.value
    offset = to_offset(freq_value)

    for i in range(batch_size):
        # Each start[i] is a np.datetime64
        hist_range = pd.date_range(
            start=pd.Timestamp(start[i]),
            periods=history_length,
            freq=freq_value,
        )
        # For target, start at the next time step after the last history timestamp
        target_start = hist_range[-1] + offset
        targ_range = pd.date_range(
            start=target_start,
            periods=target_length,
            freq=freq_value,
        )
        history_timestamps[i, :] = hist_range.astype(np.int64)
        target_timestamps[i, :] = targ_range.astype(np.int64)

    # Compute time features for both sequences
    history_time_features = compute_time_features(history_timestamps, include_subday)
    target_time_features = compute_time_features(target_timestamps, include_subday)

    return history_time_features.to(device), target_time_features.to(device)


# TODO: Move to gluonts https://github.com/awslabs/gluonts/tree/dev/src/gluonts/time_feature
def compute_time_features(
    timestamps,
    include_subday=False,
):
    """
    Compute comprehensive time features from timestamps.

    Parameters
    ----------
    timestamps : array-like, shape (batch_size, length)
        2D array of timestamps.
    include_subday : bool, optional
        Whether to include hour, minute, second features (default: False).

    Returns
    -------
    torch.Tensor
        Tensor of time features with shape (batch_size, length, n_features).
    """
    timestamps = np.asarray(timestamps)
    batch_size, length = timestamps.shape
    # Flatten to 1D for vectorized processing
    flat_timestamps = timestamps.reshape(-1)
    ts = pd.to_datetime(flat_timestamps)
    if include_subday:
        features = np.stack(
            [
                ts.year.values,
                ts.month.values,
                ts.day.values,
                ts.day_of_week.values + 1,
                ts.day_of_year.values,
                ts.hour.values,
                ts.minute.values,
                ts.second.values,
            ],
            axis=-1,
        )
    else:
        features = np.stack(
            [
                ts.year.values,
                ts.month.values,
                ts.day.values,
                ts.day_of_week.values + 1,
                ts.day_of_year.values,
            ],
            axis=-1,
        )
    # Reshape back to (batch_size, length, n_features)
    n_features = features.shape[-1]
    features = features.reshape(batch_size, length, n_features)
    # Convert to torch tensor (use long for integer features)
    return torch.from_numpy(features).long()
