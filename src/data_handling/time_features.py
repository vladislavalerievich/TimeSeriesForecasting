import numpy as np
import pandas as pd
import torch

from src.data_handling.data_containers import Frequency
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
    """
    # Convert start to numpy array if it's not already
    start = np.asarray(start)

    # Generate timestamps for history and target sequences
    history_timestamps = pd.date_range(
        start=pd.Timestamp(start[0]),
        periods=history_length,
        freq=frequency,
    ).astype(np.int64)
    target_timestamps = pd.date_range(
        start=pd.Timestamp(start[0]) + pd.Timedelta(history_length, unit=frequency),
        periods=target_length,
        freq=frequency,
    ).astype(np.int64)

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
