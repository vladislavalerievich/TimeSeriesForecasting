import numpy as np
import pandas as pd
import torch


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
