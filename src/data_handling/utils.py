import numpy as np
import pandas as pd


def compute_time_features(
    timestamps,
    include_subday=False,
):
    """
    Compute comprehensive time features from timestamps.

    Parameters
    ----------
    timestamps : array-like
        Array of timestamps.
    include_subday : bool, optional
        Whether to include hour, minute, second features (default: False).

    Returns
    -------
    np.ndarray
        Array of time features with shape (len(ts), n_features).
    """
    ts = pd.to_datetime(timestamps)
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
    return features
