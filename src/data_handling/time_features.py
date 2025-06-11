import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import time_features_from_frequency_str
from pandas.tseries.frequencies import to_offset

from src.synthetic_generation.common.constants import Frequency
from src.utils.utils import device


def map_frequency_to_gluonts(frequency: Frequency) -> str:
    """Map custom Frequency enum to GluonTS-compatible frequency string"""
    mapping = {
        Frequency.M: "ME",  # MonthEnd
        Frequency.W: "W",  # Weekly
        Frequency.D: "D",  # Daily
        Frequency.H: "h",  # Hourly
        Frequency.S: "s",  # Seconds
        Frequency.T5: "5min",
        Frequency.T10: "10min",
        Frequency.T15: "15min",
    }
    return mapping[frequency]


def compute_batch_time_features(
    start,
    history_length,
    target_length,
    batch_size,
    frequency,
    include_subday=False,
):
    """Compute time features using GluonTS feature engineering"""
    try:
        freq_str = map_frequency_to_gluonts(frequency)
        offset = to_offset(freq_str)

        # Convert start timestamps to pandas DatetimeIndex
        start_dates = pd.to_datetime(start)

        # Initialize feature arrays
        history_features = []
        target_features = []

        # Get GluonTS time features for this frequency
        time_features = time_features_from_frequency_str(freq_str)

        for i in range(batch_size):
            # Generate history timestamps
            hist_range = pd.date_range(
                start=start_dates[i], periods=history_length, freq=offset
            )

            # Generate target timestamps
            target_start = hist_range[-1] + offset
            targ_range = pd.date_range(
                start=target_start, periods=target_length, freq=offset
            )

            # Convert to PeriodIndex for feature calculation
            hist_periods = pd.period_range(
                start=hist_range[0], periods=history_length, freq=freq_str
            )
            targ_periods = pd.period_range(
                start=targ_range[0], periods=target_length, freq=freq_str
            )

            # Compute features
            hist_feats = np.stack(
                [feat(hist_periods) for feat in time_features], axis=-1
            )
            targ_feats = np.stack(
                [feat(targ_periods) for feat in time_features], axis=-1
            )

            history_features.append(hist_feats)
            target_features.append(targ_feats)

        # Convert to tensors
        history_tensor = torch.tensor(np.array(history_features), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(target_features), dtype=torch.float32)

        return history_tensor.to(device), target_tensor.to(device)

    except Exception as e:
        raise ValueError(
            f"Failed to compute time features for frequency {frequency}. "
            f"Original error: {str(e)}"
        ) from e
