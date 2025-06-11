from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from packaging.version import Version
from pandas.tseries.frequencies import to_offset

from src.synthetic_generation.common.constants import Frequency
from src.utils.utils import device

# ===== GLUONTS TIME FEATURE FUNCTIONS =====
# Imported and adapted from gluonts.time_feature._base

TimeFeature = Callable[[pd.PeriodIndex], np.ndarray]


def _normalize(xs, num: float):
    """
    Scale values of ``xs`` to [-0.5, 0.5].
    """
    return np.asarray(xs) / (num - 1) - 0.5


def second_of_minute(index: pd.PeriodIndex) -> np.ndarray:
    """
    Second of minute encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.second, num=60)


def second_of_minute_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Second of minute encoded as zero-based index, between 0 and 59.
    """
    return np.asarray(index.second)


def minute_of_hour(index: pd.PeriodIndex) -> np.ndarray:
    """
    Minute of hour encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.minute, num=60)


def minute_of_hour_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Minute of hour encoded as zero-based index, between 0 and 59.
    """
    return np.asarray(index.minute)


def hour_of_day(index: pd.PeriodIndex) -> np.ndarray:
    """
    Hour of day encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.hour, num=24)


def hour_of_day_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Hour of day encoded as zero-based index, between 0 and 23.
    """
    return np.asarray(index.hour)


def day_of_week(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of week encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.dayofweek, num=7)


def day_of_week_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of week encoded as zero-based index, between 0 and 6.
    """
    return np.asarray(index.dayofweek)


def day_of_month(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of month encoded as value between [-0.5, 0.5]
    """
    # first day of month is `1`, thus we deduct one
    return _normalize(index.day - 1, num=31)


def day_of_month_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of month encoded as zero-based index, between 0 and 30.
    """
    return np.asarray(index.day) - 1


def day_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of year encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.dayofyear - 1, num=366)


def day_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Day of year encoded as zero-based index, between 0 and 365.
    """
    return np.asarray(index.dayofyear) - 1


def month_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as value between [-0.5, 0.5]
    """
    return _normalize(index.month - 1, num=12)


def month_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Month of year encoded as zero-based index, between 0 and 11.
    """
    return np.asarray(index.month) - 1


def week_of_year(index: pd.PeriodIndex) -> np.ndarray:
    """
    Week of year encoded as value between [-0.5, 0.5]
    """
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week
    return _normalize(week - 1, num=53)


def week_of_year_index(index: pd.PeriodIndex) -> np.ndarray:
    """
    Week of year encoded as zero-based index, between 0 and 52.
    """
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week
    return np.asarray(week) - 1


def norm_freq_str(freq_str: str) -> str:
    """Normalize frequency string for compatibility."""
    base_freq = freq_str.split("-")[0]

    # Handle pandas frequency changes
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        base_freq = base_freq[:-1]
        if Version(pd.__version__) >= Version("2.2.0"):
            base_freq += "E"

    return base_freq


# ===== FREQUENCY MAPPING =====
# Map our Frequency enum to gluonts-compatible frequency strings


def frequency_to_gluonts_str(frequency: Frequency) -> str:
    """Convert our Frequency enum to gluonts-compatible frequency string."""
    mapping = {
        Frequency.S: "s",
        Frequency.T5: "5min",
        Frequency.T10: "10min",
        Frequency.T15: "15min",
        Frequency.H: "h",
        Frequency.D: "D",
        Frequency.W: "W",
        Frequency.M: "M",
    }
    return mapping.get(frequency, "D")  # Default to daily


def time_features_from_frequency(
    frequency: Frequency, use_normalized: bool = True
) -> List[TimeFeature]:
    """
    Returns a list of time features appropriate for the given frequency.

    Parameters
    ----------
    frequency : Frequency
        Our custom Frequency enum
    use_normalized : bool
        If True, use normalized features ([-0.5, 0.5]). If False, use index features.

    Returns
    -------
    List[TimeFeature]
        List of time feature functions
    """
    freq_str = frequency_to_gluonts_str(frequency)

    # Choose between normalized and index features
    if use_normalized:
        features_by_freq = {
            "s": [
                second_of_minute,
                minute_of_hour,
                hour_of_day,
                day_of_week,
                day_of_month,
                day_of_year,
            ],
            "5min": [
                minute_of_hour,
                hour_of_day,
                day_of_week,
                day_of_month,
                day_of_year,
            ],
            "10min": [
                minute_of_hour,
                hour_of_day,
                day_of_week,
                day_of_month,
                day_of_year,
            ],
            "15min": [
                minute_of_hour,
                hour_of_day,
                day_of_week,
                day_of_month,
                day_of_year,
            ],
            "h": [hour_of_day, day_of_week, day_of_month, day_of_year],
            "D": [day_of_week, day_of_month, day_of_year],
            "W": [day_of_month, week_of_year],
            "M": [month_of_year],
        }
    else:
        features_by_freq = {
            "s": [
                second_of_minute_index,
                minute_of_hour_index,
                hour_of_day_index,
                day_of_week_index,
                day_of_month_index,
                day_of_year_index,
            ],
            "5min": [
                minute_of_hour_index,
                hour_of_day_index,
                day_of_week_index,
                day_of_month_index,
                day_of_year_index,
            ],
            "10min": [
                minute_of_hour_index,
                hour_of_day_index,
                day_of_week_index,
                day_of_month_index,
                day_of_year_index,
            ],
            "15min": [
                minute_of_hour_index,
                hour_of_day_index,
                day_of_week_index,
                day_of_month_index,
                day_of_year_index,
            ],
            "h": [
                hour_of_day_index,
                day_of_week_index,
                day_of_month_index,
                day_of_year_index,
            ],
            "D": [day_of_week_index, day_of_month_index, day_of_year_index],
            "W": [day_of_month_index, week_of_year_index],
            "M": [month_of_year_index],
        }

    return features_by_freq.get(freq_str, features_by_freq["D"])  # Default to daily


# ===== SEASONALITY DETECTION =====
# Adapted from gluonts.time_feature.seasonality

DEFAULT_SEASONALITIES = {
    "s": 3600,  # 1 hour
    "5min": 288,  # 1 day (24*60/5)
    "10min": 144,  # 1 day (24*60/10)
    "15min": 96,  # 1 day (24*60/15)
    "h": 24,  # 1 day
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,  # 1 year
}


def get_seasonality(frequency: Frequency) -> int:
    """
    Return the seasonality for a given frequency.

    Parameters
    ----------
    frequency : Frequency
        Our custom Frequency enum

    Returns
    -------
    int
        Seasonality value
    """
    freq_str = frequency_to_gluonts_str(frequency)
    return DEFAULT_SEASONALITIES.get(freq_str, 1)


# ===== MAIN TIME FEATURE COMPUTATION FUNCTIONS =====


def compute_batch_time_features(
    start,
    history_length,
    target_length,
    batch_size,
    frequency,
    use_gluonts_features=True,
    use_normalized_features=True,
):
    """
    Compute time features from start timestamps and frequency using gluonts-style features.

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
        Our custom Frequency enum.
    use_gluonts_features : bool, optional
        If True, use gluonts-style normalized features. If False, use legacy features.
    use_normalized_features : bool, optional
        If True, use normalized features ([-0.5, 0.5]). If False, use index features.

    Returns
    -------
    tuple
        (history_time_features, target_time_features) where each is a torch.Tensor
        of shape (batch_size, length, n_features).
    """
    start = np.asarray(start)

    if use_gluonts_features:
        # Use gluonts-style time features
        return _compute_gluonts_time_features(
            start,
            history_length,
            target_length,
            batch_size,
            frequency,
            use_normalized_features,
        )
    else:
        # Fall back to legacy features for backward compatibility
        return _compute_legacy_time_features(
            start, history_length, target_length, batch_size, frequency
        )


def _compute_gluonts_time_features(
    start, history_length, target_length, batch_size, frequency, use_normalized_features
):
    """Compute time features using gluonts approach."""
    freq_str = frequency_to_gluonts_str(frequency)
    time_feature_functions = time_features_from_frequency(
        frequency, use_normalized_features
    )

    # Generate timestamps for history and target sequences
    history_features_list = []
    target_features_list = []

    for i in range(batch_size):
        # Generate datetime range for history
        hist_range = pd.date_range(
            start=pd.Timestamp(start[i]),
            periods=history_length,
            freq=freq_str,
        )

        # Generate datetime range for target (continuing from history)
        target_start = hist_range[-1] + pd.tseries.frequencies.to_offset(freq_str)
        targ_range = pd.date_range(
            start=target_start,
            periods=target_length,
            freq=freq_str,
        )

        # Convert to PeriodIndex for gluonts time features
        hist_period_idx = hist_range.to_period(freq=freq_str)
        targ_period_idx = targ_range.to_period(freq=freq_str)

        # Compute time features using gluonts functions
        hist_features = np.stack(
            [time_feat(hist_period_idx) for time_feat in time_feature_functions],
            axis=-1,
        )

        targ_features = np.stack(
            [time_feat(targ_period_idx) for time_feat in time_feature_functions],
            axis=-1,
        )

        history_features_list.append(hist_features)
        target_features_list.append(targ_features)

    # Stack all batch items
    history_features = np.stack(history_features_list, axis=0)
    target_features = np.stack(target_features_list, axis=0)

    return (
        torch.from_numpy(history_features).float().to(device),
        torch.from_numpy(target_features).float().to(device),
    )


def _compute_legacy_time_features(
    start, history_length, target_length, batch_size, frequency
):
    """Compute legacy time features for backward compatibility."""
    # Generate timestamps for history and target sequences
    history_timestamps = np.zeros((batch_size, history_length), dtype=np.int64)
    target_timestamps = np.zeros((batch_size, target_length), dtype=np.int64)

    freq_value = frequency.value
    offset = to_offset(freq_value)

    for i in range(batch_size):
        hist_range = pd.date_range(
            start=pd.Timestamp(start[i]),
            periods=history_length,
            freq=freq_value,
        )
        target_start = hist_range[-1] + offset
        targ_range = pd.date_range(
            start=target_start,
            periods=target_length,
            freq=freq_value,
        )
        history_timestamps[i, :] = hist_range.astype(np.int64)
        target_timestamps[i, :] = targ_range.astype(np.int64)

    # Compute time features for both sequences
    history_time_features = compute_time_features(
        history_timestamps, include_subday=True
    )
    target_time_features = compute_time_features(target_timestamps, include_subday=True)

    return history_time_features.to(device), target_time_features.to(device)


def compute_time_features(timestamps, include_subday=False):
    """
    Legacy time feature computation for backward compatibility.

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

    n_features = features.shape[-1]
    features = features.reshape(batch_size, length, n_features)
    return torch.from_numpy(features).long()


# ===== ENHANCED TIME FEATURES =====


def compute_enhanced_time_features(
    start,
    history_length,
    target_length,
    batch_size,
    frequency,
    include_seasonality=True,
    include_cyclical_encoding=True,
):
    """
    Compute enhanced time features including seasonality and cyclical encodings.

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
        Our custom Frequency enum.
    include_seasonality : bool, optional
        Whether to include seasonality features.
    include_cyclical_encoding : bool, optional
        Whether to include cyclical sine/cosine encodings.

    Returns
    -------
    tuple
        (history_time_features, target_time_features) with enhanced features.
    """
    # Get base gluonts features
    hist_feats, targ_feats = compute_batch_time_features(
        start,
        history_length,
        target_length,
        batch_size,
        frequency,
        use_gluonts_features=True,
        use_normalized_features=True,
    )

    enhanced_hist_feats = [hist_feats]
    enhanced_targ_feats = [targ_feats]

    if include_seasonality:
        # Add seasonality features
        seasonality = get_seasonality(frequency)

        # Create seasonal position features
        hist_seasonal = (
            torch.arange(history_length).float().unsqueeze(0).repeat(batch_size, 1)
        )
        hist_seasonal = (
            hist_seasonal % seasonality
        ) / seasonality  # Normalize to [0, 1]
        hist_seasonal = hist_seasonal.unsqueeze(-1).to(device)

        targ_seasonal = (
            torch.arange(history_length, history_length + target_length)
            .float()
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        targ_seasonal = (targ_seasonal % seasonality) / seasonality
        targ_seasonal = targ_seasonal.unsqueeze(-1).to(device)

        enhanced_hist_feats.append(hist_seasonal)
        enhanced_targ_feats.append(targ_seasonal)

    if include_cyclical_encoding:
        # Add cyclical sine/cosine encodings for key time components
        freq_str = frequency_to_gluonts_str(frequency)

        # Add different cyclical encodings based on frequency
        if freq_str in ["s", "5min", "10min", "15min", "h"]:
            # Hour of day cyclical encoding
            hour_cycle = (
                2 * np.pi * hist_feats[:, :, -4] if hist_feats.shape[-1] > 4 else None
            )
            if hour_cycle is not None:
                hist_hour_sin = torch.sin(hour_cycle * 2 * np.pi).unsqueeze(-1)
                hist_hour_cos = torch.cos(hour_cycle * 2 * np.pi).unsqueeze(-1)
                enhanced_hist_feats.extend([hist_hour_sin, hist_hour_cos])

                targ_hour_cycle = (
                    2 * np.pi * targ_feats[:, :, -4]
                    if targ_feats.shape[-1] > 4
                    else None
                )
                if targ_hour_cycle is not None:
                    targ_hour_sin = torch.sin(targ_hour_cycle * 2 * np.pi).unsqueeze(-1)
                    targ_hour_cos = torch.cos(targ_hour_cycle * 2 * np.pi).unsqueeze(-1)
                    enhanced_targ_feats.extend([targ_hour_sin, targ_hour_cos])

        # Day of week cyclical encoding (for all frequencies)
        dow_idx = (
            -3 if hist_feats.shape[-1] > 3 else 0
        )  # Day of week is typically 3rd from end
        if hist_feats.shape[-1] > abs(dow_idx):
            dow_cycle = hist_feats[:, :, dow_idx]
            hist_dow_sin = torch.sin(dow_cycle * 2 * np.pi).unsqueeze(-1)
            hist_dow_cos = torch.cos(dow_cycle * 2 * np.pi).unsqueeze(-1)
            enhanced_hist_feats.extend([hist_dow_sin, hist_dow_cos])

            targ_dow_cycle = targ_feats[:, :, dow_idx]
            targ_dow_sin = torch.sin(targ_dow_cycle * 2 * np.pi).unsqueeze(-1)
            targ_dow_cos = torch.cos(targ_dow_cycle * 2 * np.pi).unsqueeze(-1)
            enhanced_targ_feats.extend([targ_dow_sin, targ_dow_cos])

    # Concatenate all features
    final_hist_feats = torch.cat(enhanced_hist_feats, dim=-1)
    final_targ_feats = torch.cat(enhanced_targ_feats, dim=-1)

    return final_hist_feats, final_targ_feats
