from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature._base import (
    day_of_month,
    day_of_month_index,
    day_of_week,
    day_of_week_index,
    day_of_year,
    hour_of_day,
    hour_of_day_index,
    minute_of_hour,
    minute_of_hour_index,
    month_of_year,
    month_of_year_index,
    second_of_minute,
    second_of_minute_index,
    week_of_year,
    week_of_year_index,
)
from gluonts.time_feature.holiday import (
    BLACK_FRIDAY,
    CHRISTMAS_DAY,
    CHRISTMAS_EVE,
    CYBER_MONDAY,
    EASTER_MONDAY,
    EASTER_SUNDAY,
    GOOD_FRIDAY,
    INDEPENDENCE_DAY,
    LABOR_DAY,
    MEMORIAL_DAY,
    NEW_YEARS_DAY,
    NEW_YEARS_EVE,
    THANKSGIVING,
    SpecialDateFeatureSet,
    exponential_kernel,
    squared_exponential_kernel,
)
from gluonts.time_feature.seasonality import get_seasonality

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

PERIOD_FREQUENCY_STR_MAPPING = FREQUENCY_STR_MAPPING.copy()
PERIOD_FREQUENCY_STR_MAPPING[Frequency.M] = "M"

# Enhanced feature sets for different frequencies
ENHANCED_TIME_FEATURES = {
    # High-frequency features (seconds, minutes)
    "high_freq": {
        "normalized": [
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
        ],
        "index": [
            second_of_minute_index,
            minute_of_hour_index,
            hour_of_day_index,
            day_of_week_index,
        ],
    },
    # Medium-frequency features (hourly, daily)
    "medium_freq": {
        "normalized": [
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
            month_of_year,
        ],
        "index": [
            hour_of_day_index,
            day_of_week_index,
            day_of_month_index,
            week_of_year_index,
        ],
    },
    # Low-frequency features (weekly, monthly)
    "low_freq": {
        "normalized": [day_of_week, day_of_month, month_of_year, week_of_year],
        "index": [day_of_week_index, month_of_year_index, week_of_year_index],
    },
}

# Holiday features for different markets/regions
HOLIDAY_FEATURE_SETS = {
    "us_business": [
        NEW_YEARS_DAY,
        MEMORIAL_DAY,
        INDEPENDENCE_DAY,
        LABOR_DAY,
        THANKSGIVING,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE,
    ],
    "us_retail": [
        NEW_YEARS_DAY,
        EASTER_SUNDAY,
        MEMORIAL_DAY,
        INDEPENDENCE_DAY,
        LABOR_DAY,
        THANKSGIVING,
        BLACK_FRIDAY,
        CYBER_MONDAY,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE,
    ],
    "christian": [
        NEW_YEARS_DAY,
        GOOD_FRIDAY,
        EASTER_SUNDAY,
        EASTER_MONDAY,
        CHRISTMAS_EVE,
        CHRISTMAS_DAY,
        NEW_YEARS_EVE,
    ],
}


class TimeFeatureGenerator:
    """
    Enhanced time feature generator that leverages full GluonTS capabilities.
    """

    def __init__(
        self,
        use_enhanced_features: bool = True,
        use_holiday_features: bool = True,
        holiday_set: str = "us_business",
        holiday_kernel: str = "exponential",
        holiday_kernel_alpha: float = 1.0,
        use_index_features: bool = True,
        k_max: Union[int, Tuple[int, int]] = 15,
        auto_adjust_k_max: bool = False,
        include_seasonality_info: bool = True,
    ):
        """
        Initialize enhanced time feature generator.

        Parameters
        ----------
        use_enhanced_features : bool
            Whether to use frequency-specific enhanced features
        use_holiday_features : bool
            Whether to include holiday features
        holiday_set : str
            Which holiday set to use ('us_business', 'us_retail', 'christian')
        holiday_kernel : str
            Holiday kernel type ('indicator', 'exponential', 'squared_exponential')
        holiday_kernel_alpha : float
            Kernel parameter for exponential kernels
        use_index_features : bool
            Whether to include index-based features alongside normalized ones
        k_max : Union[int, Tuple[int, int]]
            Either an integer specifying K_max directly, or a tuple (min_k_max, max_k_max)
            specifying the range for automatic adjustment
        auto_adjust_k_max : bool
            Whether to automatically adjust K_max based on available features
        include_seasonality_info : bool
            Whether to include seasonality information as features
        """
        self.use_enhanced_features = use_enhanced_features
        self.use_holiday_features = use_holiday_features
        self.holiday_set = holiday_set
        self.use_index_features = use_index_features
        self.k_max = k_max
        self.auto_adjust_k_max = auto_adjust_k_max
        self.include_seasonality_info = include_seasonality_info

        # Initialize holiday feature set
        self.holiday_feature_set = None
        if use_holiday_features and holiday_set in HOLIDAY_FEATURE_SETS:
            kernel_func = self._get_holiday_kernel(holiday_kernel, holiday_kernel_alpha)
            self.holiday_feature_set = SpecialDateFeatureSet(
                HOLIDAY_FEATURE_SETS[holiday_set], kernel_func
            )

    def _get_holiday_kernel(self, kernel_type: str, alpha: float):
        """Get holiday kernel function."""
        if kernel_type == "exponential":
            return exponential_kernel(alpha)
        elif kernel_type == "squared_exponential":
            return squared_exponential_kernel(alpha)
        else:
            # Default indicator kernel
            return lambda x: float(x == 0)

    def _get_feature_category(self, freq_str: str) -> str:
        """Determine feature category based on frequency."""
        if freq_str in ["s", "1min", "5min", "10min", "15min"]:
            return "high_freq"
        elif freq_str in ["h", "D"]:
            return "medium_freq"
        else:
            return "low_freq"

    def _compute_enhanced_features(
        self, period_index: pd.PeriodIndex, freq_str: str
    ) -> np.ndarray:
        """Compute enhanced time features based on frequency."""
        if not self.use_enhanced_features:
            return np.array([]).reshape(len(period_index), 0)

        category = self._get_feature_category(freq_str)
        feature_config = ENHANCED_TIME_FEATURES[category]

        features = []

        # Add normalized features
        for feat_func in feature_config["normalized"]:
            try:
                feat_values = feat_func(period_index)
                features.append(feat_values)
            except Exception:
                continue

        # Add index features if enabled
        if self.use_index_features:
            for feat_func in feature_config["index"]:
                try:
                    feat_values = feat_func(period_index)
                    # Normalize index features to [0, 1] range
                    if feat_values.max() > 0:
                        feat_values = feat_values / feat_values.max()
                    features.append(feat_values)
                except Exception:
                    continue

        if features:
            return np.stack(features, axis=-1)
        else:
            return np.array([]).reshape(len(period_index), 0)

    def _compute_holiday_features(self, date_range: pd.DatetimeIndex) -> np.ndarray:
        """Compute holiday features."""
        if not self.use_holiday_features or self.holiday_feature_set is None:
            return np.array([]).reshape(len(date_range), 0)

        try:
            holiday_features = self.holiday_feature_set(date_range)
            return holiday_features.T  # Transpose to get [time, features] shape
        except Exception:
            return np.array([]).reshape(len(date_range), 0)

    def _compute_seasonality_features(
        self, period_index: pd.PeriodIndex, freq_str: str
    ) -> np.ndarray:
        """Compute seasonality-aware features."""
        if not self.include_seasonality_info:
            return np.array([]).reshape(len(period_index), 0)

        try:
            seasonality = get_seasonality(freq_str)
            if seasonality > 1:
                # Create sinusoidal seasonality features
                positions = np.arange(len(period_index))
                sin_feat = np.sin(2 * np.pi * positions / seasonality)
                cos_feat = np.cos(2 * np.pi * positions / seasonality)
                return np.stack([sin_feat, cos_feat], axis=-1)
        except Exception:
            pass

        return np.array([]).reshape(len(period_index), 0)

    def compute_features(
        self, period_index: pd.PeriodIndex, date_range: pd.DatetimeIndex, freq_str: str
    ) -> np.ndarray:
        """
        Compute all time features for given period index.

        Parameters
        ----------
        period_index : pd.PeriodIndex
            Period index for computing features
        date_range : pd.DatetimeIndex
            Corresponding datetime index for holiday features
        freq_str : str
            Frequency string

        Returns
        -------
        np.ndarray
            Time features array of shape [time_steps, num_features]
        """
        all_features = []

        # Standard GluonTS features
        try:
            standard_features = time_features_from_frequency_str(freq_str)
            if standard_features:
                std_feat = np.stack(
                    [feat(period_index) for feat in standard_features], axis=-1
                )
                all_features.append(std_feat)
        except Exception:
            pass

        # Enhanced features
        enhanced_feat = self._compute_enhanced_features(period_index, freq_str)
        if enhanced_feat.shape[1] > 0:
            all_features.append(enhanced_feat)

        # Holiday features
        holiday_feat = self._compute_holiday_features(date_range)
        if holiday_feat.shape[1] > 0:
            all_features.append(holiday_feat)

        # Seasonality features
        seasonality_feat = self._compute_seasonality_features(period_index, freq_str)
        if seasonality_feat.shape[1] > 0:
            all_features.append(seasonality_feat)

        if all_features:
            combined_features = np.concatenate(all_features, axis=-1)
        else:
            combined_features = np.zeros((len(period_index), 1))

        return combined_features

    def get_optimal_k_max(self, freq_str: str) -> int:
        """Determine optimal K_max based on frequency and enabled features."""
        if not self.auto_adjust_k_max:
            # If k_max is an integer, return it directly
            if isinstance(self.k_max, int):
                return self.k_max
            # If k_max is a tuple, return the minimum value as default
            elif isinstance(self.k_max, tuple):
                return self.k_max[0]
            else:
                return 6  # fallback default

        # Get min and max values from k_max
        if isinstance(self.k_max, int):
            min_k_max = max_k_max = self.k_max
        elif isinstance(self.k_max, tuple) and len(self.k_max) == 2:
            min_k_max, max_k_max = self.k_max
        else:
            min_k_max, max_k_max = 6, 20  # fallback defaults

        # Estimate number of features
        base_features = len(time_features_from_frequency_str(freq_str) or [])

        enhanced_count = 0
        if self.use_enhanced_features:
            category = self._get_feature_category(freq_str)
            enhanced_count += len(ENHANCED_TIME_FEATURES[category]["normalized"])
            if self.use_index_features:
                enhanced_count += len(ENHANCED_TIME_FEATURES[category]["index"])

        holiday_count = 0
        if self.use_holiday_features and self.holiday_set in HOLIDAY_FEATURE_SETS:
            holiday_count = len(HOLIDAY_FEATURE_SETS[self.holiday_set])

        seasonality_count = 2 if self.include_seasonality_info else 0

        total_estimated = (
            base_features + enhanced_count + holiday_count + seasonality_count
        )

        return min(max(total_estimated, min_k_max), max_k_max)


def compute_batch_time_features(
    start,
    history_length,
    target_length,
    batch_size,
    frequency,
    K_max=6,
    time_feature_config: Optional[Dict[str, Any]] = None,
):
    """
    Compute enhanced time features from start timestamps and frequency.

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
    time_feature_config : dict, optional
        Configuration for enhanced time features.

    Returns
    -------
    tuple
        (history_time_features, target_time_features) where each is a torch.Tensor
        of shape (batch_size, length, K_max).
    """
    freq_str = FREQUENCY_STR_MAPPING[frequency]
    period_freq_str = PERIOD_FREQUENCY_STR_MAPPING[frequency]

    # Initialize enhanced feature generator
    feature_config = time_feature_config or {}
    feature_generator = TimeFeatureGenerator(**feature_config)

    # Auto-adjust K_max if enabled
    if feature_generator.auto_adjust_k_max:
        K_max = feature_generator.get_optimal_k_max(freq_str)

    # Define safe timestamp bounds
    min_timestamp = pd.Timestamp("1900-01-01")
    max_timestamp = pd.Timestamp("2200-12-31")

    # Generate timestamps and features
    history_features_list = []
    future_features_list = []

    for i in range(batch_size):
        try:
            # Validate start timestamp is within safe bounds
            start_ts = pd.Timestamp(start[i])
            if start_ts < min_timestamp or start_ts > max_timestamp:
                start_ts = pd.Timestamp("2000-01-01")

            # Create history range with bounds checking
            history_range = pd.date_range(
                start=start_ts, periods=history_length, freq=freq_str
            )

            # Check if history range goes beyond safe bounds
            if history_range[-1] > max_timestamp:
                safe_start = max_timestamp - pd.tseries.frequencies.to_offset(
                    freq_str
                ) * (history_length + target_length)
                if safe_start < min_timestamp:
                    safe_start = min_timestamp
                history_range = pd.date_range(
                    start=safe_start, periods=history_length, freq=freq_str
                )

            future_start = history_range[-1] + pd.tseries.frequencies.to_offset(
                freq_str
            )
            future_range = pd.date_range(
                start=future_start, periods=target_length, freq=freq_str
            )

            # Final bounds check
            if future_range[-1] > max_timestamp:
                safe_start = pd.Timestamp("2000-01-01")
                history_range = pd.date_range(
                    start=safe_start, periods=history_length, freq=freq_str
                )
                future_start = history_range[-1] + pd.tseries.frequencies.to_offset(
                    freq_str
                )
                future_range = pd.date_range(
                    start=future_start, periods=target_length, freq=freq_str
                )

        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
            # Fallback to safe default timestamps
            safe_start = pd.Timestamp("2000-01-01")
            history_range = pd.date_range(
                start=safe_start, periods=history_length, freq=freq_str
            )
            future_start = history_range[-1] + pd.tseries.frequencies.to_offset(
                freq_str
            )
            future_range = pd.date_range(
                start=future_start, periods=target_length, freq=freq_str
            )

        # Convert to period indices
        history_period_idx = history_range.to_period(period_freq_str)
        future_period_idx = future_range.to_period(period_freq_str)

        # Compute enhanced features
        history_features = feature_generator.compute_features(
            history_period_idx, history_range, freq_str
        )
        future_features = feature_generator.compute_features(
            future_period_idx, future_range, freq_str
        )

        # Pad or truncate to K_max
        history_features = _pad_or_truncate_features(history_features, K_max)
        future_features = _pad_or_truncate_features(future_features, K_max)

        history_features_list.append(history_features)
        future_features_list.append(future_features)

    # Stack into batch tensors
    history_time_features = np.stack(history_features_list, axis=0)
    target_time_features = np.stack(future_features_list, axis=0)

    return (
        torch.from_numpy(history_time_features).float().to(device),
        torch.from_numpy(target_time_features).float().to(device),
    )


def _pad_or_truncate_features(features: np.ndarray, K_max: int) -> np.ndarray:
    """Pad with zeros or truncate features to K_max dimensions."""
    seq_len, num_features = features.shape

    if num_features < K_max:
        # Pad with zeros
        padding = np.zeros((seq_len, K_max - num_features))
        features = np.concatenate([features, padding], axis=-1)
    elif num_features > K_max:
        # Truncate to K_max (keep most important features first)
        features = features[:, :K_max]

    return features
