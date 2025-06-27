from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.random import Generator
from pandas.tseries.frequencies import to_offset

from src.data_handling.data_containers import Frequency
from src.synthetic_generation.common.constants import BASE_END_DATE, BASE_START_DATE

# Cap the total length for low-frequency series to prevent timestamp overflow
# Pandas datetime range is roughly 1677-2262, so we need conservative limits

MAX_YEARS = 99

# Maximum sequence lengths to avoid pandas OutOfBoundsDatetime errors
SHORT_FREQUENCY_MAX_LENGTHS = {
    Frequency.A: MAX_YEARS,
    Frequency.Q: MAX_YEARS * 4,
    Frequency.M: MAX_YEARS * 12,
    Frequency.W: int(MAX_YEARS * 52.1775),
    Frequency.D: int(MAX_YEARS * 365.2425),
}

# Additional constraints for high-frequency data to prevent unrealistic scenarios
HIGH_FREQUENCY_MAX_LENGTHS = {
    Frequency.H: int(MAX_YEARS * 365.2425 * 24),  # 24 hours per day
    Frequency.S: int(MAX_YEARS * 365.2425 * 24 * 60 * 60),  # Seconds per day
    Frequency.T1: int(MAX_YEARS * 365.2425 * 24 * 60),  # Minutes per day
    Frequency.T5: int(MAX_YEARS * 365.2425 * 24 * 12),  # 5-minute intervals per day
    Frequency.T10: int(MAX_YEARS * 365.2425 * 24 * 6),  # 10-minute intervals per day
    Frequency.T15: int(MAX_YEARS * 365.2425 * 24 * 4),  # 15-minute intervals per day
}

# Combined max lengths for all frequencies
ALL_FREQUENCY_MAX_LENGTHS = {
    **SHORT_FREQUENCY_MAX_LENGTHS,
    **HIGH_FREQUENCY_MAX_LENGTHS,
}

# Frequency to pandas offset mapping for calculating time deltas
FREQUENCY_TO_OFFSET = {
    Frequency.A: "AS",  # Annual start
    Frequency.Q: "QS",  # Quarter start
    Frequency.M: "MS",  # Month start
    Frequency.W: "W",  # Weekly
    Frequency.D: "D",  # Daily
    Frequency.H: "H",  # Hourly
    Frequency.T1: "1T",  # 1 minute
    Frequency.T5: "5T",  # 5 minutes
    Frequency.T10: "10T",  # 10 minutes
    Frequency.T15: "15T",  # 15 minutes
    Frequency.S: "S",  # Seconds
}

GIFT_EVAL_FREQUENCY_WEIGHTS = {
    Frequency.H: 25.0,  # Hourly - most common
    Frequency.D: 23.4,  # Daily - second most common
    Frequency.W: 12.9,  # Weekly - third most common
    Frequency.T15: 9.7,  # 15-minute
    Frequency.T5: 9.7,  # 5-minute
    Frequency.M: 7.3,  # Monthly
    Frequency.T10: 4.8,  # 10-minute
    Frequency.S: 4.8,  # 10-second
    Frequency.T1: 1.6,  # 1-minute
    Frequency.Q: 0.8,  # Quarterly
    Frequency.A: 0.8,  # Annual
}

# GIFT eval-based length ranges derived from actual dataset analysis
# Format: (min_length, max_length, optimal_start, optimal_end)
GIFT_EVAL_LENGTH_RANGES = {
    # Low frequency ranges (based on actual GIFT eval data + logical extensions)
    Frequency.A: (25, 100, 30, 70),
    Frequency.Q: (25, 150, 50, 120),
    Frequency.M: (40, 1000, 100, 600),
    Frequency.W: (50, 3500, 100, 1500),
    # Medium frequency ranges
    Frequency.D: (150, 25000, 300, 7000),  # Daily: covers 1-year+ scenarios
    Frequency.H: (600, 35000, 700, 17000),
    # High frequency ranges (extended for shorter realistic scenarios)
    Frequency.T1: (200, 2500, 1200, 1800),  # 1-minute: day to few days
    Frequency.S: (7500, 9500, 7900, 9000),
    Frequency.T15: (1000, 140000, 50000, 130000),
    Frequency.T5: (
        200,
        105000,
        20000,
        95000,
    ),
    Frequency.T10: (
        40000,
        55000,
        47000,
        52000,
    ),
}


def select_safe_random_frequency(total_length: int, rng: Generator) -> Frequency:
    """
    Selects a random frequency suitable for a given total length of a time series,
    based on actual GIFT eval dataset patterns and distributions.

    This function uses data-driven weights and realistic length ranges extracted from
    analysis of GIFT eval datasets across short/medium/long term horizons.

    The selection logic:
    1. Filters frequencies that can handle the given total_length
    2. Applies base weights derived from actual GIFT eval frequency distribution
    3. Strongly boosts frequencies that are in their optimal length ranges
    4. Handles edge cases gracefully with fallbacks

    Parameters
    ----------
    total_length : int
        The total length of the time series (history + future).
    rng : np.random.Generator
        A numpy random number generator instance.

    Returns
    -------
    Frequency
        A randomly selected frequency that matches GIFT eval patterns.
    """

    # Find valid frequencies and calculate weighted scores
    valid_frequencies = []
    frequency_scores = []

    for freq in Frequency:
        # Check basic timestamp overflow limits
        max_allowed = ALL_FREQUENCY_MAX_LENGTHS.get(freq, float("inf"))
        if total_length > max_allowed:
            continue

        # Check if frequency has defined ranges
        if freq not in GIFT_EVAL_LENGTH_RANGES:
            continue

        min_len, max_len, optimal_start, optimal_end = GIFT_EVAL_LENGTH_RANGES[freq]

        # Must be within the frequency's realistic range
        if total_length < min_len or total_length > max_len:
            continue

        valid_frequencies.append(freq)

        # Calculate fitness score based on GIFT eval patterns
        base_weight = GIFT_EVAL_FREQUENCY_WEIGHTS.get(freq, 0.1)

        # Enhanced length-based fitness scoring
        if optimal_start <= total_length <= optimal_end:
            # In optimal range - very strong preference
            length_multiplier = 5.0
        else:
            # Outside optimal but within valid range - calculate penalty
            if total_length < optimal_start:
                # Below optimal range
                distance_ratio = (optimal_start - total_length) / (
                    optimal_start - min_len
                )
            else:
                # Above optimal range
                distance_ratio = (total_length - optimal_end) / (max_len - optimal_end)

            # Apply graduated penalty: closer to optimal = higher score
            length_multiplier = 0.3 + 1.2 * (1.0 - distance_ratio)  # Range: 0.3-1.5

        final_score = base_weight * length_multiplier
        frequency_scores.append(final_score)

    # Handle edge cases with smart fallbacks
    if not valid_frequencies:
        # Fallback strategy based on typical length patterns
        if total_length <= 100:
            # Very short series - prefer low frequencies
            fallback_order = [
                Frequency.A,
                Frequency.Q,
                Frequency.M,
                Frequency.W,
                Frequency.D,
            ]
        elif total_length <= 1000:
            # Medium short series - prefer daily/weekly
            fallback_order = [Frequency.D, Frequency.W, Frequency.H, Frequency.M]
        else:
            # Longer series - prefer higher frequencies
            fallback_order = [Frequency.H, Frequency.D, Frequency.T15, Frequency.T5]

        for fallback_freq in fallback_order:
            max_allowed = ALL_FREQUENCY_MAX_LENGTHS.get(fallback_freq, float("inf"))
            if total_length <= max_allowed:
                return fallback_freq
        # Last resort
        return Frequency.D

    if len(valid_frequencies) == 1:
        return valid_frequencies[0]

    # Select based on weighted probabilities
    scores = np.array(frequency_scores)
    probabilities = scores / scores.sum()

    return rng.choice(valid_frequencies, p=probabilities)


def select_safe_start_date(
    history_length: int,
    future_length: int,
    frequency: Frequency,
    rng: Generator,
    max_retries: int = 3,
) -> np.datetime64:
    """
    Select a safe start date that ensures the entire time series (history + future)
    will not exceed pandas' datetime bounds and won't cause OutOfBoundsDatetime errors
    in time_features.py, preventing the errors seen in training.

    This function:
    1. Calculates safe bounds based on the series length and frequency
    2. Randomly selects start dates within those bounds
    3. Tests each candidate against time_features.py logic to ensure compatibility
    4. Retries until a safe start date is found

    Parameters
    ----------
    history_length : int
        Length of the history window
    future_length : int
        Length of the future window
    frequency : Frequency
        Time series frequency
    rng : np.random.Generator
        Random number generator instance
    max_retries : int, optional
        Maximum number of retry attempts (default: 50)

    Returns
    -------
    np.datetime64
        A safe start date that prevents timestamp overflow
    """
    total_length = history_length + future_length

    # Get the pandas offset string for the frequency
    offset_str = FREQUENCY_TO_OFFSET.get(frequency, "D")

    # Define the time_features.py safe bounds for validation
    time_features_min = pd.Timestamp("1900-01-01")
    time_features_max = pd.Timestamp("2200-12-31")

    for attempt in range(max_retries):
        try:
            # Calculate the time span that the total series will cover
            test_start = pd.Timestamp("2000-01-01")
            test_end = test_start + pd.Timedelta(
                to_offset(offset_str) * (total_length - 1)
            )
            series_duration = test_end - test_start

            # Calculate bounds considering both our constants and time_features.py constraints
            earliest_start = max(pd.Timestamp(BASE_START_DATE), time_features_min)
            latest_start = min(
                pd.Timestamp(BASE_END_DATE) - series_duration,
                time_features_max - series_duration,
            )

            if latest_start < earliest_start:
                # If the series is too long, use a fallback in the middle of safe range
                mid_point = earliest_start + (time_features_max - earliest_start) / 4
                candidate_start = mid_point
            else:
                # Select a random date between earliest_start and latest_start
                date_range_days = (latest_start - earliest_start).days
                if date_range_days <= 0:
                    candidate_start = earliest_start
                else:
                    random_days = rng.integers(0, date_range_days + 1)
                    candidate_start = earliest_start + pd.Timedelta(days=random_days)

            # Test the candidate start date by simulating time_features.py logic
            if _check_start_date_safety(
                candidate_start, history_length, future_length, offset_str
            ):
                return np.datetime64(candidate_start.strftime("%Y-%m-%d"))

        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
            # Continue to next attempt
            pass

    # If all retries failed, return the base start date
    return BASE_START_DATE


def _check_start_date_safety(
    start_ts: pd.Timestamp, history_length: int, future_length: int, offset_str: str
) -> bool:
    """
    Test if a start date is safe by simulating the time_features.py logic.

    This function replicates the key parts of time_features.py that can cause
    OutOfBoundsDatetime errors to ensure our selected start date is safe.

    Parameters
    ----------
    start_ts : pd.Timestamp
        The candidate start timestamp to test
    history_length : int
        Length of history sequence
    future_length : int
        Length of future sequence
    offset_str : str
        Pandas frequency offset string

    Returns
    -------
    bool
        True if the start date is safe, False otherwise
    """
    try:
        # Test creating history range (matches time_features.py logic)
        history_range = pd.date_range(
            start=start_ts, periods=history_length, freq=offset_str
        )

        # Test creating future range (matches time_features.py logic)
        future_start = history_range[-1] + pd.tseries.frequencies.to_offset(offset_str)
        future_range = pd.date_range(
            start=future_start, periods=future_length, freq=offset_str
        )

        # Verify all timestamps are within time_features.py bounds
        time_features_min = pd.Timestamp(BASE_START_DATE)
        time_features_max = pd.Timestamp(BASE_END_DATE)

        if (
            history_range[0] < time_features_min
            or history_range[-1] > time_features_max
            or future_range[0] < time_features_min
            or future_range[-1] > time_features_max
        ):
            return False

        # Test period index conversion (another source of potential errors)
        # Use a mapping similar to time_features.py
        period_freq_mapping = {
            "AS": "A",
            "QS": "Q",
            "MS": "M",
            "W": "W",
            "D": "D",
            "H": "H",
            "h": "H",
            "1T": "T",
            "5T": "5T",
            "10T": "10T",
            "15T": "15T",
            "S": "S",
            "s": "S",
        }
        period_freq = period_freq_mapping.get(offset_str, "D")

        # Test period index creation
        _ = history_range.to_period(period_freq)
        _ = future_range.to_period(period_freq)

        return True

    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
        return False


def generate_spikes(
    size: int,
    spikes_type: str = "choose_randomly",
    spike_intervals: Optional[int] = None,
    n_spikes: Optional[int] = None,
    to_keep_rate: float = 0.4,
):
    spikes = np.zeros(size)
    if size < 120:
        build_up_points = 1
    elif size < 250:
        build_up_points = np.random.choice([2, 1], p=[0.3, 0.7])
    else:
        build_up_points = np.random.choice([3, 2, 1], p=[0.15, 0.45, 0.4])

    spike_duration = build_up_points * 2

    if spikes_type == "choose_randomly":
        spikes_type = np.random.choice(
            ["regular", "patchy", "random"], p=[0.4, 0.5, 0.1]
        )

    if spikes_type == "patchy" and size < 64:
        spikes_type = "regular"

    if spikes_type in ["regular", "patchy"]:
        if spike_intervals is None:
            upper_bound = np.ceil(
                spike_duration / 0.05
            )  ## at least 1 spike every 24 periods (120 if 5 spike duration) #np.ceil(spike_duration * size/(size*0.05))
            lower_bound = np.ceil(
                spike_duration / 0.15
            )  ## at most 3 spikes every 24 periods
            spike_intervals = np.random.randint(lower_bound, upper_bound)
        n_spikes = np.ceil(size / spike_intervals)
        spike_intervals = np.arange(spike_intervals, size, spike_intervals)
        if spikes_type == "patchy":
            patch_size = np.random.randint(2, max(n_spikes * 0.7, 3))
            to_keep = np.random.randint(np.ceil(patch_size * to_keep_rate), patch_size)
    else:
        n_spikes = (
            n_spikes
            if n_spikes is not None
            else np.random.randint(4, min(max(size // (spike_duration * 3), 6), 20))
        )
        spike_intervals = np.sort(
            np.random.choice(
                np.arange(spike_duration, size), size=n_spikes, replace=False
            )
        )

    constant_build_rate = False
    if spikes_type in ["regular", "patchy"]:
        random_ = np.random.random()
        constant_build_rate = True

    patch_count = 0
    spike_intervals -= 1
    for interval in spike_intervals:
        interval = np.round(interval).astype(int)
        if spikes_type == "patchy":
            if patch_count >= patch_size:
                patch_count = 0
            if patch_count < to_keep:
                patch_count += 1
            else:
                patch_count += 1
                continue
        if not constant_build_rate:
            random_ = np.random.random()
        build_up_rate = (
            np.random.uniform(0.5, 2) if random_ < 0.7 else np.random.uniform(2.5, 5)
        )

        spike_start = interval - build_up_points + 1
        for i in range(build_up_points):
            if 0 <= spike_start + i < len(spikes):
                spikes[spike_start + i] = build_up_rate * (i + 1)

        for i in range(1, build_up_points):
            if (interval + i) < len(spikes):
                spikes[interval + i] = spikes[interval - i]

    # randomly make it positive or negative
    spikes += 1
    spikes = spikes * np.random.choice([1, -1], 1, p=[0.7, 0.3])

    return torch.Tensor(spikes)


def generate_peak_spikes(ts_size, peak_period, spikes_type="regular"):
    return generate_spikes(
        ts_size, spikes_type=spikes_type, spike_intervals=peak_period
    )
