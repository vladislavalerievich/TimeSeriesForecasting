from typing import Optional

import numpy as np
import pandas as pd
import torch
from numpy.random import Generator

from src.data_handling.data_containers import Frequency
from src.synthetic_generation.common.constants import (
    BASE_END_DATE,
    BASE_START_DATE,
    FREQUENCY_MAPPING,
)

# Cap the total length for low-frequency series to prevent timestamp overflow
# Pandas datetime range is roughly 1677-2262, so we need conservative limits

MAX_YEARS = 500

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


def check_start_date_safety(
    start_date: np.datetime64, total_length: int, frequency: Frequency
) -> bool:
    """
    Enhanced check for start date safety with comprehensive bounds validation.

    This function verifies that pd.date_range(start=start_date, periods=total_length, freq=freq_str)
    will not raise an OutOfBoundsDatetime error, accounting for pandas' datetime bounds
    (1677-09-21 to 2262-04-11) and realistic frequency limitations.

    Parameters
    ----------
    start_date : np.datetime64
        The proposed start date for the time series
    total_length : int
        Total length of the time series (history_length + future_length)
    frequency : Frequency
        The frequency of the time series

    Returns
    -------
    bool
        True if the start date is safe, False otherwise
    """
    try:
        # Get the pandas frequency string using FREQUENCY_MAPPING
        freq_str_base, freq_str_prefix, _ = FREQUENCY_MAPPING[frequency]

        # Construct frequency string with special handling for date_range compatibility
        if frequency == Frequency.M:
            freq_str = "ME"  # Month End for date_range compatibility
        elif frequency == Frequency.A:
            freq_str = "YE"  # Year End for date_range compatibility
        elif frequency == Frequency.Q:
            freq_str = "QE"  # Quarter End for date_range compatibility
        elif freq_str_prefix:
            freq_str = f"{freq_str_prefix}{freq_str_base}"
        else:
            freq_str = freq_str_base

        # Convert numpy datetime64 to pandas Timestamp for date_range
        start_pd = pd.Timestamp(start_date)

        # Check if start date is within pandas' valid datetime range
        if start_pd < pd.Timestamp.min or start_pd > pd.Timestamp.max:
            return False

        # Calculate the approximate end date to check if it exceeds bounds
        _, _, days_per_period = FREQUENCY_MAPPING[frequency]

        # For very long series or low frequencies, be extra conservative
        if frequency in [Frequency.A, Frequency.Q, Frequency.M]:
            # For low frequencies, check more strictly
            if frequency == Frequency.A and total_length > 500:  # Max ~500 years
                return False
            elif frequency == Frequency.Q and total_length > 2000:  # Max ~500 years
                return False
            elif frequency == Frequency.M and total_length > 6000:  # Max ~500 years
                return False

        # Calculate approximate end date
        approx_days = total_length * days_per_period

        # For annual/quarterly frequencies, add extra safety margin
        if frequency in [Frequency.A, Frequency.Q]:
            approx_days *= 1.1  # 10% safety margin

        end_date = start_pd + pd.Timedelta(days=approx_days)

        # Check if end date is within pandas' valid datetime range
        if end_date < pd.Timestamp.min or end_date > pd.Timestamp.max:
            return False

        # Try to create the date range
        pd.date_range(start=start_pd, periods=total_length, freq=freq_str)
        return True

    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
        return False


def select_safe_start_date(
    history_length: int,
    future_length: int,
    frequency: Frequency,
    rng: np.random.Generator = np.random.default_rng(),
    max_retries: int = 5,
) -> np.datetime64:
    """
    Select a safe start date that ensures the entire time series (history + future)
    will not exceed pandas' datetime bounds and won't cause OutOfBoundsDatetime errors.

    This function:
    1. Calculates safe bounds based on the series length and frequency
    2. Randomly selects start dates within those bounds using uniform distribution
    3. Verifies safety using check_start_date_safety
    4. Retries until a safe start date is found or max_retries is reached

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
        Maximum number of retry attempts (default: 5)

    Returns
    -------
    np.datetime64
        A safe start date that prevents timestamp overflow

    Raises
    ------
    ValueError
        If no safe start date is found after max_retries or if the required
        time span exceeds the available date window
    """
    total_length = history_length + future_length
    _, _, days_per_period = FREQUENCY_MAPPING[frequency]

    # Calculate approximate duration in days
    total_days = total_length * days_per_period

    # Define safe bounds: ensure end date doesn't exceed BASE_END_DATE
    latest_safe_start = BASE_END_DATE - np.timedelta64(int(total_days), "D")
    earliest_safe_start = BASE_START_DATE

    # Check if the required time span exceeds the available window
    if latest_safe_start < earliest_safe_start:
        available_days = (
            (BASE_END_DATE - BASE_START_DATE).astype("timedelta64[D]").astype(int)
        )
        available_years = available_days / 365.25
        required_years = total_days / 365.25
        raise ValueError(
            f"Required time span ({required_years:.1f} years, {total_days:.0f} days) "
            f"exceeds available date window ({available_years:.1f} years, {available_days} days). "
            f"Reduce total_length ({total_length}) or extend the date window."
        )

    # Convert to nanoseconds for random sampling
    earliest_ns = earliest_safe_start.astype("datetime64[ns]").astype(np.int64)
    latest_ns = latest_safe_start.astype("datetime64[ns]").astype(np.int64)

    for _ in range(max_retries):
        # Uniformly sample a start date within bounds
        random_ns = rng.integers(earliest_ns, latest_ns + 1)
        start_date = np.datetime64(int(random_ns), "ns")

        # Verify safety
        if check_start_date_safety(start_date, total_length, frequency):
            return start_date

    raise ValueError(f"No safe start date found after {max_retries} attempts")


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
