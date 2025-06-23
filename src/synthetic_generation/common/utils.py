from typing import Optional

import numpy as np
import torch
from numpy.random import Generator

from src.data_handling.data_containers import Frequency

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

# GIFT eval-based frequency distribution (actual percentages from 124 datasets)
GIFT_EVAL_FREQUENCY_WEIGHTS = {
    Frequency.H: 21.8,  # Hourly - most common (27/124 datasets)
    Frequency.D: 18.5,  # Daily - second most common (23/124 datasets)
    Frequency.W: 12.9,  # Weekly - third most common (16/124 datasets)
    Frequency.T15: 9.7,  # 15-minute (12/124 datasets)
    Frequency.T5: 9.7,  # 5-minute (12/124 datasets)
    Frequency.M: 6.5,  # Monthly (8/124 datasets)
    Frequency.T10: 5.6,  # 10-minute (7/124 datasets)
    Frequency.S: 4.8,  # 10-second (6/124 datasets)
    Frequency.T1: 2.0,  # 1-minute (rare, estimated)
    Frequency.Q: 0.8,  # Quarterly (1/124 datasets)
    Frequency.A: 0.8,  # Annual (1/124 datasets)
}

# More conservative GIFT eval-based ranges with clear discrimination
# Format: (min_length, max_length, optimal_start, optimal_end)
GIFT_EVAL_LENGTH_RANGES = {
    # Low frequency ranges
    Frequency.A: (30, 65, 35, 50),  # Annual: very specific range
    Frequency.Q: (25, 100, 30, 60),  # Quarterly: narrow range
    Frequency.M: (40, 900, 80, 700),  # Monthly: 40-900, optimal 80-700
    Frequency.W: (
        50,
        3400,
        100,
        1000,
    ),  # Weekly: 50-3.4k, optimal 100-1k (reduced overlap)
    # Medium frequency ranges
    Frequency.D: (30, 7000, 200, 2500),  # Daily: 30-7k, optimal 200-2.5k (lowered min)
    Frequency.H: (
        24,
        35000,
        700,
        20000,
    ),  # Hourly: 24-35k, optimal 700-20k (lowered min)
    # High frequency ranges
    Frequency.T1: (1400, 2500, 1400, 2000),  # 1-minute: narrow optimal range
    Frequency.S: (7600, 9000, 7600, 9000),  # 10-second: narrow range
    Frequency.T15: (2600, 140000, 5000, 80000),  # 15-minute: 2.6k-140k, optimal 5k-80k
    Frequency.T5: (7000, 105000, 10000, 60000),  # 5-minute: 7k-105k, optimal 10k-60k
    Frequency.T10: (46000, 53000, 47000, 52000),  # 10-minute: very narrow range
}


def select_safe_random_frequency(total_length: int, rng: Generator) -> Frequency:
    """
    Selects a random frequency suitable for a given total length of a time series,
    based on actual GIFT eval dataset patterns and distributions.

    This function uses conservative data-driven weights and realistic length ranges
    derived from analysis of 124 GIFT eval datasets across short/medium/long term horizons.

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

    # Find valid frequencies and calculate scores
    valid_frequencies = []
    frequency_scores = []

    for freq in Frequency:
        # Check basic timestamp limits
        max_allowed = ALL_FREQUENCY_MAX_LENGTHS.get(freq, float("inf"))
        if total_length > max_allowed:
            continue

        # Check if frequency has GIFT eval ranges defined
        if freq not in GIFT_EVAL_LENGTH_RANGES:
            continue

        min_len, max_len, optimal_start, optimal_end = GIFT_EVAL_LENGTH_RANGES[freq]

        # Must be within the frequency's realistic range
        if total_length < min_len or total_length > max_len:
            continue

        valid_frequencies.append(freq)

        # Calculate fitness score
        base_weight = GIFT_EVAL_FREQUENCY_WEIGHTS.get(freq, 0.1)

        # Length-based fitness with moderate boosts
        if optimal_start <= total_length <= optimal_end:
            # In optimal range - frequency gets its strongest preference
            length_multiplier = 2.0  # Moderate boost for optimal range
        else:
            # Outside optimal but within valid range
            # Calculate distance-based penalty
            if total_length < optimal_start:
                distance_penalty = (optimal_start - total_length) / (
                    optimal_start - min_len
                )
            else:
                distance_penalty = (total_length - optimal_end) / (
                    max_len - optimal_end
                )

            length_multiplier = 0.3 + 0.4 * (1.0 - distance_penalty)  # Range 0.3-0.7

        final_score = base_weight * length_multiplier
        frequency_scores.append(final_score)

    # Handle edge cases
    if not valid_frequencies:
        return Frequency.D  # Fallback to Daily

    if len(valid_frequencies) == 1:
        return valid_frequencies[0]

    # Select based on weighted probabilities
    scores = np.array(frequency_scores)
    probabilities = scores / scores.sum()

    return rng.choice(valid_frequencies, p=probabilities)


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
