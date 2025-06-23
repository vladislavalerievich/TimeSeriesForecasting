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

# Simplified frequency selection based on GIFT eval patterns
# Non-overlapping ranges for clear preferences
FREQUENCY_OPTIMAL_RANGES = {
    Frequency.A: (20, 60),  # Annual: 20-60 total length
    Frequency.Q: (60, 120),  # Quarterly: 60-120 total length
    Frequency.M: (120, 400),  # Monthly: 120-400 total length
    Frequency.W: (200, 600),  # Weekly: 200-600 total length
    Frequency.D: (400, 1800),  # Daily: 400-1800 total length (main workhorse)
    Frequency.H: (800, 2200),  # Hourly: 800-2200 total length (main workhorse)
    Frequency.S: (1000, 2000),  # 10-second: 1000-2000 total length
    Frequency.T1: (1800, 3500),  # 1-minute: 1800-3500 total length
    Frequency.T10: (2500, 5000),  # 10-minute: 2500-5000 total length
    Frequency.T5: (4000, 8000),  # 5-minute: 4000-8000 total length
    Frequency.T15: (5000, 9000),  # 15-minute: 5000-9000 total length
}


def select_safe_random_frequency(total_length: int, rng: Generator) -> Frequency:
    """
    Selects a random frequency suitable for a given total length of a time series,
    avoiding combinations that could lead to pandas timestamp overflow errors.
    Uses simplified logic based on GIFT eval patterns.

    Parameters
    ----------
    total_length : int
        The total length of the time series (history + future).
    rng : np.random.Generator
        A numpy random number generator instance.

    Returns
    -------
    Frequency
        A randomly selected, suitable frequency with GIFT eval-aligned weighting.
    """
    # GIFT eval frequency weights (simplified)
    base_weights = {
        Frequency.D: 30.0,  # Daily - most common in GIFT eval
        Frequency.H: 23.0,  # Hourly - very common
        Frequency.T15: 13.0,  # 15-minute - common
        Frequency.W: 10.0,  # Weekly - moderate
        Frequency.T5: 10.0,  # 5-minute - common
        Frequency.T10: 8.0,  # 10-minute - moderate
        Frequency.M: 7.0,  # Monthly - moderate
        Frequency.S: 4.0,  # 10-second - less common
        Frequency.T1: 2.0,  # 1-minute - rare
        Frequency.Q: 1.0,  # Quarterly - rare
        Frequency.A: 1.0,  # Annual - rare
    }

    # Get all frequencies that don't exceed max length constraints
    valid_frequencies = []
    weights = []

    for freq in Frequency:
        max_allowed = ALL_FREQUENCY_MAX_LENGTHS.get(freq, float("inf"))

        if total_length <= max_allowed:
            valid_frequencies.append(freq)

            # Get base weight
            base_weight = base_weights.get(freq, 0.1)

            # Check if length is in optimal range
            optimal_min, optimal_max = FREQUENCY_OPTIMAL_RANGES.get(
                freq, (1, float("inf"))
            )

            if optimal_min <= total_length <= optimal_max:
                # In optimal range - use full weight
                final_weight = base_weight * 10.0  # Strong boost for optimal range
            else:
                # Outside optimal range - use reduced weight
                final_weight = base_weight * 0.1

            weights.append(final_weight)

    if not valid_frequencies:
        # Fallback to Daily
        return Frequency.D

    # Normalize weights to probabilities
    weights = np.array(weights)
    probabilities = weights / weights.sum()

    # Select frequency based on weighted probabilities
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
