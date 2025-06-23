#!/usr/bin/env python3
"""
Test script to validate the improved frequency selection function.
"""

import sys

sys.path.append("/home/moroshav/TimeSeriesForecasting")

from collections import Counter

import numpy as np

from src.data_handling.data_containers import Frequency
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.common.utils import select_safe_random_frequency
from src.synthetic_generation.generator_params import GeneratorParams


class TestGeneratorWrapper(GeneratorWrapper):
    """Test wrapper to access frequency selection."""

    def generate_batch(self, batch_size: int, seed=None, **kwargs):
        pass


def test_improved_frequency_selection():
    """Test the improved frequency selection for realistic scenarios."""

    print("=== Testing improved frequency selection ===\n")

    scenarios = [
        ("Daily data for 1 year", 365, [Frequency.D]),
        (
            "Monthly data for 10 years",
            120,
            [Frequency.M, Frequency.W],
        ),  # Both are reasonable
        (
            "Quarterly data for 20 years",
            80,
            [Frequency.Q, Frequency.W, Frequency.M],
        ),  # Multiple valid options
        (
            "Annual data for 50 years",
            50,
            [Frequency.A, Frequency.Q],
        ),  # Both valid for this length
        ("Hourly data for 1 month", 720, [Frequency.H]),
        ("Long hourly sequence", 1440, [Frequency.H]),  # More realistic for hourly
        (
            "High-freq 5-minute data",
            8000,
            [Frequency.T5],
        ),  # Realistic 5-min length from GIFT eval
        ("Weekly data for 2 years", 104, [Frequency.W, Frequency.M]),  # Both reasonable
        ("Very short sequence", 10, []),  # Should handle gracefully
        (
            "Very long daily sequence",
            5000,
            [Frequency.D, Frequency.H],
        ),  # Both valid for long sequences
    ]

    rng = np.random.default_rng(42)

    for description, length, expected_reasonable in scenarios:
        print(f"Scenario: {description} (length={length})")

        # Test selection multiple times
        selected_frequencies = []
        for _ in range(100):
            freq = select_safe_random_frequency(length, rng)
            selected_frequencies.append(freq)

        freq_counts = Counter(selected_frequencies)
        most_common = freq_counts.most_common(3)
        print(f"  Most selected: {[(f.value, count) for f, count in most_common]}")

        if expected_reasonable:
            most_selected_freq = most_common[0][0]
            reasonable_selected = most_selected_freq in expected_reasonable
            print(f"  Expected reasonable: {[f.value for f in expected_reasonable]}")
            print(f"  Reasonable selection: {'✅' if reasonable_selected else '❌'}")
        else:
            print("  Expected: Should handle gracefully")
        print()


def test_edge_cases():
    """Test edge cases for frequency selection."""

    print("=== Testing edge cases ===\n")

    rng = np.random.default_rng(42)

    # Test very short sequences
    for length in [1, 2, 5, 10]:
        freq = select_safe_random_frequency(length, rng)
        print(f"Length {length}: Selected {freq.value}")

    print()

    # Test very long sequences that exceed some limits
    for length in [50000, 100000, 500000]:
        try:
            freq = select_safe_random_frequency(length, rng)
            print(f"Length {length}: Selected {freq.value}")
        except Exception as e:
            print(f"Length {length}: Error - {e}")

    print()


def test_weight_distribution():
    """Test that the weighting system produces reasonable distributions."""

    print("=== Testing weight distribution ===\n")

    test_cases = [
        (365, "Daily for 1 year - should prefer D"),
        (120, "Monthly for 10 years - M and W both reasonable"),
        (52, "Weekly for 1 year - W, M, Q all reasonable for short sequences"),
        (720, "Hourly data - should strongly prefer H"),
        (1440, "Long hourly sequence - should prefer H"),
    ]

    rng = np.random.default_rng(42)

    for length, description in test_cases:
        print(f"{description} (length={length})")

        # Generate many samples to see distribution
        samples = []
        for _ in range(1000):
            freq = select_safe_random_frequency(length, rng)
            samples.append(freq)

        counts = Counter(samples)
        total = sum(counts.values())

        print("  Distribution:")
        for freq, count in counts.most_common():
            percentage = (count / total) * 100
            print(f"    {freq.value}: {count:3d} ({percentage:5.1f}%)")
        print()


def test_full_parameter_sampling():
    """Test the full parameter sampling with improved frequency selection."""

    print("=== Testing full parameter sampling with improved frequency ===\n")

    params = GeneratorParams(
        history_length=(100, 2000), future_length=(50, 1000), num_channels=[1, 2, 3]
    )

    wrapper = TestGeneratorWrapper(params)

    frequency_counts = Counter()
    realistic_pairs = 0
    total_samples = 200

    for i in range(total_samples):
        sampled = wrapper._sample_parameters()
        total_length = sampled["history_length"] + sampled["future_length"]
        frequency = sampled["frequency"]

        frequency_counts[frequency] += 1

        # Check if the pairing seems realistic based on GIFT eval patterns
        if (
            (frequency == Frequency.D and 200 <= total_length <= 25000)
            or (frequency == Frequency.H and 650 <= total_length <= 35000)
            or (frequency == Frequency.M and 40 <= total_length <= 1000)
            or (frequency == Frequency.W and 50 <= total_length <= 3500)
            or (frequency == Frequency.Q and 25 <= total_length <= 150)
            or (frequency == Frequency.A and 25 <= total_length <= 100)
            or (frequency == Frequency.T15 and 1000 <= total_length <= 140000)
            or (frequency == Frequency.T5 and 200 <= total_length <= 105000)
            or (frequency == Frequency.T1 and 200 <= total_length <= 2500)
            or (frequency == Frequency.S and 7500 <= total_length <= 9500)
            or (frequency == Frequency.T10 and 40000 <= total_length <= 55000)
        ):
            realistic_pairs += 1

    print("Frequency distribution:")
    for freq, count in frequency_counts.most_common():
        percentage = (count / total_samples) * 100
        print(f"  {freq.value}: {count:3d} ({percentage:5.1f}%)")

    print(
        f"\nRealistic frequency-length pairings: {realistic_pairs}/{total_samples} ({realistic_pairs / total_samples * 100:.1f}%)"
    )

    if realistic_pairs / total_samples > 0.7:
        print("✅ Good ratio of realistic pairings")
    else:
        print("❌ Too many unrealistic pairings")


if __name__ == "__main__":
    test_improved_frequency_selection()
    test_edge_cases()
    test_weight_distribution()
    test_full_parameter_sampling()
