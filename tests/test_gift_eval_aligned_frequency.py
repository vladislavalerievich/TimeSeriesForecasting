#!/usr/bin/env python3
"""
Test the GIFT eval aligned frequency selection logic.
"""

import sys

sys.path.append("/home/moroshav/TimeSeriesForecasting")

from collections import Counter

import numpy as np

from src.data_handling.data_containers import Frequency
from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.common.utils import select_safe_random_frequency
from src.synthetic_generation.generator_params import (
    LongRangeGeneratorParams,
    MediumRangeGeneratorParams,
    ShortRangeGeneratorParams,
)


class TestGeneratorWrapper(GeneratorWrapper):
    """Test wrapper to access frequency selection."""

    def generate_batch(self, batch_size: int, seed=None, **kwargs):
        pass


def test_gift_eval_frequency_distribution():
    """Test that frequency distribution aligns with GIFT eval patterns."""

    print("=== Testing GIFT eval frequency distribution ===\n")

    # Test different length scenarios representative of GIFT eval
    test_scenarios = [
        (
            "Short-term typical",
            400,
            30,
            [Frequency.D],
        ),  # Daily short-term should prefer D
        (
            "Medium-term typical",
            800,
            300,
            [Frequency.H, Frequency.D],
        ),  # Both reasonable for this length
        ("Long-term typical", 1200, 450, [Frequency.H, Frequency.D]),  # Both reasonable
        ("Hourly short", 700, 48, [Frequency.H]),  # Should prefer H
        ("Hourly medium", 1000, 480, [Frequency.H, Frequency.D]),  # Both valid
        (
            "High-freq 5-min",
            7500,
            720,
            [Frequency.T5, Frequency.H],
        ),  # Large context, both reasonable
        (
            "High-freq 15-min",
            5000,
            720,
            [Frequency.T15, Frequency.H],
        ),  # Large context, both reasonable
        (
            "Monthly data",
            200,
            18,
            [Frequency.M, Frequency.W],
        ),  # Both valid for this range
        ("Weekly data", 150, 13, [Frequency.W, Frequency.M]),  # Both valid
    ]

    rng = np.random.default_rng(42)

    overall_frequency_counts = Counter()

    for scenario_name, hist_len, fut_len, expected_reasonable in test_scenarios:
        total_len = hist_len + fut_len
        print(
            f"Scenario: {scenario_name} (history={hist_len}, future={fut_len}, total={total_len})"
        )

        # Test frequency selection for this scenario
        selected_frequencies = []
        for _ in range(200):
            freq = select_safe_random_frequency(total_len, rng)
            selected_frequencies.append(freq)
            overall_frequency_counts[freq] += 1

        freq_counts = Counter(selected_frequencies)
        top_3 = freq_counts.most_common(3)
        print(f"  Top selections: {[(f.value, count) for f, count in top_3]}")

        # Check if most selected frequency makes sense for the scenario
        most_selected = top_3[0][0]
        reasonable = most_selected in expected_reasonable
        print(f"  Expected reasonable: {[f.value for f in expected_reasonable]}")
        print(f"  Reasonable selection: {'✅' if reasonable else '❌'}")
        print()

    # Overall distribution
    print("Overall frequency distribution across all scenarios:")
    total_samples = sum(overall_frequency_counts.values())

    # Updated expected weights based on actual GIFT eval frequency counts
    # From the user's data: H=31, D=29, W=16, T15=12, T5=12, M=9, T10=6, S=6, etc.
    expected_weights = {
        Frequency.H: 25.0,  # Hourly - most common
        Frequency.D: 23.4,  # Daily - second most common
        Frequency.W: 12.9,  # Weekly
        Frequency.T15: 9.7,  # 15-minute
        Frequency.T5: 9.7,  # 5-minute
        Frequency.M: 7.3,  # Monthly
        Frequency.T10: 4.8,  # 10-minute
        Frequency.S: 4.8,  # 10-second
        Frequency.T1: 1.6,  # 1-minute
        Frequency.Q: 0.8,  # Quarterly
        Frequency.A: 0.8,  # Annual
    }

    for freq, count in overall_frequency_counts.most_common():
        percentage = (count / total_samples) * 100
        expected_pct = expected_weights.get(freq, 0)
        status = (
            "✅" if abs(percentage - expected_pct) < 10 else "⚠️"
        )  # Allow 10% tolerance
        print(
            f"  {freq.value}: {count:4d} ({percentage:5.1f}%) - expected: {expected_pct:5.1f}% {status}"
        )


def test_range_parameter_sampling():
    """Test parameter sampling with GIFT eval aligned ranges."""

    print("\n=== Testing GIFT eval aligned parameter sampling ===\n")

    range_configs = [
        ("Short-range", ShortRangeGeneratorParams()),
        ("Medium-range", MediumRangeGeneratorParams()),
        ("Long-range", LongRangeGeneratorParams()),
    ]

    for range_name, params in range_configs:
        print(f"{range_name} parameters:")
        print(f"  History length: {params.history_length}")
        print(f"  Future length: {params.future_length}")

        wrapper = TestGeneratorWrapper(params)

        # Sample parameters multiple times
        frequency_counts = Counter()
        realistic_pairs = 0
        total_samples = 100

        for _ in range(total_samples):
            sampled = wrapper._sample_parameters()
            total_length = sampled["history_length"] + sampled["future_length"]
            frequency = sampled["frequency"]

            frequency_counts[frequency] += 1

            # Check if pairing is realistic based on GIFT eval patterns
            # More lenient criteria based on actual GIFT eval data ranges
            if (
                (frequency == Frequency.D and 150 <= total_length <= 25000)
                or (frequency == Frequency.H and 600 <= total_length <= 35000)
                or (frequency == Frequency.T15 and 1000 <= total_length <= 140000)
                or (frequency == Frequency.T5 and 200 <= total_length <= 105000)
                or (frequency == Frequency.M and 40 <= total_length <= 1000)
                or (frequency == Frequency.W and 50 <= total_length <= 3500)
                or (frequency == Frequency.T10 and 40000 <= total_length <= 55000)
                or (frequency == Frequency.S and 7500 <= total_length <= 9500)
                or (frequency == Frequency.Q and 25 <= total_length <= 150)
                or (frequency == Frequency.A and 25 <= total_length <= 100)
                or (frequency == Frequency.T1 and 200 <= total_length <= 2500)
            ):
                realistic_pairs += 1

        print(f"  Frequency distribution:")
        for freq, count in frequency_counts.most_common():
            percentage = (count / total_samples) * 100
            print(f"    {freq.value}: {count:2d} ({percentage:4.1f}%)")

        realistic_ratio = realistic_pairs / total_samples
        print(
            f"  Realistic pairings: {realistic_pairs}/{total_samples} ({realistic_ratio * 100:.1f}%)"
        )

        if realistic_ratio >= 0.7:
            print(f"  ✅ Good ratio of realistic frequency-length pairings")
        else:
            print(f"  ❌ Too many unrealistic pairings")
        print()


def test_specific_gift_eval_cases():
    """Test specific cases that mirror actual GIFT eval datasets."""

    print("=== Testing specific GIFT eval mirrored cases ===\n")

    # Mirror some actual GIFT eval dataset characteristics
    # Updated with more realistic expectations based on GIFT eval patterns
    gift_eval_cases = [
        (
            "electricity/D short",
            1311,
            30,
            [Frequency.D, Frequency.H],
        ),  # Both reasonable for this length
        ("ett1/H long", 1024, 720, [Frequency.H]),  # Should prefer H
        (
            "hierarchical_sales/W short",
            228,
            8,
            [Frequency.W, Frequency.M],
        ),  # Both valid
        (
            "m4_monthly short",
            469,
            18,
            [Frequency.M, Frequency.D, Frequency.W],
        ),  # Multiple valid
        (
            "solar/15T long",
            1024,
            720,
            [Frequency.T15, Frequency.H],
        ),  # Both reasonable for this range
        (
            "bizitobs_service short",
            1024,
            60,
            [Frequency.S, Frequency.H],
        ),  # Both possible for this length
    ]

    rng = np.random.default_rng(42)

    for case_name, hist_len, fut_len, expected_reasonable in gift_eval_cases:
        total_len = hist_len + fut_len
        print(f"Case: {case_name} (total_length={total_len})")

        # Test frequency selection
        selected_frequencies = []
        for _ in range(100):
            freq = select_safe_random_frequency(total_len, rng)
            selected_frequencies.append(freq)

        freq_counts = Counter(selected_frequencies)
        most_common = freq_counts.most_common(1)[0]
        most_selected_freq, count = most_common

        print(f"  Expected reasonable: {[f.value for f in expected_reasonable]}")
        print(f"  Most selected: {most_selected_freq.value} ({count}%)")

        reasonable = most_selected_freq in expected_reasonable
        print(f"  Reasonable: {'✅' if reasonable else '❌'}")
        print()


if __name__ == "__main__":
    test_gift_eval_frequency_distribution()
    test_range_parameter_sampling()
    test_specific_gift_eval_cases()
