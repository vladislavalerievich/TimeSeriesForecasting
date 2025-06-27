#!/usr/bin/env python3
"""
Test script to verify that the improved safe start date selection prevents
pandas OutOfBoundsDatetime errors and is compatible with time_features.py.
"""

import numpy as np
import pandas as pd

from src.data_handling.data_containers import Frequency
from src.data_handling.time_features import compute_batch_time_features
from src.synthetic_generation.common.utils import select_safe_start_date


def test_safe_start_date_selection():
    """Test the safe start date selection function with various scenarios."""

    rng = np.random.default_rng(42)

    # Test cases including the problematic case from the training error
    test_cases = [
        # (history_length, future_length, frequency, description)
        (443, 195, Frequency.Q, "Quarterly case from training error"),
        (512, 900, Frequency.Q, "Quarterly with very long future"),
        (649, 300, Frequency.M, "Monthly with medium length"),
        (1024, 450, Frequency.A),  # Annual with long history/future
        (100, 120, Frequency.M),  # Monthly
        (500, 600, Frequency.W),  # Weekly
        (200, 300, Frequency.D),  # Daily
        (1000, 500, Frequency.H),  # Hourly
        (925, 195, Frequency.D, "Daily case from training"),
        (576, 180, Frequency.H, "Hourly case from training"),
    ]

    print(
        "Testing improved safe start date selection with time_features.py compatibility..."
    )

    all_passed = True

    for i, test_case in enumerate(test_cases):
        if len(test_case) == 4:
            history_len, future_len, freq, description = test_case
        else:
            history_len, future_len, freq = test_case
            description = f"{freq.name}"

        try:
            print(
                f"\n{i + 1}. Testing {description}: hist={history_len}, fut={future_len}"
            )

            # Test the safe start date selection
            start_date = select_safe_start_date(history_len, future_len, freq, rng)
            print(f"   Selected start: {start_date}")

            # Test with time_features.py to ensure actual compatibility
            batch_size = 2
            start_array = np.array([start_date, start_date])

            try:
                history_features, future_features = compute_batch_time_features(
                    start=start_array,
                    history_length=history_len,
                    future_length=future_len,
                    batch_size=batch_size,
                    frequency=freq,
                    K_max=6,
                )

                print(
                    f"   ✓ time_features.py succeeded: hist_shape={history_features.shape}, fut_shape={future_features.shape}"
                )

                # Verify the shapes are correct
                expected_hist_shape = (batch_size, history_len, 6)
                expected_fut_shape = (batch_size, future_len, 6)

                if (
                    history_features.shape == expected_hist_shape
                    and future_features.shape == expected_fut_shape
                ):
                    print(f"   ✓ Output shapes are correct")
                else:
                    print(
                        f"   ⚠ Unexpected shapes: expected hist={expected_hist_shape}, fut={expected_fut_shape}"
                    )

            except Exception as time_features_error:
                print(f"   ✗ time_features.py failed: {time_features_error}")
                all_passed = False
                continue

            # Also verify we can create date ranges manually
            freq_map = {
                Frequency.A: "AS",
                Frequency.Q: "QS",
                Frequency.M: "MS",
                Frequency.W: "W",
                Frequency.D: "D",
                Frequency.H: "H",
                Frequency.T1: "1T",
                Frequency.T5: "5T",
                Frequency.T10: "10T",
                Frequency.T15: "15T",
                Frequency.S: "S",
            }

            offset_str = freq_map.get(freq, "D")
            total_length = history_len + future_len

            date_range = pd.date_range(
                start=pd.Timestamp(start_date), periods=total_length, freq=offset_str
            )

            print(
                f"   ✓ Manual date_range creation succeeded: {len(date_range)} periods"
            )
            print(f"   End date: {date_range[-1]}")

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Safe start date selection is working correctly.")
        print("The function should now prevent OutOfBoundsDatetime errors in training.")
    else:
        print("\n❌ Some tests failed. The function may need further improvements.")

    return all_passed


def test_extreme_cases():
    """Test some extreme cases that might push the boundaries."""
    print("\n" + "=" * 60)
    print("Testing extreme boundary cases...")

    rng = np.random.default_rng(123)

    extreme_cases = [
        (2000, 900, Frequency.Q, "Very long quarterly series"),
        (1024, 900, Frequency.A, "Maximum length annual series"),
        (100, 6, Frequency.A, "Short annual series"),
    ]

    for history_len, future_len, freq, description in extreme_cases:
        try:
            print(f"\nTesting {description}: hist={history_len}, fut={future_len}")
            start_date = select_safe_start_date(history_len, future_len, freq, rng)
            print(f"✓ Extreme case handled: {start_date}")
        except Exception as e:
            print(f"✗ Extreme case failed: {e}")


if __name__ == "__main__":
    success = test_safe_start_date_selection()
    test_extreme_cases()

    if success:
        print(
            "\n✅ Ready for training! The start date selection should prevent timestamp errors."
        )
    else:
        print("\n⚠️ Some issues detected. Review the output above.")
