#!/usr/bin/env python3
"""
Test script to verify gluonts time feature integration.
"""

from datetime import datetime

import numpy as np
import torch

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.time_features import (
    compute_batch_time_features,
    compute_enhanced_time_features,
    frequency_to_gluonts_str,
    get_seasonality,
    time_features_from_frequency,
)
from src.synthetic_generation.common.constants import Frequency
from src.synthetic_generation.generator_params import KernelGeneratorParams
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorWrapper,
)


def test_time_feature_functions():
    """Test individual time feature functions."""
    print("=== Testing Time Feature Functions ===")

    # Test frequency mapping
    for freq in Frequency:
        gluonts_str = frequency_to_gluonts_str(freq)
        seasonality = get_seasonality(freq)
        features = time_features_from_frequency(freq, use_normalized=True)
        print(
            f"{freq.name}: {gluonts_str} -> {len(features)} features, seasonality={seasonality}"
        )
    print()


def test_time_feature_computation():
    """Test time feature computation with different modes."""
    print("=== Testing Time Feature Computation ===")

    batch_size = 2
    history_length = 48  # 2 days for hourly data
    target_length = 24  # 1 day prediction
    frequency = Frequency.H

    # Create sample start times
    start_times = np.array(
        [
            np.datetime64("2023-01-01T00:00:00"),
            np.datetime64("2023-01-15T12:00:00"),
        ]
    )

    # Test legacy features
    print("Testing legacy features...")
    hist_legacy, targ_legacy = compute_batch_time_features(
        start_times,
        history_length,
        target_length,
        batch_size,
        frequency,
        use_gluonts_features=False,
        use_normalized_features=False,
    )
    print(
        f"Legacy - History shape: {hist_legacy.shape}, Target shape: {targ_legacy.shape}"
    )
    print(f"Legacy - History sample: {hist_legacy[0, 0, :].tolist()}")

    # Test gluonts features
    print("\nTesting gluonts features...")
    hist_gluonts, targ_gluonts = compute_batch_time_features(
        start_times,
        history_length,
        target_length,
        batch_size,
        frequency,
        use_gluonts_features=True,
        use_normalized_features=True,
    )
    print(
        f"Gluonts - History shape: {hist_gluonts.shape}, Target shape: {targ_gluonts.shape}"
    )
    print(f"Gluonts - History sample: {hist_gluonts[0, 0, :].tolist()}")

    # Test enhanced features
    print("\nTesting enhanced features...")
    hist_enhanced, targ_enhanced = compute_enhanced_time_features(
        start_times,
        history_length,
        target_length,
        batch_size,
        frequency,
        include_seasonality=True,
        include_cyclical_encoding=True,
    )
    print(
        f"Enhanced - History shape: {hist_enhanced.shape}, Target shape: {targ_enhanced.shape}"
    )
    print(f"Enhanced - History sample: {hist_enhanced[0, 0, :].tolist()}")
    print()


def test_with_synthetic_data():
    """Test integration with synthetic data generation."""
    print("=== Testing with Synthetic Data ===")

    # Create synthetic data generator
    params = KernelGeneratorParams(
        history_length=48, target_length=24, num_channels=(2, 4), num_kernels=(1, 3)
    )

    generator = KernelGeneratorWrapper(params)

    # Generate a batch
    batch_container = generator.generate_batch(batch_size=3, seed=42)

    print(f"Generated batch:")
    print(f"  History shape: {batch_container.history_values.shape}")
    print(f"  Target shape: {batch_container.target_values.shape}")
    print(f"  Frequency: {batch_container.frequency}")
    print(f"  Start times: {batch_container.start}")

    # Test time feature computation with the generated data
    hist_features, targ_features = compute_batch_time_features(
        batch_container.start,
        batch_container.history_length,
        batch_container.target_length,
        batch_container.batch_size,
        batch_container.frequency,
        use_gluonts_features=True,
        use_normalized_features=True,
    )

    print(
        f"  Time features - History: {hist_features.shape}, Target: {targ_features.shape}"
    )
    print()


def test_different_frequencies():
    """Test with different frequencies to ensure robustness."""
    print("=== Testing Different Frequencies ===")

    frequencies_to_test = [
        Frequency.H,
        Frequency.D,
        Frequency.W,
        Frequency.M,
        Frequency.T5,
        Frequency.T15,
        Frequency.S,
    ]

    for freq in frequencies_to_test:
        try:
            start_times = np.array([np.datetime64("2023-01-01T00:00:00")])

            # Adjust lengths based on frequency
            if freq in [Frequency.S, Frequency.T5, Frequency.T15]:
                history_length, target_length = 120, 60  # 2 hours, 1 hour
            elif freq == Frequency.H:
                history_length, target_length = 48, 24  # 2 days, 1 day
            elif freq == Frequency.D:
                history_length, target_length = 14, 7  # 2 weeks, 1 week
            elif freq == Frequency.W:
                history_length, target_length = 8, 4  # 2 months, 1 month
            else:  # Monthly
                history_length, target_length = 24, 12  # 2 years, 1 year

            hist_feats, targ_feats = compute_batch_time_features(
                start_times,
                history_length,
                target_length,
                1,
                freq,
                use_gluonts_features=True,
                use_normalized_features=True,
            )

            print(f"{freq.name}: {hist_feats.shape[-1]} features - OK")

        except Exception as e:
            print(f"{freq.name}: ERROR - {str(e)}")

    print()


if __name__ == "__main__":
    print("Testing GluonTS Time Feature Integration\n")

    test_time_feature_functions()
    test_time_feature_computation()
    test_with_synthetic_data()
    test_different_frequencies()

    print("All tests completed!")
