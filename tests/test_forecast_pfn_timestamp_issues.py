import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_handling.data_containers import Frequency
from src.data_handling.time_features import compute_batch_time_features
from src.synthetic_generation.dataset_composer import DefaultSyntheticComposer
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
    ForecastPFNGeneratorWrapper,
)
from src.synthetic_generation.generator_params import ForecastPFNGeneratorParams

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestForecastPFNTimestampIssues:
    """Test suite to identify and debug timestamp overflow issues in ForecastPFN generator."""

    def test_forecast_pfn_direct_generation(self):
        """Test direct ForecastPFN generation with various parameters."""
        print("\n=== Testing Direct ForecastPFN Generation ===")

        # Test with different frequencies that are known to cause issues
        problematic_frequencies = [Frequency.A, Frequency.Q, Frequency.M]

        for freq in problematic_frequencies:
            print(f"\nTesting frequency: {freq}")

            params = ForecastPFNGeneratorParams(
                global_seed=42,
                history_length=1024,  # Long length to stress test
                future_length=900,  # Long length to stress test
                num_channels=1,
                frequency=freq,
            )

            wrapper = ForecastPFNGeneratorWrapper(params)

            try:
                # Generate a small batch
                batch = wrapper.generate_batch(batch_size=2, seed=42)
                print(f"✓ Successfully generated batch for {freq}")
                print(f"  Start timestamps: {batch.start}")
                print(f"  History shape: {batch.history_values.shape}")
                print(f"  Future shape: {batch.future_values.shape}")

                # Test time features computation
                history_features, target_features = compute_batch_time_features(
                    start=batch.start,
                    history_length=batch.history_length,
                    target_length=batch.future_length,
                    batch_size=batch.batch_size,
                    frequency=batch.frequency,
                )
                print(f"✓ Successfully computed time features for {freq}")

            except Exception as e:
                print(f"✗ Failed for frequency {freq}: {e}")
                raise

    def test_dataset_composer_generation(self):
        """Test dataset composer generation under training conditions."""
        print("\n=== Testing Dataset Composer Generation ===")

        # Create composer similar to training setup
        composer = DefaultSyntheticComposer(
            seed=42,
            range_proportions={"short": 0.5, "medium": 0.3, "long": 0.2},
            generator_proportions={
                "short": {"forecast_pfn": 1.0},  # Only ForecastPFN for focused testing
                "medium": {"forecast_pfn": 1.0},
                "long": {"forecast_pfn": 1.0},
            },
        )

        print("Created composer successfully")

        # Generate multiple batches to stress test
        for i in range(5):
            print(f"\nGenerating batch {i + 1}...")
            try:
                batch, generator_name = composer.generate_batch(
                    batch_size=4, seed=42 + i
                )
                print(f"✓ Generated batch {i + 1} with generator: {generator_name}")
                print(f"  Batch size: {batch.batch_size}")
                print(f"  History length: {batch.history_length}")
                print(f"  Future length: {batch.future_length}")
                print(f"  Frequency: {batch.frequency}")
                print(f"  Start timestamps: {batch.start}")

                # Check for problematic timestamps
                latest_start = pd.Timestamp(max(batch.start))
                total_length = batch.history_length + batch.future_length

                if batch.frequency == Frequency.A:
                    estimated_end = latest_start + pd.DateOffset(years=total_length)
                elif batch.frequency == Frequency.Q:
                    estimated_end = latest_start + pd.DateOffset(
                        months=total_length * 3
                    )
                elif batch.frequency == Frequency.M:
                    estimated_end = latest_start + pd.DateOffset(months=total_length)
                else:
                    estimated_end = latest_start + pd.DateOffset(days=total_length)

                print(f"  Latest start: {latest_start}")
                print(f"  Estimated end: {estimated_end}")

                if estimated_end.year > 2250:
                    print(
                        f"⚠️  WARNING: Estimated end year {estimated_end.year} may cause overflow!"
                    )

                # Test time features computation
                history_features, target_features = compute_batch_time_features(
                    start=batch.start,
                    history_length=batch.history_length,
                    target_length=batch.future_length,
                    batch_size=batch.batch_size,
                    frequency=batch.frequency,
                )
                print(f"✓ Successfully computed time features for batch {i + 1}")

            except Exception as e:
                print(f"✗ Failed on batch {i + 1}: {e}")
                raise

    def test_extreme_parameters(self):
        """Test with extreme parameters that might trigger overflow."""
        print("\n=== Testing Extreme Parameters ===")

        extreme_cases = [
            # (frequency, history_length, future_length, description)
            (Frequency.A, 1024, 900, "Very long annual series"),
            (Frequency.Q, 1024, 900, "Very long quarterly series"),
            (Frequency.M, 1024, 900, "Very long monthly series"),
            (
                Frequency.A,
                2048,
                1800,
                "Extremely long annual series (should be capped)",
            ),
        ]

        for freq, hist_len, fut_len, description in extreme_cases:
            print(f"\nTesting: {description}")
            print(f"  Frequency: {freq}, History: {hist_len}, Future: {fut_len}")

            params = ForecastPFNGeneratorParams(
                global_seed=42,
                history_length=hist_len,
                future_length=fut_len,
                num_channels=1,
                frequency=freq,
            )

            wrapper = ForecastPFNGeneratorWrapper(params)

            try:
                batch = wrapper.generate_batch(batch_size=2, seed=42)
                print("✓ Generated batch successfully")
                print(f"  Actual history length: {batch.history_length}")
                print(f"  Actual future length: {batch.future_length}")
                print(f"  Start timestamps: {batch.start}")

                # Check if lengths were capped
                if batch.history_length < hist_len or batch.future_length < fut_len:
                    print("  ℹ️  Lengths were capped for safety")

                # Test time features computation
                history_features, target_features = compute_batch_time_features(
                    start=batch.start,
                    history_length=batch.history_length,
                    target_length=batch.future_length,
                    batch_size=batch.batch_size,
                    frequency=batch.frequency,
                )
                print("✓ Time features computed successfully")

            except Exception as e:
                print(f"✗ Failed for {description}: {e}")
                # Don't raise here, continue testing other cases
                continue

    def test_timestamp_bounds_validation(self):
        """Test timestamp bounds validation specifically."""
        print("\n=== Testing Timestamp Bounds Validation ===")

        # Create timestamps that might be problematic
        test_timestamps = [
            np.datetime64("2020-01-01"),
            np.datetime64("2050-01-01"),
            np.datetime64("2100-01-01"),
            np.datetime64("2200-01-01"),
            np.datetime64("1900-01-01"),
        ]

        for ts in test_timestamps:
            print(f"\nTesting with start timestamp: {ts}")

            try:
                history_features, target_features = compute_batch_time_features(
                    start=np.array([ts]),
                    history_length=50,
                    target_length=20,
                    batch_size=1,
                    frequency=Frequency.A,  # Annual is most problematic
                )
                print(f"✓ Successfully computed features for {ts}")

            except Exception as e:
                print(f"✗ Failed for timestamp {ts}: {e}")
                continue

    def test_constants_validation(self):
        """Test the constants used for timestamp generation."""
        print("\n=== Testing Constants Validation ===")

        from src.synthetic_generation.common.constants import (
            BASE_END,
            BASE_START,
            DEFAULT_END_DATE,
            DEFAULT_START_DATE,
        )

        print(f"DEFAULT_START_DATE: {DEFAULT_START_DATE}")
        print(f"DEFAULT_END_DATE: {DEFAULT_END_DATE}")
        print(f"BASE_START: {BASE_START} (ordinal)")
        print(f"BASE_END: {BASE_END} (ordinal)")

        # Convert back to dates for validation
        start_date = datetime.fromordinal(BASE_START)
        end_date = datetime.fromordinal(BASE_END)

        print(f"Converted BASE_START: {start_date}")
        print(f"Converted BASE_END: {end_date}")

        # Test range
        date_range_years = (end_date - start_date).days / 365.25
        print(f"Date range: {date_range_years:.1f} years")

        # Test if sampling from this range with beta distribution creates safe timestamps
        rng = np.random.default_rng(42)
        for i in range(10):
            from scipy.stats import beta

            sampled_ordinal = int(
                (BASE_END - BASE_START) * beta.rvs(5, 1, random_state=rng) + BASE_START
            )
            sampled_date = datetime.fromordinal(sampled_ordinal)
            print(f"  Sample {i + 1}: {sampled_date}")

            # Check if this timestamp + 100 years would be safe
            if sampled_date.year + 100 > 2250:
                print(
                    f"    ⚠️  WARNING: This timestamp + 100 years = {sampled_date.year + 100} (risky)"
                )

    def test_training_like_conditions(self):
        """Test under conditions that closely match actual training."""
        print("\n=== Testing Training-Like Conditions ===")

        # Test with parameters that would be used in actual training
        training_cases = [
            # Cases that might occur during training with long sequences
            {"frequency": Frequency.A, "history": 720, "future": 720, "batch_size": 8},
            {"frequency": Frequency.Q, "history": 1024, "future": 900, "batch_size": 8},
            {"frequency": Frequency.M, "history": 1024, "future": 900, "batch_size": 8},
            {
                "frequency": Frequency.A,
                "history": 1024,
                "future": 1024,
                "batch_size": 8,
            },
        ]

        for i, case in enumerate(training_cases):
            print(f"\nTesting training case {i + 1}: {case}")

            params = ForecastPFNGeneratorParams(
                global_seed=42 + i,
                history_length=case["history"],
                future_length=case["future"],
                num_channels=1,
                frequency=case["frequency"],
            )

            wrapper = ForecastPFNGeneratorWrapper(params)

            try:
                batch = wrapper.generate_batch(
                    batch_size=case["batch_size"], seed=42 + i
                )
                print("✓ Generated batch successfully")
                print(f"  Actual history length: {batch.history_length}")
                print(f"  Actual future length: {batch.future_length}")
                print(f"  Frequency: {batch.frequency}")
                print(
                    f"  Start timestamps range: {batch.start[0]} to {batch.start[-1]}"
                )

                # Check for timestamps that might cause overflow
                latest_start = pd.Timestamp(max(batch.start))
                total_length = batch.history_length + batch.future_length

                if batch.frequency == Frequency.A:
                    estimated_end = latest_start + pd.DateOffset(years=total_length)
                elif batch.frequency == Frequency.Q:
                    estimated_end = latest_start + pd.DateOffset(
                        months=total_length * 3
                    )
                elif batch.frequency == Frequency.M:
                    estimated_end = latest_start + pd.DateOffset(months=total_length)
                else:
                    estimated_end = latest_start + pd.DateOffset(days=total_length)

                print(f"  Latest start: {latest_start}")
                print(f"  Estimated end: {estimated_end}")

                if estimated_end.year > 2250:
                    print(
                        f"⚠️  WARNING: Estimated end year {estimated_end.year} may cause overflow!"
                    )

                # Test time features computation (this is where the overflow often occurs)
                history_features, target_features = compute_batch_time_features(
                    start=batch.start,
                    history_length=batch.history_length,
                    target_length=batch.future_length,
                    batch_size=batch.batch_size,
                    frequency=batch.frequency,
                )
                print("✓ Time features computed successfully")

            except Exception as e:
                print(f"✗ Failed for training case {i + 1}: {e}")
                print("  This might be the source of the training overflow!")
                raise


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestForecastPFNTimestampIssues()
    test_instance.test_constants_validation()
    test_instance.test_timestamp_bounds_validation()
    test_instance.test_forecast_pfn_direct_generation()
    test_instance.test_extreme_parameters()
    test_instance.test_dataset_composer_generation()
    test_instance.test_training_like_conditions()
