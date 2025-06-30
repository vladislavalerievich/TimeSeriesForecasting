# Add src to path for imports

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_handling.data_containers import Frequency
from src.data_handling.time_features import (
    TimeFeatureGenerator,
    compute_batch_time_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_basic_vs_enhanced_features():
    """Compare basic vs enhanced time features."""
    print("=== Testing Basic vs Enhanced Time Features ===\n")

    # Test parameters
    batch_size = 3
    history_length = 1024
    future_length = 900
    frequency = Frequency.H

    start = np.datetime64("2023-11-22")

    print("Test setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  History length: {history_length}")
    print(f"  Future length: {future_length}")
    print(f"  Frequency: {frequency.value}")
    print()

    # Test enhanced features with different configurations
    configs = [
        {
            "name": "Enhanced (no holidays)",
            "config": {
                "use_enhanced_features": True,
                "use_holiday_features": False,
                "auto_adjust_k_max": True,
                "k_max": 15,
            },
        },
        {
            "name": "Enhanced + Holidays (US Business)",
            "config": {
                "use_enhanced_features": True,
                "use_holiday_features": True,
                "holiday_set": "us_business",
                "holiday_kernel": "exponential",
                "auto_adjust_k_max": True,
                "k_max": 20,
            },
        },
        {
            "name": "Enhanced + Holidays (US Retail)",
            "config": {
                "use_enhanced_features": True,
                "use_holiday_features": True,
                "holiday_set": "us_retail",
                "holiday_kernel": "squared_exponential",
                "auto_adjust_k_max": True,
                "k_max": (6, 25),
            },
        },
    ]

    for test_config in configs:
        print(f"Computing {test_config['name']}...")
        enhanced_hist, enhanced_future = compute_batch_time_features(
            start=start,
            history_length=history_length,
            future_length=future_length,
            frequency=frequency,
            time_feature_config=test_config["config"],
        )

        print(
            f"Enhanced features shape - History: {enhanced_hist.shape}, Future: {enhanced_future.shape}"
        )
        print("Enhanced features example (first timestep):")
        print(f"  {enhanced_hist[0, :].detach().cpu().numpy()}")

        # Check for holiday effects (should be non-zero near holidays if holiday features enabled)
        if test_config["config"].get("use_holiday_features", False):
            holiday_effects = []
            # Check if any features show holiday patterns
            batch_features = enhanced_hist[:, :].detach().cpu().numpy()
            max_vals = np.max(np.abs(batch_features), axis=0)
            holiday_effects.append(max_vals)

            print("Holiday effect indicators (max absolute values per feature):")
            print(f"  {holiday_effects}")

        print()


def test_different_frequencies():
    """Test enhanced features across different frequencies."""
    print("=== Testing Different Frequencies ===\n")

    frequencies = [
        (Frequency.D, "Daily", 14, 7),  # 2 weeks history, 1 week prediction
        (Frequency.H, "Hourly", 48, 24),  # 2 days history, 1 day prediction
        (Frequency.T15, "15-minute", 96, 48),  # 1 day history, 12 hours prediction
        (Frequency.W, "Weekly", 8, 4),  # 8 weeks history, 4 weeks prediction
    ]

    start = np.array([datetime(2023, 6, 15)], dtype="datetime64[ns]")

    enhanced_config = {
        "use_enhanced_features": True,
        "use_holiday_features": True,
        "holiday_set": "us_business",
        "auto_adjust_k_max": True,
        "k_max": 20,
    }

    for freq, freq_name, hist_len, target_len in frequencies:
        print(f"Testing {freq_name} frequency ({freq.value})...")

        try:
            hist_features, target_features = compute_batch_time_features(
                start=start,
                history_length=hist_len,
                future_length=target_len,
                frequency=freq,
                time_feature_config=enhanced_config,
            )

            print(f"  Success! Shape: {hist_features.shape}")
            print(
                f"  Feature range: [{hist_features.min():.3f}, {hist_features.max():.3f}]"
            )
            print(
                f"  Non-zero features: {(hist_features != 0).sum().item()}/{hist_features.numel()}"
            )

        except Exception as e:
            print(f"  Error: {e}")

        print()


def test_feature_generator_directly():
    """Test the EnhancedTimeFeatureGenerator class directly."""
    print("=== Testing EnhancedTimeFeatureGenerator Directly ===\n")

    # Create test data
    dates = pd.date_range(start="2023-12-20", periods=48, freq="h")  # Christmas period
    period_idx = dates.to_period("h")

    configs = [
        {
            "name": "Basic",
            "config": {"use_enhanced_features": False, "use_holiday_features": False},
        },
        {
            "name": "Enhanced Only",
            "config": {"use_enhanced_features": True, "use_holiday_features": False},
        },
        {
            "name": "Holidays Only",
            "config": {"use_enhanced_features": False, "use_holiday_features": True},
        },
        {
            "name": "Full Enhanced",
            "config": {"use_enhanced_features": True, "use_holiday_features": True},
        },
    ]

    for test_config in configs:
        print(f"Testing {test_config['name']}:")
        generator = TimeFeatureGenerator(**test_config["config"])

        features = generator.compute_features(period_idx, dates, "h")

        print(f"  Features shape: {features.shape}")
        print(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Optimal K_max: {generator.get_optimal_k_max('h')}")

        # Show feature diversity (std dev across time)
        feature_stds = np.std(features, axis=0)
        print(f"  Feature diversity (std): {feature_stds}")
        print()


def test_holiday_kernels():
    """Test different holiday kernel functions."""
    print("=== Testing Holiday Kernels ===\n")

    # Christmas period
    dates = pd.date_range(start="2023-12-20", end="2023-12-30", freq="D")
    period_idx = dates.to_period("D")

    kernels = ["indicator", "exponential", "squared_exponential"]

    for kernel in kernels:
        print(f"Testing {kernel} kernel:")
        generator = TimeFeatureGenerator(
            use_enhanced_features=False,
            use_holiday_features=True,
            holiday_set="us_business",
            holiday_kernel=kernel,
            holiday_kernel_alpha=1.0,
        )

        features = generator.compute_features(period_idx, dates, "D")

        print(f"  Features shape: {features.shape}")
        print(
            f"  Christmas Day (Dec 25) features: {features[5, :]}"
        )  # Dec 25 should be index 5
        print(f"  Dec 24 features: {features[4, :]}")
        print(f"  Dec 26 features: {features[6, :]}")
        print()


def main():
    """Run all tests."""
    print("Enhanced Time Features Test Suite")
    print("=" * 50)
    print()

    try:
        test_basic_vs_enhanced_features()
        test_different_frequencies()
        test_feature_generator_directly()
        test_holiday_kernels()

        print("=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
