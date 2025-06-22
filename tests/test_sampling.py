"""
Test script to verify that the improved _sample_parameters method
ensures history_length >= future_length across different parameter types.
"""

import sys

sys.path.append("src")

from src.synthetic_generation.abstract_classes import GeneratorWrapper
from src.synthetic_generation.generator_params import (
    GeneratorParams,
    LongRangeGeneratorParams,
    MediumRangeGeneratorParams,
    ShortRangeGeneratorParams,
)


class TestGeneratorWrapper(GeneratorWrapper):
    """Test wrapper to access the _sample_parameters method."""

    def generate_batch(self, batch_size: int, seed=None, **kwargs):
        pass


def test_parameter_sampling():
    """Test parameter sampling for different parameter configurations."""

    # Test configurations
    test_configs = [
        # Base GeneratorParams with default ranges
        GeneratorParams(),
        # ShortRangeGeneratorParams
        ShortRangeGeneratorParams(),
        # MediumRangeGeneratorParams
        MediumRangeGeneratorParams(),
        # LongRangeGeneratorParams
        LongRangeGeneratorParams(),
        # Custom parameters with different types
        GeneratorParams(
            history_length=100,  # int
            future_length=(10, 50),  # tuple
        ),
        GeneratorParams(
            history_length=(200, 500),  # tuple
            future_length=[20, 30, 40, 50],  # list
        ),
    ]

    for i, params in enumerate(test_configs):
        print(f"\n=== Test Configuration {i + 1}: {params.__class__.__name__} ===")
        print(f"history_length: {params.history_length}")
        print(f"future_length: {params.future_length}")

        wrapper = TestGeneratorWrapper(params)

        # Sample parameters multiple times to verify constraint
        violations = 0
        total_samples = 100

        for _ in range(total_samples):
            sampled = wrapper._sample_parameters()
            hist_len = sampled["history_length"]
            fut_len = sampled["future_length"]

            if hist_len < fut_len:
                violations += 1
                print(f"VIOLATION: history_length={hist_len} < future_length={fut_len}")

        if violations == 0:
            print(
                f"✅ All {total_samples} samples satisfied history_length >= future_length"
            )
        else:
            print(f"❌ Found {violations}/{total_samples} violations")

        # Show some sample results
        sample = wrapper._sample_parameters()
        print(
            f"Sample result: history_length={sample['history_length']}, future_length={sample['future_length']}, num_channels={sample['num_channels']}"
        )


if __name__ == "__main__":
    test_parameter_sampling()
