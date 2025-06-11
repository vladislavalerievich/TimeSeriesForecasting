import numpy as np
import pandas as pd

from src.data_handling.time_features import compute_batch_time_features
from src.synthetic_generation.common.constants import Frequency

# Configuration
batch_size = 1
history_length = 5
target_length = 3
K_max = 6
start_date = np.array([np.datetime64("2025-01-01")])


def test_time_features():
    """
    Test GluonTS time features for all frequencies in the Frequency Enum.
    Prints history and target time features for each frequency.
    """
    FREQUENCY_STR_MAPPING = {
        Frequency.M: "ME",
        Frequency.W: "W",
        Frequency.D: "D",
        Frequency.H: "h",
        Frequency.S: "s",
        Frequency.T5: "5min",
        Frequency.T10: "10min",
        Frequency.T15: "15min",
    }

    PERIOD_FREQUENCY_STR_MAPPING = {
        Frequency.M: "M",
        Frequency.W: "W",
        Frequency.D: "D",
        Frequency.H: "h",
        Frequency.S: "s",
        Frequency.T5: "5min",
        Frequency.T10: "10min",
        Frequency.T15: "15min",
    }

    for freq in Frequency:
        print(f"\n=== Testing Frequency: {freq.name} ({freq.value}) ===")

        # Generate timestamps for history and target
        freq_str = FREQUENCY_STR_MAPPING[freq]
        period_freq_str = PERIOD_FREQUENCY_STR_MAPPING[freq]

        hist_range = pd.date_range(
            start=start_date[0], periods=history_length, freq=freq_str
        )
        target_start = hist_range[-1] + pd.tseries.frequencies.to_offset(freq_str)
        targ_range = pd.date_range(
            start=target_start, periods=target_length, freq=freq_str
        )

        # Compute time features
        history_features, target_features = compute_batch_time_features(
            start=start_date,
            history_length=history_length,
            target_length=target_length,
            batch_size=batch_size,
            frequency=freq,
            K_max=K_max,
        )

        # Print history features
        print("\nHistory Time Features:")
        print(f"Shape: {history_features.shape} [batch_size, history_length, K_max]")
        for i in range(history_length):
            print(f"Timestamp: {hist_range[i]}")
            print(f"Features: {history_features[0, i, :].cpu().numpy()}")

        # Print target features
        print("\nTarget Time Features:")
        print(f"Shape: {target_features.shape} [batch_size, target_length, K_max]")
        for i in range(target_length):
            print(f"Timestamp: {targ_range[i]}")
            print(f"Features: {target_features[0, i, :].cpu().numpy()}")


if __name__ == "__main__":
    test_time_features()
