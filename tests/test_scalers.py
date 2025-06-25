from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_handling.scalers import MinMaxScaler, RobustScaler


def load_test_batch(batch_path: str):
    """Load test batch from disk."""
    print(f"Loading batch from: {batch_path}")
    batch = torch.load(batch_path, weights_only=False)

    print("Batch info:")
    print(f"  - history_values shape: {batch.history_values.shape}")
    print(f"  - future_values shape: {batch.future_values.shape}")
    print(f"  - batch_size: {batch.batch_size}")
    print(f"  - num_channels: {batch.num_channels}")
    print(f"  - history_length: {batch.history_length}")
    print(f"  - future_length: {batch.future_length}")

    if batch.history_mask is not None:
        print(f"  - history_mask shape: {batch.history_mask.shape}")
        print(f"  - history_mask valid ratio: {batch.history_mask.mean().item():.3f}")
    else:
        print("  - No history_mask present")

    return batch


def create_synthetic_mask(shape, missing_ratio=0.2, seed=42):
    """Create a synthetic mask for testing."""
    torch.manual_seed(seed)
    mask = torch.rand(shape) > missing_ratio
    return mask.float()


def compute_recovery_error(original, recovered, mask=None):
    """Compute recovery error metrics."""
    if mask is not None:
        # Only compute error for valid entries
        valid_mask = mask.bool()
        if valid_mask.dim() == 2:  # [batch_size, seq_len]
            valid_mask = valid_mask.unsqueeze(-1).expand_as(original)

        original_valid = original[valid_mask]
        recovered_valid = recovered[valid_mask]
    else:
        original_valid = original.flatten()
        recovered_valid = recovered.flatten()

    if len(original_valid) == 0:
        return {"mae": float("inf"), "mse": float("inf"), "max_error": float("inf")}

    abs_error = torch.abs(original_valid - recovered_valid)
    squared_error = (original_valid - recovered_valid) ** 2

    return {
        "mae": abs_error.mean().item(),
        "mse": squared_error.mean().item(),
        "max_error": abs_error.max().item(),
        "num_valid": len(original_valid),
    }


def print_statistics_info(statistics: Dict[str, torch.Tensor], scaler_name: str):
    """Print information about computed statistics."""
    print(f"\n{scaler_name} Statistics:")
    for key, tensor in statistics.items():
        print(f"  {key}:")
        print(f"    - Shape: {tensor.shape}")
        print(f"    - Mean: {tensor.mean().item():.6f}")
        print(f"    - Std: {tensor.std().item():.6f}")
        print(f"    - Min: {tensor.min().item():.6f}")
        print(f"    - Max: {tensor.max().item():.6f}")


def test_robust_scaler(batch, use_mask=False, mask=None):
    """Test RobustScaler comprehensively."""
    print("\n" + "=" * 60)
    print("TESTING ROBUST SCALER")
    print("=" * 60)

    scaler = RobustScaler(epsilon=1e-8)

    # Test with single sample first
    sample_idx = 0
    history_single = batch.history_values[
        sample_idx : sample_idx + 1
    ]  # Keep batch dimension

    if use_mask and mask is not None:
        mask_single = mask[sample_idx : sample_idx + 1]
        print(
            f"Using mask with {mask_single.sum().item()}/{mask_single.numel()} valid entries"
        )
    else:
        mask_single = None
        print("No mask used")

    print(f"Input data shape: {history_single.shape}")
    print(
        f"Input data range: [{history_single.min().item():.3f}, {history_single.max().item():.3f}]"
    )

    # Compute statistics
    statistics = scaler.compute_statistics(history_single, mask_single)
    print_statistics_info(statistics, "RobustScaler")

    # Scale the data
    scaled_data = scaler.scale(history_single, statistics)
    print(f"\nScaled data shape: {scaled_data.shape}")
    print(
        f"Scaled data range: [{scaled_data.min().item():.3f}, {scaled_data.max().item():.3f}]"
    )
    print(f"Scaled data mean: {scaled_data.mean().item():.6f}")
    print(f"Scaled data std: {scaled_data.std().item():.6f}")

    # Test inverse scaling
    recovered_data = scaler.inverse_scale(scaled_data, statistics)
    print(f"\nRecovered data shape: {recovered_data.shape}")
    print(
        f"Recovered data range: [{recovered_data.min().item():.3f}, {recovered_data.max().item():.3f}]"
    )

    # Compute recovery error
    recovery_error = compute_recovery_error(history_single, recovered_data, mask_single)
    print("\nRecovery Error Metrics:")
    for metric, value in recovery_error.items():
        print(f"  {metric}: {value:.10f}")

    # Test with full batch
    print("\n--- Testing with full batch ---")
    batch_mask = mask if use_mask else None
    batch_statistics = scaler.compute_statistics(batch.history_values, batch_mask)
    batch_scaled = scaler.scale(batch.history_values, batch_statistics)
    batch_recovered = scaler.inverse_scale(batch_scaled, batch_statistics)

    batch_error = compute_recovery_error(
        batch.history_values, batch_recovered, batch_mask
    )
    print("Batch Recovery Error Metrics:")
    for metric, value in batch_error.items():
        print(f"  {metric}: {value:.10f}")

    return {
        "scaler": scaler,
        "statistics": statistics,
        "scaled_data": scaled_data,
        "recovered_data": recovered_data,
        "recovery_error": recovery_error,
        "batch_recovery_error": batch_error,
    }


def test_minmax_scaler(batch, use_mask=False, mask=None):
    """Test MinMaxScaler comprehensively."""
    print("\n" + "=" * 60)
    print("TESTING MINMAX SCALER")
    print("=" * 60)

    scaler = MinMaxScaler(epsilon=1e-8)

    # Test with single sample first
    sample_idx = 0
    history_single = batch.history_values[
        sample_idx : sample_idx + 1
    ]  # Keep batch dimension

    if use_mask and mask is not None:
        mask_single = mask[sample_idx : sample_idx + 1]
        print(
            f"Using mask with {mask_single.sum().item()}/{mask_single.numel()} valid entries"
        )
    else:
        mask_single = None
        print("No mask used")

    print(f"Input data shape: {history_single.shape}")
    print(
        f"Input data range: [{history_single.min().item():.3f}, {history_single.max().item():.3f}]"
    )

    # Compute statistics
    statistics = scaler.compute_statistics(history_single, mask_single)
    print_statistics_info(statistics, "MinMaxScaler")

    # Scale the data
    scaled_data = scaler.scale(history_single, statistics)
    print(f"\nScaled data shape: {scaled_data.shape}")
    print(
        f"Scaled data range: [{scaled_data.min().item():.3f}, {scaled_data.max().item():.3f}]"
    )
    print(f"Scaled data mean: {scaled_data.mean().item():.6f}")
    print(f"Scaled data std: {scaled_data.std().item():.6f}")

    # Verify data is in [-1, 1] range
    if scaled_data.min() >= -1.001 and scaled_data.max() <= 1.001:
        print("✓ Scaled data is properly in [-1, 1] range")
    else:
        print("✗ WARNING: Scaled data is outside [-1, 1] range!")

    # Test inverse scaling
    recovered_data = scaler.inverse_scale(scaled_data, statistics)
    print(f"\nRecovered data shape: {recovered_data.shape}")
    print(
        f"Recovered data range: [{recovered_data.min().item():.3f}, {recovered_data.max().item():.3f}]"
    )

    # Compute recovery error
    recovery_error = compute_recovery_error(history_single, recovered_data, mask_single)
    print("\nRecovery Error Metrics:")
    for metric, value in recovery_error.items():
        print(f"  {metric}: {value:.10f}")

    # Test with full batch
    print("\n--- Testing with full batch ---")
    batch_mask = mask if use_mask else None
    batch_statistics = scaler.compute_statistics(batch.history_values, batch_mask)
    batch_scaled = scaler.scale(batch.history_values, batch_statistics)
    batch_recovered = scaler.inverse_scale(batch_scaled, batch_statistics)

    # Verify batch scaling range
    if batch_scaled.min() >= -1.001 and batch_scaled.max() <= 1.001:
        print("✓ Batch scaled data is properly in [-1, 1] range")
    else:
        print("✗ WARNING: Batch scaled data is outside [-1, 1] range!")

    batch_error = compute_recovery_error(
        batch.history_values, batch_recovered, batch_mask
    )
    print("Batch Recovery Error Metrics:")
    for metric, value in batch_error.items():
        print(f"  {metric}: {value:.10f}")

    return {
        "scaler": scaler,
        "statistics": statistics,
        "scaled_data": scaled_data,
        "recovered_data": recovered_data,
        "recovery_error": recovery_error,
        "batch_recovery_error": batch_error,
    }


def test_edge_cases(batch):
    """Test edge cases and robustness."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # Test with all-masked data
    print("\n--- Testing with completely masked data ---")
    all_mask = (
        torch.zeros_like(batch.history_mask)
        if batch.history_mask is not None
        else torch.zeros(batch.batch_size, batch.history_length)
    )

    robust_scaler = RobustScaler()
    minmax_scaler = MinMaxScaler()

    robust_stats = robust_scaler.compute_statistics(batch.history_values, all_mask)
    minmax_stats = minmax_scaler.compute_statistics(batch.history_values, all_mask)

    print("RobustScaler with all-masked data:")
    print(f"  Median defaults: {robust_stats['median'][0, 0, :3]}")  # First 3 channels
    print(f"  IQR defaults: {robust_stats['iqr'][0, 0, :3]}")

    print("MinMaxScaler with all-masked data:")
    print(f"  Min defaults: {minmax_stats['min'][0, 0, :3]}")
    print(f"  Max defaults: {minmax_stats['max'][0, 0, :3]}")

    # Test with constant data
    print("\n--- Testing with constant data ---")
    constant_data = torch.ones_like(batch.history_values) * 5.0

    robust_stats_const = robust_scaler.compute_statistics(constant_data)
    minmax_stats_const = minmax_scaler.compute_statistics(constant_data)

    scaled_const_robust = robust_scaler.scale(constant_data, robust_stats_const)
    scaled_const_minmax = minmax_scaler.scale(constant_data, minmax_stats_const)

    print(
        f"Constant data (value=5.0) scaled by RobustScaler: {scaled_const_robust[0, 0, 0].item():.6f}"
    )
    print(
        f"Constant data (value=5.0) scaled by MinMaxScaler: {scaled_const_minmax[0, 0, 0].item():.6f}"
    )

    # Test recovery
    recovered_const_robust = robust_scaler.inverse_scale(
        scaled_const_robust, robust_stats_const
    )
    recovered_const_minmax = minmax_scaler.inverse_scale(
        scaled_const_minmax, minmax_stats_const
    )

    print(
        f"Recovered constant data (RobustScaler): {recovered_const_robust[0, 0, 0].item():.6f}"
    )
    print(
        f"Recovered constant data (MinMaxScaler): {recovered_const_minmax[0, 0, 0].item():.6f}"
    )


def compare_scalers(batch, mask=None):
    """Compare the behavior of both scalers."""
    print("\n" + "=" * 60)
    print("COMPARING SCALERS")
    print("=" * 60)

    sample_idx = 0
    history_single = batch.history_values[sample_idx : sample_idx + 1]
    mask_single = mask[sample_idx : sample_idx + 1] if mask is not None else None

    # Initialize scalers
    robust_scaler = RobustScaler()
    minmax_scaler = MinMaxScaler()

    # Compute statistics
    robust_stats = robust_scaler.compute_statistics(history_single, mask_single)
    minmax_stats = minmax_scaler.compute_statistics(history_single, mask_single)

    # Scale data
    robust_scaled = robust_scaler.scale(history_single, robust_stats)
    minmax_scaled = minmax_scaler.scale(history_single, minmax_stats)

    print("Original data statistics:")
    print(f"  Mean: {history_single.mean().item():.6f}")
    print(f"  Std: {history_single.std().item():.6f}")
    print(f"  Min: {history_single.min().item():.6f}")
    print(f"  Max: {history_single.max().item():.6f}")

    print("\nRobust scaled data statistics:")
    print(f"  Mean: {robust_scaled.mean().item():.6f}")
    print(f"  Std: {robust_scaled.std().item():.6f}")
    print(f"  Min: {robust_scaled.min().item():.6f}")
    print(f"  Max: {robust_scaled.max().item():.6f}")

    print("\nMinMax scaled data statistics:")
    print(f"  Mean: {minmax_scaled.mean().item():.6f}")
    print(f"  Std: {minmax_scaled.std().item():.6f}")
    print(f"  Min: {minmax_scaled.min().item():.6f}")
    print(f"  Max: {minmax_scaled.max().item():.6f}")

    # Test correlation between scaled versions
    robust_flat = robust_scaled.flatten()
    minmax_flat = minmax_scaled.flatten()
    correlation = torch.corrcoef(torch.stack([robust_flat, minmax_flat]))[0, 1]
    print(f"\nCorrelation between scaled versions: {correlation.item():.6f}")


def plot_scaler_comparison(
    batch, scaler, scaler_name, sample_idx=0, save_dir="outputs/plots/scalers_test"
):
    """
    Create visualization comparing original, scaled, and rescaled data.

    Args:
        batch: BatchTimeSeriesContainer with the data
        scaler: Scaler instance (RobustScaler or MinMaxScaler)
        scaler_name: Name for the scaler (for plot titles and filenames)
        sample_idx: Index of the sample to plot
        save_dir: Directory to save plots
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Extract single sample data
    history_values = batch.history_values[
        sample_idx : sample_idx + 1
    ]  # Keep batch dimension
    future_values = batch.future_values[sample_idx : sample_idx + 1]

    # Use mask if available
    history_mask = None
    future_mask = None
    if batch.history_mask is not None:
        history_mask = batch.history_mask[sample_idx : sample_idx + 1]
    if batch.future_mask is not None:
        future_mask = batch.future_mask[sample_idx : sample_idx + 1]

    # Compute statistics from history
    statistics = scaler.compute_statistics(history_values, history_mask)

    # Scale both history and future
    scaled_history = scaler.scale(history_values, statistics)
    scaled_future = scaler.scale(future_values, statistics)

    # Rescale both back to original scale
    rescaled_history = scaler.inverse_scale(scaled_history, statistics)
    rescaled_future = scaler.inverse_scale(scaled_future, statistics)

    # Convert to numpy for plotting (remove batch dimension)
    history_np = history_values[0].cpu().numpy()  # [seq_len, num_channels]
    future_np = future_values[0].cpu().numpy()
    scaled_history_np = scaled_history[0].cpu().numpy()
    scaled_future_np = scaled_future[0].cpu().numpy()
    rescaled_history_np = rescaled_history[0].cpu().numpy()
    rescaled_future_np = rescaled_future[0].cpu().numpy()

    # Create masks for plotting
    history_mask_np = None
    future_mask_np = None
    if history_mask is not None:
        history_mask_np = history_mask[0].cpu().numpy().astype(bool)
    if future_mask is not None:
        future_mask_np = future_mask[0].cpu().numpy().astype(bool)

    # Get dimensions
    history_len = history_np.shape[0]
    future_len = future_np.shape[0]
    num_channels = history_np.shape[1]

    # Create time indices
    history_time = np.arange(history_len)
    future_time = np.arange(history_len, history_len + future_len)

    # Determine number of channels to plot (max 6 for readability)
    channels_to_plot = min(num_channels, 6)

    # Create figure with subplots
    fig, axes = plt.subplots(3, channels_to_plot, figsize=(4 * channels_to_plot, 12))
    if channels_to_plot == 1:
        axes = axes.reshape(3, 1)

    # Plot titles
    plot_titles = [
        f"Original Data - {scaler_name}",
        f"Scaled Data - {scaler_name}",
        f"Rescaled Data - {scaler_name}",
    ]

    for channel in range(channels_to_plot):
        for plot_idx, (hist_data, fut_data, title) in enumerate(
            [
                (history_np[:, channel], future_np[:, channel], plot_titles[0]),
                (
                    scaled_history_np[:, channel],
                    scaled_future_np[:, channel],
                    plot_titles[1],
                ),
                (
                    rescaled_history_np[:, channel],
                    rescaled_future_np[:, channel],
                    plot_titles[2],
                ),
            ]
        ):
            ax = axes[plot_idx, channel]

            # Plot history data
            if history_mask_np is not None:
                # Plot valid and invalid points separately
                valid_hist = hist_data.copy()
                valid_hist[~history_mask_np] = np.nan
                invalid_hist = hist_data.copy()
                invalid_hist[history_mask_np] = np.nan

                ax.plot(
                    history_time,
                    valid_hist,
                    "b-",
                    linewidth=2,
                    label="History (valid)",
                    alpha=0.8,
                )
                ax.scatter(
                    history_time[~history_mask_np],
                    invalid_hist[~history_mask_np],
                    c="red",
                    s=20,
                    alpha=0.6,
                    label="History (masked)",
                    zorder=5,
                )
            else:
                ax.plot(
                    history_time,
                    hist_data,
                    "b-",
                    linewidth=2,
                    label="History",
                    alpha=0.8,
                )

            # Plot future data
            if future_mask_np is not None:
                # Plot valid and invalid points separately
                valid_fut = fut_data.copy()
                valid_fut[~future_mask_np] = np.nan
                invalid_fut = fut_data.copy()
                invalid_fut[future_mask_np] = np.nan

                ax.plot(
                    future_time,
                    valid_fut,
                    "g-",
                    linewidth=2,
                    label="Future (valid)",
                    alpha=0.8,
                )
                ax.scatter(
                    future_time[~future_mask_np],
                    invalid_fut[~future_mask_np],
                    c="orange",
                    s=20,
                    alpha=0.6,
                    label="Future (masked)",
                    zorder=5,
                )
            else:
                ax.plot(
                    future_time, fut_data, "g-", linewidth=2, label="Future", alpha=0.8
                )

            # Add vertical line to separate history and future
            ax.axvline(
                x=history_len - 0.5,
                color="black",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )

            # Formatting
            ax.set_title(f"{title}\nChannel {channel}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Add statistics text
            if plot_idx == 1:  # Scaled data
                y_min, y_max = ax.get_ylim()
                stats_text = f"Range: [{hist_data.min():.2f}, {hist_data.max():.2f}]\nMean: {hist_data.mean():.3f}"
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

    # Add overall title
    fig.suptitle(
        f"{scaler_name} Scaling Comparison - Sample {sample_idx}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    # Save plot
    filename = (
        f"{scaler_name.lower().replace(' ', '_')}_comparison_sample_{sample_idx}.png"
    )
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {scaler_name} comparison plot to: {filepath}")

    # Compute and return recovery metrics
    recovery_metrics = {
        "history_recovery_error": compute_recovery_error(
            history_values, rescaled_history, history_mask
        ),
        "future_recovery_error": compute_recovery_error(
            future_values, rescaled_future, future_mask
        ),
    }

    return recovery_metrics


def test_robust_scaler_with_plots(
    batch, sample_idx=0, save_dir="outputs/plots/scalers_test"
):
    """Test RobustScaler with comprehensive plots."""
    print("\n" + "=" * 60)
    print("TESTING ROBUST SCALER WITH PLOTS")
    print("=" * 60)

    scaler = RobustScaler(epsilon=1e-8)

    # Create plots and get recovery metrics
    recovery_metrics = plot_scaler_comparison(
        batch, scaler, "Robust Scaler", sample_idx, save_dir
    )

    # Print recovery metrics
    print(f"\nRecovery Metrics for Sample {sample_idx}:")
    print("History Recovery:")
    for metric, value in recovery_metrics["history_recovery_error"].items():
        print(f"  {metric}: {value:.10f}")

    print("Future Recovery:")
    for metric, value in recovery_metrics["future_recovery_error"].items():
        print(f"  {metric}: {value:.10f}")

    # Additional analysis
    sample_history = batch.history_values[sample_idx : sample_idx + 1]
    sample_future = batch.future_values[sample_idx : sample_idx + 1]

    history_mask = (
        batch.history_mask[sample_idx : sample_idx + 1]
        if batch.history_mask is not None
        else None
    )

    # Compute statistics and scaling info
    statistics = scaler.compute_statistics(sample_history, history_mask)
    scaled_history = scaler.scale(sample_history, statistics)
    scaled_future = scaler.scale(sample_future, statistics)

    print("\nScaling Statistics:")
    print(
        f"  Median range: [{statistics['median'].min().item():.3f}, {statistics['median'].max().item():.3f}]"
    )
    print(
        f"  IQR range: [{statistics['iqr'].min().item():.3f}, {statistics['iqr'].max().item():.3f}]"
    )
    print(
        f"  Scaled history range: [{scaled_history.min().item():.3f}, {scaled_history.max().item():.3f}]"
    )
    print(
        f"  Scaled future range: [{scaled_future.min().item():.3f}, {scaled_future.max().item():.3f}]"
    )

    return recovery_metrics


def test_minmax_scaler_with_plots(
    batch, sample_idx=0, save_dir="outputs/plots/scalers_test"
):
    """Test MinMaxScaler with comprehensive plots."""
    print("\n" + "=" * 60)
    print("TESTING MINMAX SCALER WITH PLOTS")
    print("=" * 60)

    scaler = MinMaxScaler(epsilon=1e-8)

    # Create plots and get recovery metrics
    recovery_metrics = plot_scaler_comparison(
        batch, scaler, "MinMax Scaler", sample_idx, save_dir
    )

    # Print recovery metrics
    print(f"\nRecovery Metrics for Sample {sample_idx}:")
    print("History Recovery:")
    for metric, value in recovery_metrics["history_recovery_error"].items():
        print(f"  {metric}: {value:.10f}")

    print("Future Recovery:")
    for metric, value in recovery_metrics["future_recovery_error"].items():
        print(f"  {metric}: {value:.10f}")

    # Additional analysis
    sample_history = batch.history_values[sample_idx : sample_idx + 1]
    sample_future = batch.future_values[sample_idx : sample_idx + 1]

    history_mask = (
        batch.history_mask[sample_idx : sample_idx + 1]
        if batch.history_mask is not None
        else None
    )

    # Compute statistics and scaling info
    statistics = scaler.compute_statistics(sample_history, history_mask)
    scaled_history = scaler.scale(sample_history, statistics)
    scaled_future = scaler.scale(sample_future, statistics)

    print("\nScaling Statistics:")
    print(
        f"  Min range: [{statistics['min'].min().item():.3f}, {statistics['min'].max().item():.3f}]"
    )
    print(
        f"  Max range: [{statistics['max'].min().item():.3f}, {statistics['max'].max().item():.3f}]"
    )
    print(
        f"  Scaled history range: [{scaled_history.min().item():.3f}, {scaled_history.max().item():.3f}]"
    )
    print(
        f"  Scaled future range: [{scaled_future.min().item():.3f}, {scaled_future.max().item():.3f}]"
    )

    # Verify [-1, 1] range
    if scaled_history.min() >= -1.001 and scaled_history.max() <= 1.001:
        print("  ✓ Scaled history is properly in [-1, 1] range")
    else:
        print("  ✗ WARNING: Scaled history is outside [-1, 1] range!")

    if scaled_future.min() >= -1.001 and scaled_future.max() <= 1.001:
        print("  ✓ Scaled future is properly in [-1, 1] range")
    else:
        print("  ✗ WARNING: Scaled future is outside [-1, 1] range!")

    return recovery_metrics


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    # Configuration
    batch_path = "data/synthetic_validation_dataset/batch_00001.pt"

    try:
        # Load batch
        batch = load_test_batch(batch_path)

        # Create synthetic mask for testing
        mask_shape = (batch.batch_size, batch.history_length)
        synthetic_mask = create_synthetic_mask(mask_shape, missing_ratio=0.15, seed=42)
        print(
            f"\nCreated synthetic mask with {synthetic_mask.mean().item():.3f} valid ratio"
        )

        # Test 1: RobustScaler without mask
        print("\n" + "=" * 80)
        print("TEST 1: ROBUST SCALER WITHOUT MASK")
        print("=" * 80)
        robust_results_no_mask = test_robust_scaler(batch, use_mask=False)

        # Test 2: RobustScaler with mask
        print("\n" + "=" * 80)
        print("TEST 2: ROBUST SCALER WITH MASK")
        print("=" * 80)
        robust_results_with_mask = test_robust_scaler(
            batch, use_mask=True, mask=synthetic_mask
        )

        # Test 3: MinMaxScaler without mask
        print("\n" + "=" * 80)
        print("TEST 3: MINMAX SCALER WITHOUT MASK")
        print("=" * 80)
        minmax_results_no_mask = test_minmax_scaler(batch, use_mask=False)

        # Test 4: MinMaxScaler with mask
        print("\n" + "=" * 80)
        print("TEST 4: MINMAX SCALER WITH MASK")
        print("=" * 80)
        minmax_results_with_mask = test_minmax_scaler(
            batch, use_mask=True, mask=synthetic_mask
        )

        # Test 5: Edge cases
        test_edge_cases(batch)

        # Test 6: Compare scalers
        compare_scalers(batch, mask=synthetic_mask)

        # Test 7: RobustScaler with plots
        print("\n" + "=" * 80)
        print("TEST 7: ROBUST SCALER WITH VISUALIZATION")
        print("=" * 80)
        robust_plot_results = test_robust_scaler_with_plots(batch, sample_idx=0)

        # Test 8: MinMaxScaler with plots
        print("\n" + "=" * 80)
        print("TEST 8: MINMAX SCALER WITH VISUALIZATION")
        print("=" * 80)
        minmax_plot_results = test_minmax_scaler_with_plots(batch, sample_idx=0)

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        print("\nRecovery Error Summary:")
        print(
            f"RobustScaler (no mask) - MAE: {robust_results_no_mask['recovery_error']['mae']:.2e}, MSE: {robust_results_no_mask['recovery_error']['mse']:.2e}"
        )
        print(
            f"RobustScaler (w/ mask) - MAE: {robust_results_with_mask['recovery_error']['mae']:.2e}, MSE: {robust_results_with_mask['recovery_error']['mse']:.2e}"
        )
        print(
            f"MinMaxScaler (no mask) - MAE: {minmax_results_no_mask['recovery_error']['mae']:.2e}, MSE: {minmax_results_no_mask['recovery_error']['mse']:.2e}"
        )
        print(
            f"MinMaxScaler (w/ mask) - MAE: {minmax_results_with_mask['recovery_error']['mae']:.2e}, MSE: {minmax_results_with_mask['recovery_error']['mse']:.2e}"
        )

        print("\nBatch Recovery Error Summary:")
        print(
            f"RobustScaler (no mask) - MAE: {robust_results_no_mask['batch_recovery_error']['mae']:.2e}"
        )
        print(
            f"RobustScaler (w/ mask) - MAE: {robust_results_with_mask['batch_recovery_error']['mae']:.2e}"
        )
        print(
            f"MinMaxScaler (no mask) - MAE: {minmax_results_no_mask['batch_recovery_error']['mae']:.2e}"
        )
        print(
            f"MinMaxScaler (w/ mask) - MAE: {minmax_results_with_mask['batch_recovery_error']['mae']:.2e}"
        )

        # Check if all tests passed
        all_errors = [
            robust_results_no_mask["recovery_error"]["mae"],
            robust_results_with_mask["recovery_error"]["mae"],
            minmax_results_no_mask["recovery_error"]["mae"],
            minmax_results_with_mask["recovery_error"]["mae"],
        ]

        max_error = max(all_errors)
        if max_error < 1e-6:
            print(f"\n✓ ALL TESTS PASSED! Maximum recovery error: {max_error:.2e}")
        else:
            print(f"\n⚠ Some tests have high recovery error. Maximum: {max_error:.2e}")

        return {
            "robust_no_mask": robust_results_no_mask,
            "robust_with_mask": robust_results_with_mask,
            "minmax_no_mask": minmax_results_no_mask,
            "minmax_with_mask": minmax_results_with_mask,
            "robust_plot_results": robust_plot_results,
            "minmax_plot_results": minmax_plot_results,
        }

    except FileNotFoundError:
        print(f"Error: Could not find batch file at {batch_path}")
        print("Please ensure the file exists or update the path.")
        return None
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    results = run_comprehensive_tests()
