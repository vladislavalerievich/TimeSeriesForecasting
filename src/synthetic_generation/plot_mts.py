from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data_handling.time_series_data_structure import TimeSeriesData


def plot_synthetic_mts(
    ts_data: TimeSeriesData,
    sample_idx: int = 0,
    channel_indices: Optional[List[int]] = None,
    max_channels_to_plot: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Synthetic Multivariate Time Series",
    subplot_titles: Optional[List[str]] = None,
    show_target: bool = True,
    highlight_target: bool = True,
    show_time_features: bool = False,
    legend_loc: str = "best",
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> Figure:
    """
    Plot a synthetic multivariate time series from TimeSeriesData.

    Parameters
    ----------
    ts_data : TimeSeriesData
        The time series data to plot
    sample_idx : int
        Index of the sample in the batch to plot
    channel_indices : List[int], optional
        Indices of channels to plot. If None, plots the first max_channels_to_plot channels
    max_channels_to_plot : int
        Maximum number of channels to plot if channel_indices is None
    figsize : Tuple[int, int]
        Figure size in inches
    title : str
        Main title for the plot
    subplot_titles : List[str], optional
        Titles for individual channel subplots
    show_target : bool
        Whether to show target values
    highlight_target : bool
        Whether to highlight the target region with different color
    show_time_features : bool
        Whether to show time features in a separate subplot
    legend_loc : str
        Legend location
    output_file : str, optional
        If provided, save the plot to this file path
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot with plt.show()

    Returns
    -------
    Figure
        The matplotlib figure object
    """
    # Extract data for the specified sample
    history_values = ts_data.history_values[sample_idx].detach().cpu().numpy()
    history_ts = ts_data.history_ts[sample_idx].detach().cpu().numpy()

    if show_target:
        target_values = ts_data.target_values[sample_idx].detach().cpu().numpy()
        target_ts = ts_data.target_ts[sample_idx].detach().cpu().numpy()

    # Get non-zero length of history and target (for variable length data)
    history_len = history_values.shape[0]
    for i in range(history_len - 1, 0, -1):
        if not np.all(history_values[i] == 0):
            history_len = i + 1
            break

    target_len = 0
    if show_target:
        target_len = target_values.shape[0]
        for i in range(target_len - 1, 0, -1):
            if not np.all(target_values[i] == 0):
                target_len = i + 1
                break

    # Create time indices for x-axis
    history_time_idx = np.arange(history_len)
    if show_target:
        target_time_idx = np.arange(history_len, history_len + target_len)

    # Select channels to plot
    num_channels = history_values.shape[1]
    if channel_indices is None:
        num_to_plot = min(num_channels, max_channels_to_plot)
        channel_indices = list(range(num_to_plot))
    else:
        num_to_plot = len(channel_indices)

    # Create subplots
    if show_time_features:
        fig, axs = plt.subplots(num_to_plot + 1, 1, figsize=figsize, sharex=True)
        ts_ax = axs[-1]  # Last subplot for time features
        value_axs = axs[:-1]
    else:
        fig, axs = plt.subplots(num_to_plot, 1, figsize=figsize, sharex=True)
        value_axs = axs if num_to_plot > 1 else [axs]

    fig.suptitle(f"{title} (Task: {ts_data.task[sample_idx].item()})", fontsize=16)

    # Plot each selected channel
    for i, channel_idx in enumerate(channel_indices):
        ax = value_axs[i]

        # Plot history values
        ax.plot(
            history_time_idx,
            history_values[:history_len, channel_idx],
            label="History",
            color="blue",
        )

        # Plot target values if requested
        if show_target:
            ax.plot(
                target_time_idx,
                target_values[:target_len, channel_idx],
                label="Target",
                color="green" if highlight_target else "blue",
                linestyle="--" if highlight_target else "-",
            )

            # Add vertical line separating history and target
            ax.axvline(x=history_len - 0.5, color="red", linestyle=":", alpha=0.7)

        # Set title and labels for subplot
        if subplot_titles and i < len(subplot_titles):
            ax.set_title(subplot_titles[i])
        else:
            ax.set_title(f"Channel {channel_idx}")

        ax.set_ylabel("Value")

        # Only show legend for the first subplot to avoid clutter
        if i == 0:
            ax.legend(loc=legend_loc)

    # Plot time features if requested
    if show_time_features:
        num_ts_features = history_ts.shape[1]
        for j in range(num_ts_features):
            ts_ax.plot(
                history_time_idx, history_ts[:history_len, j], label=f"Feature {j}"
            )

            if show_target:
                ts_ax.plot(target_time_idx, target_ts[:target_len, j], linestyle="--")

        ts_ax.set_title("Time Features")
        ts_ax.set_ylabel("Value")
        ts_ax.legend(loc=legend_loc)

    # Set common x-axis label
    axs[-1].set_xlabel("Time Steps")

    # Add gray background for target region if highlighting
    if show_target and highlight_target:
        for ax in value_axs:
            ax.axvspan(
                history_len - 0.5,
                history_len + target_len,
                alpha=0.1,
                color="gray",
                label="Target Region",
            )

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path is provided
    if output_file is not None:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    # Show plot if requested
    if show:
        plt.show()

    return fig


def plot_multiple_channels(
    ts_data: TimeSeriesData,
    sample_idx: int = 0,
    channel_indices: Optional[List[int]] = None,
    max_channels: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multiple Channels Overlay",
    show_target: bool = True,
    use_colors: bool = True,
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> Figure:
    """
    Plot multiple channels of a synthetic multivariate time series on a single plot.

    Parameters
    ----------
    ts_data : TimeSeriesData
        The time series data to plot
    sample_idx : int
        Index of the sample in the batch to plot
    channel_indices : List[int], optional
        Indices of channels to plot. If None, plots the first max_channels
    max_channels : int
        Maximum number of channels to plot if channel_indices is None
    figsize : Tuple[int, int]
        Figure size in inches
    title : str
        Title for the plot
    show_target : bool
        Whether to show target values
    use_colors : bool
        Whether to use different colors for different channels
    output_file : str, optional
        If provided, save the plot to this file path
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot with plt.show()

    Returns
    -------
    Figure
        The matplotlib figure object
    """
    # Extract data for the specified sample
    history_values = ts_data.history_values[sample_idx].detach().cpu().numpy()

    if show_target:
        target_values = ts_data.target_values[sample_idx].detach().cpu().numpy()

    # Get non-zero length of history and target (for variable length data)
    history_len = history_values.shape[0]
    for i in range(history_len - 1, 0, -1):
        if not np.all(history_values[i] == 0):
            history_len = i + 1
            break

    target_len = 0
    if show_target:
        target_len = target_values.shape[0]
        for i in range(target_len - 1, 0, -1):
            if not np.all(target_values[i] == 0):
                target_len = i + 1
                break

    # Create time indices for x-axis
    history_time_idx = np.arange(history_len)
    if show_target:
        target_time_idx = np.arange(history_len, history_len + target_len)

    # Select channels to plot
    num_channels = history_values.shape[1]
    if channel_indices is None:
        num_to_plot = min(num_channels, max_channels)
        channel_indices = list(range(num_to_plot))
    else:
        num_to_plot = len(channel_indices)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each selected channel
    for i, channel_idx in enumerate(channel_indices):
        if use_colors:
            # Plot history values
            ax.plot(
                history_time_idx,
                history_values[:history_len, channel_idx],
                label=f"Channel {channel_idx}",
            )

            # Plot target values if requested
            if show_target:
                ax.plot(
                    target_time_idx,
                    target_values[:target_len, channel_idx],
                    linestyle="--",
                )
        else:
            # Use same color but different line styles
            color = f"C{i % 10}"
            # Plot history values
            ax.plot(
                history_time_idx,
                history_values[:history_len, channel_idx],
                label=f"Channel {channel_idx}",
                color=color,
            )

            # Plot target values if requested
            if show_target:
                ax.plot(
                    target_time_idx,
                    target_values[:target_len, channel_idx],
                    linestyle="--",
                    color=color,
                )

    # Add vertical line separating history and target
    if show_target:
        ax.axvline(x=history_len - 0.5, color="red", linestyle=":", alpha=0.7)

        # Add shaded region for target
        ax.axvspan(
            history_len - 0.5,
            history_len + target_len,
            alpha=0.1,
            color="gray",
            label="Target Region",
        )

    # Set title and labels
    ax.set_title(f"{title} (Task: {ts_data.task[sample_idx].item()})")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend(loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path is provided
    if output_file is not None:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    # Show plot if requested
    if show:
        plt.show()

    return fig


# Example usage demonstration
if __name__ == "__main__":
    from src.synthetic_generation.lmc_synth import TimeSeriesGenerator

    # Create a generator and generate a batch of data
    generator = TimeSeriesGenerator(
        batch_size=2,
        history_len=200,
        target_len=50,
        num_channels=20,
        num_ts_features=4,
        variable_lengths=True,
        seed=42,
    )

    # Generate a batch of data
    ts_data = generator.generate_batch()

    # Plot the first sample with default settings
    fig1 = plot_synthetic_mts(
        ts_data=ts_data,
        sample_idx=0,
        title="Example Synthetic MTS - Individual Channels",
        output_file="outputs/plots/synthetic_mts_individual.png",
    )

    # Plot selected channels on a single plot
    fig2 = plot_multiple_channels(
        ts_data=ts_data,
        sample_idx=0,
        channel_indices=[0, 2, 5, 10, 15],
        title="Example Synthetic MTS - Channel Overlay",
        output_file="outputs/plots/synthetic_mts_overlay.png",
    )
