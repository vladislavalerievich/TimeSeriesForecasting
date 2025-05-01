from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.data_handling.data_containers import TimeSeriesDataContainer


def plot_multivariate_timeseries_sample(
    ts_data: TimeSeriesDataContainer,
    sample_idx: int = 0,
    channel_indices: Optional[List[int]] = None,
    max_channels: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multivariate Time Series",
    show_target: bool = True,
    use_colors: bool = True,
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    grid: bool = True,
    linestyle_history: str = "-",
    linestyle_target: str = "--",
) -> Figure:
    """
    Plot multiple channels of a multivariate time series from a TimeSeriesDataContainer.

    Parameters
    ----------
    ts_data : TimeSeriesDataContainer
        The time series data to plot.
    sample_idx : int
        Index of the sample in the batch to plot.
    channel_indices : List[int], optional
        Indices of channels to plot. If None, plots up to max_channels.
    max_channels : int
        Maximum number of channels to plot if channel_indices is None.
    figsize : Tuple[int, int]
        Figure size in inches.
    title : str
        Title for the plot.
    show_target : bool
        Whether to plot target values.
    use_colors : bool
        Whether to use different colors for each channel.
    output_file : str, optional
        Path to save the plot, if provided.
    dpi : int
        DPI for saved figure.
    show : bool
        Whether to display the plot.
    grid : bool
        Whether to show a grid.
    linestyle_history : str
        Line style for history values (e.g., "-", "--", ":").
    linestyle_target : str
        Line style for target values.

    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    # Validate inputs
    if not isinstance(ts_data, TimeSeriesDataContainer):
        raise ValueError("ts_data must be a TimeSeriesDataContainer")
    batch_size = ts_data.history_values.shape[0]
    if not 0 <= sample_idx < batch_size:
        raise ValueError(f"sample_idx must be between 0 and {batch_size - 1}")

    # Extract data for the specified sample
    history_values = ts_data.history_values[sample_idx].detach().cpu().numpy()
    history_time = (
        ts_data.history_time_features[sample_idx, :, 0].detach().cpu().numpy()
        if ts_data.history_time_features is not None
        else np.arange(history_values.shape[0])
    )

    num_channels = history_values.shape[1]
    if channel_indices is None:
        channel_indices = list(range(min(num_channels, max_channels)))
    else:
        if any(idx < 0 or idx >= num_channels for idx in channel_indices):
            raise ValueError(
                f"channel_indices must be between 0 and {num_channels - 1}"
            )

    # Handle target data
    target_values = None
    target_time = None
    target_channel_map = None
    if show_target:
        if ts_data.target_values is None or ts_data.target_channels_indices is None:
            raise ValueError("Target values or indices missing when show_target=True")
        target_values = ts_data.target_values[sample_idx].detach().cpu().numpy()
        target_time = (
            ts_data.target_time_features[sample_idx, :, 0].detach().cpu().numpy()
            if ts_data.target_time_features is not None
            else np.arange(
                history_values.shape[0],
                history_values.shape[0] + target_values.shape[0],
            )
        )
        target_channel_indices = (
            ts_data.target_channels_indices[sample_idx].detach().cpu().numpy()
        )
        # Map target indices to history channels
        target_channel_map = {
            i: idx
            for i, idx in enumerate(target_channel_indices)
            if idx in channel_indices
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each selected channel
    colors = (
        plt.cm.tab10(np.linspace(0, 1, len(channel_indices)))
        if use_colors
        else [None] * len(channel_indices)
    )
    for i, channel_idx in enumerate(channel_indices):
        label = f"Channel {channel_idx}"
        color = colors[i] if use_colors else f"C{(i % 10)}"

        # Plot history
        ax.plot(
            history_time,
            history_values[:, channel_idx],
            label=label,
            color=color,
            linestyle=linestyle_history,
        )

        # Plot target if available and channel is a target
        if (
            show_target
            and target_values is not None
            and channel_idx in target_channel_map.values()
        ):
            target_idx = next(
                k for k, v in target_channel_map.items() if v == channel_idx
            )
            ax.plot(
                target_time,
                target_values[:, target_idx],
                color=color,
                linestyle=linestyle_target,
                label=f"{label} (Target)" if not use_colors else None,
            )

    # Add target region separator and shading
    if show_target and target_time is not None:
        ax.axvline(
            x=history_time[-1],
            color="red",
            linestyle=":",
            alpha=0.7,
            label="History/Target Split",
        )
        ax.axvspan(
            history_time[-1],
            target_time[-1],
            alpha=0.1,
            color="gray",
            label="Target Region",
        )

    # Set title and labels
    ax.set_title(f"{title} (Sample {sample_idx})")
    ax.set_xlabel("Normalized Time (Days)")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    if grid:
        ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {output_file}")

    # Show plot if requested
    if show:
        plt.show()

    return fig
