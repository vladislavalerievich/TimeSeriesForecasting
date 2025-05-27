from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    target_values: Optional[np.ndarray] = None,
    target_index: Optional[int] = None,
    predicted_values: Optional[np.ndarray] = None,
    max_channels: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multivariate Time Series",
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
) -> Figure:
    """
    Plot a single channel of a multivariate time series with history and optional targets/predictions.
    Only the channel specified by target_index is highlighted, but up to max_channels channels are shown in the background.

    Parameters
    ----------
    history_values : np.ndarray
        Historical values with shape [seq_len, num_channels]
    target_values : np.ndarray, optional
        Target values with shape [pred_len, num_targets]
    target_index : int, optional
        Index of the target channel to plot
    predicted_values : np.ndarray, optional
        Model's predicted values with shape [pred_len, num_targets]
    max_channels : int
        Maximum number of channels to plot, if the number of channels is greater than max_channels,
        the plot will be truncated, but the target channel will still be plotted.
    figsize : Tuple[int, int]
        Figure size in inches.
    title : str
        Title for the plot.
    output_file : str, optional
        Path to save the plot, if provided.
    dpi : int
        DPI for saved figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    # Validate and prepare inputs
    history_values = np.asarray(history_values)
    num_channels = history_values.shape[1]
    if target_index is None:
        raise ValueError(
            "target_index must be provided to specify which channel to plot"
        )
    if isinstance(target_index, np.ndarray):
        target_index = int(target_index.item())

    # Set up time values
    history_len = history_values.shape[0]
    history_time = np.arange(history_len)

    # Set up target time if available
    target_time = None
    if target_values is not None:
        target_values = np.asarray(target_values)
        target_len = target_values.shape[0]
        target_time = np.arange(history_len, history_len + target_len)

    # Handle predicted_values dimensions
    if predicted_values is not None:
        predicted_values = np.asarray(predicted_values)
        if predicted_values.ndim == 1:
            predicted_values = predicted_values[:, None]
        elif predicted_values.ndim == 0:
            predicted_values = predicted_values[None, None]
        if predicted_values.shape[0] != (
            target_values.shape[0]
            if target_values is not None
            else predicted_values.shape[0]
        ):
            predicted_values = predicted_values.T

    # Determine which channels to plot
    selected_channels = list(range(min(num_channels, max_channels)))
    if target_index not in selected_channels:
        selected_channels.append(target_index)
    selected_channels = sorted(set(selected_channels))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot background channels (not the target channel)
    for ch in selected_channels:
        if ch == target_index:
            continue
        y_hist = history_values[:, ch]
        ax.plot(
            history_time,
            y_hist.reshape(-1),
            color=f"C{ch % 10}",
            linestyle="-",
            alpha=0.3,
            label=f"Ch{ch}" if ch < max_channels else f"Ch{ch} (extra)",
        )
        # Optionally plot targets for background channels if available
        if (
            target_values is not None
            and target_values.ndim == 2
            and ch < target_values.shape[1]
        ):
            y_target = target_values[:, ch]
            ax.plot(
                target_time,
                y_target.reshape(-1),
                color=f"C{ch % 10}",
                linestyle="-",
                alpha=0.3,
            )

    # Plot the target channel (highlighted)
    y_hist = history_values[:, target_index]
    ax.plot(
        history_time,
        y_hist.reshape(-1),
        color=f"C{target_index % 10}",
        linestyle="-",
        linewidth=2.0,
        label=f"Ch{target_index}: True",
    )
    if target_values is not None:
        y_target = target_values[:, 0] if target_values.ndim == 2 else target_values
        ax.plot(
            target_time,
            y_target.reshape(-1),
            color=f"C{target_index % 10}",
            linestyle="-",
            linewidth=2.0,
            label="Target",
        )
    if predicted_values is not None:
        y_pred = (
            predicted_values[:, 0] if predicted_values.ndim == 2 else predicted_values
        )
        ax.plot(
            target_time,
            y_pred.reshape(-1),
            color="red",
            linestyle="--",
            linewidth=2.0,
            label="Prediction",
        )

    # Add history/target separator
    if target_time is not None:
        ax.axvline(
            x=history_len,
            color="red",
            linestyle=":",
            alpha=0.7,
            label="History/Target Split",
        )
        ax.axvspan(
            history_len,
            target_time[-1],
            alpha=0.1,
            color="gray",
            label="Target Region",
        )

    # Set title and labels
    ax.set_title(f"{title} (Total channels: {num_channels})")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")

    # Show plot if requested
    if show:
        plt.show()

    return fig


def plot_from_container(
    ts_data,
    sample_idx: int = 0,
    predicted_values: Optional[np.ndarray] = None,
    **kwargs,
) -> Figure:
    """
    Helper function to plot from a TimeSeriesDataContainer.

    Parameters
    ----------
    ts_data : TimeSeriesDataContainer
        The time series data container.
    sample_idx : int
        Index of the sample in the batch to plot.
    predicted_values : np.ndarray, optional
        Model's predicted values with shape [batch_size, pred_len, num_targets]
        If provided, will extract the predictions for sample_idx.
    **kwargs :
        Additional arguments to pass to plot_multivariate_timeseries.

    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    # Extract data for the specified sample
    history_values = ts_data.history_values[sample_idx].detach().cpu().numpy()
    target_values = ts_data.target_values[sample_idx].detach().cpu().numpy()
    target_index = ts_data.target_index[sample_idx].detach().cpu().numpy()
    if np.ndim(target_index) > 0:
        target_index = int(target_index.item())
    if predicted_values is not None:
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu().numpy()
        if predicted_values.ndim == 3:
            predicted_values = predicted_values[sample_idx]
    return plot_multivariate_timeseries(
        history_values=history_values,
        target_values=target_values,
        target_index=target_index,
        predicted_values=predicted_values,
        **kwargs,
    )
