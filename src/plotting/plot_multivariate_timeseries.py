from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    history_time_features: Optional[np.ndarray] = None,
    target_values: Optional[np.ndarray] = None,
    target_time_features: Optional[np.ndarray] = None,
    target_channels_indices: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
    channel_indices: Optional[List[int]] = None,
    max_channels: int = 5,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multivariate Time Series",
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    timestamp_index: int = 0,
) -> Figure:
    """
    Plot multiple channels of a multivariate time series with history and optional targets/predictions.

    Parameters
    ----------
    history_values : np.ndarray
        Historical values with shape [seq_len, num_channels]
    history_time_features : np.ndarray, optional
        Time features for history values with shape [seq_len, num_time_features]
        The column specified by timestamp_index is used as timestamp.
    target_values : np.ndarray, optional
        Target values with shape [pred_len, num_targets]
    target_time_features : np.ndarray, optional
        Time features for target values with shape [pred_len, num_time_features]
        The column specified by timestamp_index is used as timestamp.
    target_channels_indices : np.ndarray, optional
        Indices mapping target columns to history channels with shape [num_targets]
    predicted_values : np.ndarray, optional
        Model's predicted values with shape [pred_len, num_targets]
    channel_indices : List[int], optional
        Indices of channels to plot. If None, plots up to max_channels.
    max_channels : int
        Maximum number of channels to plot if channel_indices is None.
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
    timestamp_index : int
        Index of the column in *_time_features to use as the timestamp (default: 0).

    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    # Validate and prepare inputs
    history_values = np.asarray(history_values)
    num_channels = history_values.shape[1]

    # Set up channel indices to plot
    if channel_indices is None:
        channel_indices = list(range(min(num_channels, max_channels)))
    else:
        if any(idx < 0 or idx >= num_channels for idx in channel_indices):
            raise ValueError(
                f"channel_indices must be between 0 and {num_channels - 1}"
            )

    # Set up time values
    if history_time_features is not None:
        history_time = history_time_features[:, timestamp_index]
    else:
        history_time = np.arange(history_values.shape[0])

    # Set up target time if available
    target_time = None
    if target_values is not None and target_time_features is not None:
        target_time = target_time_features[:, timestamp_index]
    elif target_values is not None:
        target_time = np.arange(
            history_values.shape[0],
            history_values.shape[0] + target_values.shape[0],
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each selected channel's history
    for i, channel_idx in enumerate(channel_indices):
        label = f"Channel {channel_idx}"
        color = f"C{i % 10}"  # Use matplotlib's default color cycle

        # Plot history
        ax.plot(
            history_time,
            history_values[:, channel_idx],
            label=label,
            color=color,
            linestyle="-",  # Always solid line for history
        )

        # Plot target if available and this channel is a target
        if (
            target_values is not None
            and target_channels_indices is not None
            and target_time is not None
            and channel_idx in target_channels_indices
        ):
            # Find which target index corresponds to this channel
            target_idx = np.where(target_channels_indices == channel_idx)[0][0]

            # Plot actual target with solid line
            ax.plot(
                target_time,
                target_values[:, target_idx],
                color=color,
                linestyle="-",  # Solid line for actual values
                label=f"{label} (Target)",
            )

            # Plot predictions if available
            if predicted_values is not None:
                ax.plot(
                    target_time,
                    predicted_values[:, target_idx],
                    color=color,
                    linestyle="--",  # Dashed line for predictions
                    label=f"{label} (Predicted)",
                )

    # Add history/target separator
    if target_time is not None:
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
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
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

    history_time_features = None
    if ts_data.history_time_features is not None:
        history_time_features = (
            ts_data.history_time_features[sample_idx].detach().cpu().numpy()
        )

    target_values = None
    if ts_data.target_values is not None:
        target_values = ts_data.target_values[sample_idx].detach().cpu().numpy()

    target_time_features = None
    if ts_data.target_time_features is not None:
        target_time_features = (
            ts_data.target_time_features[sample_idx].detach().cpu().numpy()
        )

    target_channels_indices = None
    if ts_data.target_channels_indices is not None:
        target_channels_indices = (
            ts_data.target_channels_indices[sample_idx].detach().cpu().numpy()
        )

    pred_values = None
    if predicted_values is not None:
        import torch

        if isinstance(predicted_values, torch.Tensor):
            pred_values = predicted_values[sample_idx].detach().cpu().numpy()
        else:
            # If predicted_values is a numpy array or scalar, handle accordingly
            # If it's a batch, index it; if not, use as is
            try:
                pred_values = predicted_values[sample_idx]
            except Exception:
                pred_values = predicted_values

    return plot_multivariate_timeseries(
        history_values=history_values,
        history_time_features=history_time_features,
        target_values=target_values,
        target_time_features=target_time_features,
        target_channels_indices=target_channels_indices,
        predicted_values=pred_values,
        **kwargs,
    )
