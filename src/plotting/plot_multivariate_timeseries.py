from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    target_values: Optional[np.ndarray] = None,
    target_channels_indices: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multivariate Time Series",
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
) -> Figure:
    """
    Plot multiple channels of a multivariate time series with history and optional targets/predictions.
    Only plots the series specified in target_channels_indices, supporting any number of target channels.
    Predicted values are plotted in varying shades of red to distinguish from target values.

    Parameters
    ----------
    history_values : np.ndarray
        Historical values with shape [seq_len, num_channels]
    target_values : np.ndarray, optional
        Target values with shape [pred_len, num_targets]
    target_channels_indices : np.ndarray, optional
        Indices mapping target columns to history channels with shape [num_targets]
    predicted_values : np.ndarray, optional
        Model's predicted values with shape [pred_len, num_targets]
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

    # If target_channels_indices not provided, we can't selectively plot
    if target_channels_indices is None:
        raise ValueError(
            "target_channels_indices must be provided to specify which channels to plot"
        )

    # Convert target_channels_indices to numpy array if it's not already
    target_channels_indices = np.asarray(target_channels_indices)

    # Ensure target_channels_indices is 1D
    if target_channels_indices.ndim != 1:
        target_channels_indices = target_channels_indices.flatten()

    # Set up time values - use range instead of actual time features
    history_len = history_values.shape[0]
    history_time = np.arange(history_len)

    # Set up target time if available - also use range
    target_time = None
    if target_values is not None:
        target_len = target_values.shape[0]
        target_time = np.arange(history_len, history_len + target_len)

    # Handle predicted_values dimensions
    if predicted_values is not None:
        predicted_values = np.asarray(predicted_values)
        if predicted_values.ndim == 1:
            predicted_values = predicted_values[:, None]
        elif predicted_values.ndim == 0:
            predicted_values = predicted_values[None, None]
        elif predicted_values.shape[0] != target_values.shape[0]:
            predicted_values = predicted_values.T

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each selected channel
    for i, channel_idx in enumerate(target_channels_indices):
        channel_idx = int(channel_idx)  # Ensure it's an integer
        color = f"C{i % 10}"  # Use matplotlib's default color cycle

        # Extract and check history for this target channel
        y_hist = np.asarray(history_values[:, channel_idx])
        if y_hist.ndim != 1 or y_hist.shape[0] != history_time.shape[0]:
            raise ValueError(
                f"history_values for channel {channel_idx} has shape {y_hist.shape}, expected ({history_time.shape[0]},)"
            )
        # Plot history for this target channel
        ax.plot(
            history_time,
            y_hist.reshape(-1),
            color=color,
            linestyle="-",  # Solid line for history
            label=f"Ch{channel_idx}: True ({color}) | Pred (red shade)"
            if predicted_values is not None
            else f"Ch{channel_idx}: True ({color})",
        )

        # Plot target if available
        if target_values is not None and target_time is not None:
            # Find which target index corresponds to this channel
            target_idx = np.where(target_channels_indices == channel_idx)[0]
            if len(target_idx) == 0:
                continue  # Skip if channel_idx not in target_channels_indices
            target_idx = target_idx[0]
            y_target = np.asarray(target_values[:, target_idx])
            if y_target.ndim != 1 or y_target.shape[0] != target_time.shape[0]:
                raise ValueError(
                    f"target_values for channel {channel_idx} (target idx {target_idx}) has shape {y_target.shape}, expected ({target_time.shape[0]},)"
                )
            # Plot actual target with solid line
            ax.plot(
                target_time,
                y_target.reshape(-1),
                color=color,
                linestyle="-",  # Solid line for actual values
                label=None,  # No separate label for target
            )

            # Plot predictions if available
            if predicted_values is not None:
                y_pred = np.asarray(predicted_values[:, target_idx])
                if y_pred.ndim != 1 or y_pred.shape[0] != target_time.shape[0]:
                    raise ValueError(
                        f"predicted_values for channel {channel_idx} (target idx {target_idx}) has shape {y_pred.shape}, expected ({target_time.shape[0]},)"
                    )
                # Generate shade of red: darker for earlier indices, lighter for later
                red_shade = 1.0 - (i / max(len(target_channels_indices), 1)) * 0.5
                pred_color = (red_shade, 0.2, 0.2)  # RGB: varying red intensity
                ax.plot(
                    target_time,
                    y_pred.reshape(-1),
                    color=pred_color,
                    linestyle="--",  # Dashed line for predictions
                    label=None,  # No separate label for predictions
                )

    # Add history/target separator
    if target_time is not None:
        ax.axvline(
            x=history_len - 0.5,  # Position slightly before the target starts
            color="red",
            linestyle=":",
            alpha=0.7,
            label="History/Target Split",
        )
        ax.axvspan(
            history_len - 0.5,
            target_time[-1],
            alpha=0.1,
            color="gray",
            label="Target Region",
        )

    # Set title and labels
    ax.set_title(title)
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
        plt.savefig("multivariate_timeseries.png")

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
    target_channels_indices = (
        ts_data.target_channels_indices[sample_idx].detach().cpu().numpy()
    )

    if predicted_values is not None and isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.detach().cpu().numpy()

    return plot_multivariate_timeseries(
        history_values=history_values,
        target_values=target_values,
        target_channels_indices=target_channels_indices,
        predicted_values=predicted_values,
        **kwargs,
    )
