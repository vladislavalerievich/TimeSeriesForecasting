from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    future_values: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
    max_channels: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multivariate Time Series",
    output_file: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
) -> Figure:
    """
    Plot multivariate time series with history and optional future values and predictions.
    All channels are plotted up to max_channels.

    Parameters
    ----------
    history_values : np.ndarray
        Historical values with shape [seq_len, num_channels].
    future_values : np.ndarray, optional
        Future values with shape [pred_len, num_channels].
    predicted_values : np.ndarray, optional
        Model's predicted values with shape [pred_len, num_channels].
    max_channels : int
        Maximum number of channels to plot.
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
    if history_values.ndim == 1:
        history_values = history_values.reshape(-1, 1)
    num_channels = history_values.shape[1]

    # Set up time values
    history_len = history_values.shape[0]
    history_time = np.arange(history_len)

    # Set up future time if available
    future_time = None
    if future_values is not None:
        future_values = np.asarray(future_values)
        if future_values.ndim == 1:
            future_values = future_values.reshape(-1, 1)
        future_len = future_values.shape[0]
        future_time = np.arange(history_len, history_len + future_len)

    if predicted_values is not None:
        predicted_values = np.asarray(predicted_values)
        if predicted_values.ndim == 1:
            predicted_values = predicted_values.reshape(-1, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each channel
    channels_to_plot = min(num_channels, max_channels)
    for ch in range(channels_to_plot):
        # Plot history
        ax.plot(
            history_time,
            history_values[:, ch],
            color=f"C{ch % 10}",
            linestyle="-",
            label=f"Ch{ch} History" if ch < 5 else None,  # Avoid too many labels
        )

        # Plot future values with continuity
        if future_values is not None and ch < future_values.shape[1]:
            # Create continuous time and values arrays to ensure visual continuity
            # Include the last history point to connect the lines
            continuous_time = np.concatenate([[history_len - 1], future_time])
            continuous_values = np.concatenate(
                [[history_values[-1, ch]], future_values[:, ch]]
            )

            ax.plot(
                continuous_time,
                continuous_values,
                color=f"C{ch % 10}",
                linestyle="--",
                label=f"Ch{ch} Future" if ch < 5 else None,
            )

        # Plot predicted values with continuity
        if predicted_values is not None and ch < predicted_values.shape[1]:
            # Create continuous time and values arrays to ensure visual continuity
            continuous_time = np.concatenate([[history_len - 1], future_time])
            continuous_values = np.concatenate(
                [[history_values[-1, ch]], predicted_values[:, ch]]
            )

            ax.plot(
                continuous_time,
                continuous_values,
                color=f"C{ch % 10}",
                linestyle=":",
                label=f"Ch{ch} Prediction" if ch < 5 else None,
            )

    # Add history/future separator
    if future_time is not None:
        ax.axvline(
            x=history_len
            - 0.5,  # Adjust to be between the last history and first future point
            color="red",
            linestyle=":",
            alpha=0.7,
            label="History/Future Split",
        )
        ax.axvspan(
            history_len - 0.5,
            future_time[-1],
            alpha=0.1,
            color="gray",
        )

    # Set title and labels
    ax.set_title(f"{title} (Showing {channels_to_plot}/{num_channels} channels)")
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
    Helper function to plot from a BatchTimeSeriesContainer.

    Parameters
    ----------
    ts_data : BatchTimeSeriesContainer
        The batch time series data container.
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
    future_values = ts_data.future_values[sample_idx].detach().cpu().numpy()
    if predicted_values is not None:
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu().numpy()
        if predicted_values.ndim == 3:
            predicted_values = predicted_values[sample_idx]
    return plot_multivariate_timeseries(
        history_values=history_values,
        future_values=future_values,
        predicted_values=predicted_values,
        **kwargs,
    )
