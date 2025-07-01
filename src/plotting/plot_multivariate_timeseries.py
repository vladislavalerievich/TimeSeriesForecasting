from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency


def _frequency_to_pandas_freq(frequency: Union[Frequency, str]) -> str:
    """
    Convert Frequency enum to pandas frequency string.

    Args:
        frequency: Frequency enum or string

    Returns:
        Pandas frequency string
    """
    if isinstance(frequency, str):
        return frequency

    # Map Frequency enum values to pandas frequency strings
    freq_mapping = {
        Frequency.A: "YE",  # Annual (Year End)
        Frequency.Q: "QE",  # Quarterly (Quarter End)
        Frequency.M: "ME",  # Monthly (Month End)
        Frequency.W: "W",  # Weekly
        Frequency.D: "D",  # Daily
        Frequency.H: "h",  # Hourly
        Frequency.S: "s",  # Seconds
        Frequency.T1: "1min",  # 1 minute
        Frequency.T5: "5min",  # 5 minutes
        Frequency.T10: "10min",  # 10 minutes
        Frequency.T15: "15min",  # 15 minutes
    }

    return freq_mapping.get(frequency, "D")  # Default to daily if not found


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    future_values: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
    start: Optional[Union[np.datetime64, pd.Timestamp]] = None,
    frequency: Optional[Union[Frequency, str]] = None,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Plots a multivariate time series with history, future, and predicted values.

    Args:
        history_values : np.ndarray
            Historical observations with shape [seq_len, num_channels]
        future_values : np.ndarray, optional
            Ground truth future observations with shape [pred_len, num_channels]
        predicted_values : np.ndarray, optional
            Model's predicted values with shape [pred_len, num_channels]
        start : np.datetime64 or pd.Timestamp, optional
            Start timestamp for the time series. If None, uses dummy dates.
        frequency : Frequency or str, optional
            Frequency of the time series. If None, defaults to daily.
        title : str, optional
            Title for the plot.
        output_file : str, optional
            Path to save the plot, if provided.
        show : bool
            Whether to display the plot.

    Returns
        matplotlib.figure.Figure: The plot figure.
    """
    num_channels = history_values.shape[1]
    seq_len = history_values.shape[0]

    # Use a color-blind friendly palette
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    fig, axes = plt.subplots(
        num_channels, 1, figsize=(15, 3 * num_channels), sharex=True
    )
    if num_channels == 1:
        axes = [axes]

    # Create date range for plotting
    if start is not None and frequency is not None:
        # Convert start to pd.Timestamp if it's np.datetime64
        if isinstance(start, np.datetime64):
            start_timestamp = pd.Timestamp(start)
        else:
            start_timestamp = start

        # Convert frequency to pandas frequency string
        pandas_freq = _frequency_to_pandas_freq(frequency)

        # Create proper date range starting from the given start timestamp
        history_dates = pd.date_range(
            start=start_timestamp, periods=seq_len, freq=pandas_freq
        )

        if future_values is not None:
            pred_len = future_values.shape[0]
            # Calculate the next timestamp after history using the frequency offset
            next_timestamp = history_dates[-1] + pd.tseries.frequencies.to_offset(
                pandas_freq
            )
            future_dates = pd.date_range(
                start=next_timestamp,
                periods=pred_len,
                freq=pandas_freq,
            )
        elif predicted_values is not None:
            pred_len = predicted_values.shape[0]
            # Calculate the next timestamp after history using the frequency offset
            next_timestamp = history_dates[-1] + pd.tseries.frequencies.to_offset(
                pandas_freq
            )
            future_dates = pd.date_range(
                start=next_timestamp,
                periods=pred_len,
                freq=pandas_freq,
            )
    else:
        # Fallback to dummy dates if start or frequency not provided
        history_dates = pd.date_range(end=pd.Timestamp.now(), periods=seq_len, freq="D")
        if future_values is not None:
            pred_len = future_values.shape[0]
            future_dates = pd.date_range(
                start=history_dates[-1] + pd.Timedelta(days=1),
                periods=pred_len,
                freq="D",
            )
        elif predicted_values is not None:
            pred_len = predicted_values.shape[0]
            future_dates = pd.date_range(
                start=history_dates[-1] + pd.Timedelta(days=1),
                periods=pred_len,
                freq="D",
            )

    for i, ax in enumerate(axes):
        # Plot history
        ax.plot(
            history_dates,
            history_values[:, i],
            label="History",
            color=colors[0],
        )

        # Plot future values if provided
        if future_values is not None:
            ax.plot(
                future_dates,
                future_values[:, i],
                label="Future (Ground Truth)",
                color=colors[2],
            )

        # Plot predicted values if provided
        if predicted_values is not None:
            pred_len = predicted_values.shape[0]
            current_future_dates = future_dates[:pred_len]
            ax.plot(
                current_future_dates,
                predicted_values[:, i],
                label="Predicted",
                color=colors[1],
                linestyle="--",
            )

        # Formatting
        ax.set_title(f"Channel {i + 1}")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else None)

    if output_file:
        plt.savefig(output_file, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_from_container(
    batch: BatchTimeSeriesContainer,
    sample_idx: int,
    predicted_values: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    output_file: Optional[str] = None,
    show: bool = True,
) -> Figure:
    """
    Wrapper to plot a single sample from a BatchTimeSeriesContainer.

    Args:
        ts_data : BatchTimeSeriesContainer
            Container with the time series data. The start timestamp and frequency
            from the container will be used to create proper time axis labels.
        sample_idx: int
            The index of the sample to plot from the batch.
        predicted_values : np.ndarray, optional
            Model's predicted values with shape [batch_size, pred_len, num_targets]
            If provided, will extract the predictions for sample_idx.
        title : str, optional
            Title for the plot.
        output_file : str, optional
            Path to save the plot, if provided.
        show : bool
            Whether to display the plot.

    Returns
        matplotlib.figure.Figure: The plot figure.
    """
    # Extract data for the specified sample index
    history_values = batch.history_values[sample_idx].cpu().numpy()
    future_values = batch.future_values[sample_idx].cpu().numpy()

    # Extract start and frequency from the container
    start = batch.start
    frequency = batch.frequency

    # Extract predictions for the sample if provided
    if predicted_values is not None:
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu().numpy()
        if predicted_values.ndim == 3:
            predicted_values = predicted_values[sample_idx]

    # Include generator name in title if available
    if batch.generator_name and title:
        title = f"[{batch.generator_name}] {title}"
    elif batch.generator_name and not title:
        title = f"[{batch.generator_name}] Time Series"

    return plot_multivariate_timeseries(
        history_values=history_values,
        future_values=future_values,
        predicted_values=predicted_values,
        start=start,
        frequency=frequency,
        title=title,
        output_file=output_file,
        show=show,
    )
