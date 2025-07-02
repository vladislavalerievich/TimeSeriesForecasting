from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from src.data_handling.data_containers import BatchTimeSeriesContainer, Frequency


def _frequency_to_pandas_freq(frequency: Union[Frequency, str]) -> str:
    """Converts Frequency enum or string to a pandas frequency string."""
    if isinstance(frequency, str):
        return frequency
    freq_mapping = {
        Frequency.A: "YE", Frequency.Q: "QE", Frequency.M: "ME", Frequency.W: "W",
        Frequency.D: "D", Frequency.H: "h", Frequency.S: "s", Frequency.T1: "min",
        Frequency.T5: "5min", Frequency.T10: "10min", Frequency.T15: "15min",
    }
    return freq_mapping.get(frequency, "D")


def plot_multivariate_timeseries(
        history_values: np.ndarray,
        future_values: Optional[np.ndarray] = None,
        predicted_values: Optional[np.ndarray] = None,
        start: Optional[Union[np.datetime64, pd.Timestamp]] = None,
        frequency: Optional[Union[Frequency, str]] = None,
        title: Optional[str] = None,
        output_file: Optional[str] = None,
        show: bool = True,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
) -> Figure:
    """
    Plots a multivariate time series with history, future, predictions, and uncertainty bands.
    """
    num_channels = history_values.shape[1]
    seq_len = history_values.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    # Create date range for plotting
    pred_len = predicted_values.shape[0] if predicted_values is not None else (
        future_values.shape[0] if future_values is not None else 0)
    if start is not None and frequency is not None:
        start_timestamp = pd.Timestamp(start)
        pandas_freq = _frequency_to_pandas_freq(frequency)
        history_dates = pd.date_range(start=start_timestamp, periods=seq_len, freq=pandas_freq)
        if pred_len > 0:
            next_timestamp = history_dates[-1] + pd.tseries.frequencies.to_offset(pandas_freq)
            future_dates = pd.date_range(start=next_timestamp, periods=pred_len, freq=pandas_freq)
    else:
        history_dates = pd.date_range(end=pd.Timestamp.now(), periods=seq_len, freq="D")
        if pred_len > 0:
            future_dates = pd.date_range(start=history_dates[-1] + pd.Timedelta(days=1), periods=pred_len, freq="D")

    for i, ax in enumerate(axes):
        # Plot history and future ground truth
        ax.plot(history_dates, history_values[:, i], color="black", label="History")
        if future_values is not None:
            ax.plot(future_dates, future_values[:, i], color="blue", label="Ground Truth")

        # Plot median prediction line
        if predicted_values is not None:
            ax.plot(future_dates, predicted_values[:, i], color="orange", linestyle="--", label="Prediction (Median)")

        # Plot uncertainty band
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(future_dates, lower_bound[:, i], upper_bound[:, i], color="orange", alpha=0.2,
                            label="Uncertainty Band")

        ax.set_title(f"Channel {i + 1}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Create a single, clean legend for the whole plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

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
        predicted_values: np.ndarray,
        model_quantiles: Optional[List[float]] = None,
        title: Optional[str] = None,
        output_file: Optional[str] = None,
        show: bool = True,
) -> Figure:
    """
    Wrapper to plot a single sample from a BatchTimeSeriesContainer, correctly handling quantiles.
    """
    history_values = batch.history_values[sample_idx].cpu().numpy()
    future_values = batch.future_values[sample_idx].cpu().numpy()

    # Handle scalar case for start/frequency from batches of size 1
    start_ts = batch.start if not hasattr(batch.start, '__len__') else batch.start[sample_idx]
    frequency = batch.frequency if not hasattr(batch.frequency, '__len__') else batch.frequency[sample_idx]

    # --- NEW LOGIC: Process predictions within the plotting utility ---
    median_preds = None
    lower_bound = None
    upper_bound = None

    # Get predictions for the specific sample
    sample_preds = predicted_values[sample_idx]

    if model_quantiles:
        # We are in quantile mode, process the quantiles
        try:
            median_idx = model_quantiles.index(0.5)
            lower_idx = model_quantiles.index(0.1)
            upper_idx = model_quantiles.index(0.9)

            median_preds = sample_preds[..., median_idx]
            lower_bound = sample_preds[..., lower_idx]
            upper_bound = sample_preds[..., upper_idx]
        except (ValueError, IndexError):
            logger.warning("Could not find 0.1, 0.5, 0.9 quantiles for plotting. Plotting median only.")
            median_preds = sample_preds[..., sample_preds.shape[-1] // 2]
    else:
        # We are in Huber/point-prediction mode
        median_preds = sample_preds

    if batch.generator_name and title:
        title = f"[{batch.generator_name}] {title}"
    elif batch.generator_name and not title:
        title = f"[{batch.generator_name}] Time Series"

    return plot_multivariate_timeseries(
        history_values=history_values,
        future_values=future_values,
        predicted_values=median_preds,
        start=start_ts,
        frequency=frequency,
        title=title,
        output_file=output_file,
        show=show,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )