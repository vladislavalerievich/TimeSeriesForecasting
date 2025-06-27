from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from src.data_handling.data_containers import BatchTimeSeriesContainer


def plot_multivariate_timeseries(
    history_values: np.ndarray,
    future_values: Optional[np.ndarray] = None,
    predicted_values: Optional[np.ndarray] = None,
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
    history_dates = pd.date_range(end=pd.Timestamp.now(), periods=seq_len, freq="D")
    if future_values is not None:
        pred_len = future_values.shape[0]
        future_dates = pd.date_range(
            start=history_dates[-1] + pd.Timedelta(days=1), periods=pred_len, freq="D"
        )
    elif predicted_values is not None:
        pred_len = predicted_values.shape[0]
        future_dates = pd.date_range(
            start=history_dates[-1] + pd.Timedelta(days=1), periods=pred_len, freq="D"
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
    ts_data: BatchTimeSeriesContainer,
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
            Container with the time series data.
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
    history_values = ts_data.history_values[sample_idx].cpu().numpy()
    future_values = ts_data.future_values[sample_idx].cpu().numpy()

    # Extract predictions for the sample if provided
    if predicted_values is not None:
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.detach().cpu().numpy()
        if predicted_values.ndim == 3:
            predicted_values = predicted_values[sample_idx]

    return plot_multivariate_timeseries(
        history_values=history_values,
        future_values=future_values,
        predicted_values=predicted_values,
        title=title,
        output_file=output_file,
        show=show,
    )
