import os

import matplotlib.pyplot as plt

from src.data_handling.time_series_data_structure import TimeSeriesData


def plot_synthetic_function(
    data=None,
    history=None,
    true_future=None,
    pred_future=None,
    title="Synthetic Function",
    output_file=None,
):
    """
    Visualizes a synthetic time series (e.g., sine or step function) and optionally saves the plot.

    Args:
        data (TimeSeriesData, optional): A TimeSeriesData instance containing history and target values.
        history (np.ndarray, optional): Historical data (1D array). Used if data is None.
        true_future (np.ndarray, optional): True future values (1D array). Used if data is None.
        pred_future (np.ndarray, optional): Predicted future values (1D array). If None, not plotted.
        title (str): Title of the plot.
        output_file (str, optional): File path to save the plot. If None, plot is not saved.

    Returns:
        matplotlib.figure.Figure: The generated figure object for further use.
    """
    # Create figure
    fig = plt.figure(figsize=(10, 5))

    # Extract history and true_future from TimeSeriesData if provided, otherwise use raw inputs
    if data is not None:
        if not isinstance(data, TimeSeriesData):
            raise ValueError("data must be an instance of TimeSeriesData")
        # Take the first sample (batch_size=1) and squeeze feature dimension
        history = data.history_values[0].cpu().numpy().squeeze(-1)  # [seq_len]
        true_future = data.target_values[0].cpu().numpy().squeeze(-1)  # [pred_len]
    elif history is None or true_future is None:
        raise ValueError(
            "Either 'data' or both 'history' and 'true_future' must be provided"
        )

    # Plot history
    plt.plot(range(len(history)), history, label="History", color="blue")

    # Plot true future
    future_start = len(history)
    plt.plot(
        range(future_start, future_start + len(true_future)),
        true_future,
        label="True Future",
        color="green",
    )

    # Plot predicted future if provided
    if pred_future is not None:
        plt.plot(
            range(future_start, future_start + len(pred_future)),
            pred_future,
            label="Predicted Future",
            color="red",
        )

    # Add title and legend
    plt.title(title)
    plt.legend()

    # Save to file only if output_file is provided
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight")

    # Return figure
    return fig


if __name__ == "__main__":
    # from src.synthetic_generation.step import generate_step_batch

    # seq_lens = [128, 256, 512, 640]

    # for sl in seq_lens:
    #     step_batch = generate_step_batch(batch_size=1, seq_len=sl, pred_len=64)
    #     fig = plot_synthetic_function(
    #         data=step_batch,
    #         title=f"Step Function (seq_len={sl})",
    #         output_file=f"outputs/plots/debug_step_seq_len_{sl}.png",
    #     )
    #     plt.close(fig)

    # print("Debug plots saved.")

    from src.synthetic_generation.sine_wave import generate_sine_batch

    seq_lens = [128, 256, 512, 640]

    for sl in seq_lens:
        step_batch = generate_sine_batch(batch_size=1, seq_len=sl, pred_len=64)
        fig = plot_synthetic_function(
            data=step_batch,
            title=f"Step Function (seq_len={sl})",
            output_file=f"outputs/plots/debug_sine_seq_len_{sl}.png",
        )
        plt.close(fig)

    print("Debug plots saved.")
