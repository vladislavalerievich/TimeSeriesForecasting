import os

import matplotlib.pyplot as plt

from src.synthetic_generation.sine_wave import generate_sine_batch
from src.synthetic_generation.step import generate_step_batch


def plot_synthetic_function(
    history, true_future, pred_future=None, title="Synthetic Function", output_file=None
):
    """
    Visualizes a synthetic time series (e.g., sine or step function) and optionally saves the plot.

    Args:
        history (np.ndarray): Historical data (1D array).
        true_future (np.ndarray): True future values (1D array).
        pred_future (np.ndarray, optional): Predicted future values (1D array). If None, not plotted.
        title (str): Title of the plot.
        output_file (str, optional): File path to save the plot. If None, plot is not saved.

    Returns:
        matplotlib.figure.Figure: The generated figure object for further use.
    """
    # Create figure
    fig = plt.figure(figsize=(10, 5))

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

    # Return figure (caller can close it or use it directly)
    return fig


if __name__ == "__main__":
    config = {"context_len": 512, "pred_len": 64}
    seq_lens = [128, 256, 512, 640]  # Test different lengths

    for sl in seq_lens:
        step_batch = generate_step_batch(batch_size=1, seq_len=sl, pred_len=128)
        fig = plot_synthetic_function(
            history=step_batch["history"][0].numpy(),
            true_future=step_batch["target_values"][0].numpy(),
            title=f"Step Function (seq_len={sl})",
            output_file=f"outputs/plots/debug_step_seq_len_{sl}.png",
        )
        plt.close(fig)  # Clean up

    print("Debug plots saved.")
