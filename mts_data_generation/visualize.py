import matplotlib.pyplot as plt
import seaborn as sns


def visualize_time_series(x, y, num_sequences=3):
    """
    Plots multivariate time-series data from x (features) and y (targets).

    Args:
        x (Tensor): Shape (batch_size, 1, time_steps, num_features)
        y (Tensor): Shape (batch_size, 1, time_steps, num_outputs)
        num_sequences (int): Number of sequences to visualize from the batch.
    """
    batch_size, _, time_steps, num_features = x.shape
    _, _, _, num_outputs = y.shape

    # Convert tensors to NumPy arrays
    x_np = x.squeeze(1).detach().cpu().numpy()  # Shape: (batch_size, time_steps, num_features)
    y_np = y.squeeze(1).detach().cpu().numpy()  # Shape: (batch_size, time_steps, num_outputs)

    num_sequences = min(num_sequences, batch_size)  # Ensure we don't exceed batch size

    sns.set(style="darkgrid")

    fig, axes = plt.subplots(num_sequences, 2, figsize=(12, 5 * num_sequences), sharex=True)

    # Ensure axes is always a list (even if num_sequences = 1)
    if num_sequences == 1:
        axes = [axes]

    for i in range(num_sequences):
        ax_x, ax_y = axes[i]

        # Plot x (features)
        for f in range(num_features):
            ax_x.plot(range(time_steps), x_np[i, :, f], label=f"Feature {f + 1}", linestyle="--")

        ax_x.set_title(f"Input Features (Sequence {i + 1})")
        ax_x.legend()
        ax_x.set_ylabel("Feature Value")

        # Plot y (targets)
        for o in range(num_outputs):
            ax_y.plot(range(time_steps), y_np[i, :, o], label=f"Target {o + 1}", marker="o", linewidth=2)

        ax_y.set_title(f"Target Outputs (Sequence {i + 1})")
        ax_y.legend()
        ax_y.set_ylabel("Target Value")

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()
