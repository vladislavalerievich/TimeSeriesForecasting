import logging
import os

from src.plotting.plot_multivariate_timeseries_sample import (
    plot_multivariate_timeseries_sample,
)
from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_batch_samples(
    batch_size: int = 10,
    history_length: int = 128,
    target_length: int = 64,
    num_channels: int = 5,
    output_dir: str = "outputs/plots",
    global_seed: int = 42,
) -> None:
    """
    Visualize all samples in a batch of synthetic multivariate time series and save plots.

    Parameters
    ----------
    batch_size : int
        Number of samples in the batch.
    history_length : int
        Length of the history window.
    target_length : int
        Length of the target window.
    num_channels : int
        Number of channels in each time series.
    output_dir : str
        Directory to save the plots.
    global_seed : int
        Random seed for reproducibility.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    # Create generator and generate batch
    generator = MultivariateTimeSeriesGenerator(global_seed=global_seed)
    batch = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
    )
    logger.info(f"Generated batch with {batch_size} samples")

    # Validate batch size
    if batch.history_values.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: expected {batch_size}, got {batch.history_values.shape[0]}"
        )

    # Visualize and save each sample
    for sample_idx in range(batch_size):
        output_file = os.path.join(output_dir, f"sample_{sample_idx:03d}.png")
        try:
            plot_multivariate_timeseries_sample(
                ts_data=batch,
                sample_idx=sample_idx,
                output_file=output_file,
                show=False,  # Don't display plots to avoid blocking
                title=f"Multivariate Time Series (Sample {sample_idx})",
            )
            logger.info(f"Saved plot for sample {sample_idx} to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save plot for sample {sample_idx}: {e}")
            raise


if __name__ == "__main__":
    visualize_batch_samples()
