import logging
import os

import torch

from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.synthetic_generation.lmc_generator_wrapper import LMCGeneratorWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_batch_samples(
    batch_size: int = 1,
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
    generator = LMCGeneratorWrapper(global_seed=global_seed)
    batch = generator.generate_batch(
        batch_size=batch_size,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
    )
    logger.info(f"Batch history values shape: {batch.history_values.shape}")
    logger.info(f"Batch target values shape: {batch.target_values.shape}")
    logger.info(f"Batch target indices shape: {batch.target_channels_indices.shape}")
    logger.info(
        f"Batch history time features shape: {batch.history_time_features.shape}"
    )
    logger.info(f"Batch target time features shape: {batch.target_time_features.shape}")
    logger.info(f"Batch history time features values: {batch.history_time_features[0]}")
    # Validate batch size
    if batch.history_values.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: expected {batch_size}, got {batch.history_values.shape[0]}"
        )

    # Visualize and save each sample
    for sample_idx in range(batch_size):
        output_file = os.path.join(output_dir, f"sample_{sample_idx:03d}.png")
        try:
            plot_from_container(
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


def visualize_saved_batch(
    batch_path: str = "outputs/datasets/mts/batch_000.pt",
    output_dir: str = "outputs/plots/",
    sample_idx: int = 0,
    show: bool = False,
) -> None:
    """
    Load a saved batch file and visualize a specific sample.

    Parameters
    ----------
    batch_path : str
        Path to the saved batch file.
    output_dir : str
        Directory to save the plots.
    sample_idx : int
        Index of the sample to visualize.
    show : bool
        Whether to display the plot.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    # Load batch file
    try:
        logger.info(f"Loading batch from {batch_path}")
        batch = torch.load(batch_path)
        logger.info(f"Loaded batch with {batch.history_values.shape[0]} samples")
    except Exception as e:
        logger.error(f"Failed to load batch from {batch_path}: {e}")
        raise

    # Visualize specific sample
    output_file = os.path.join(output_dir, f"saved_sample_{sample_idx:03d}.png")
    try:
        # Visualize the sample
        plot_from_container(
            ts_data=batch,
            sample_idx=sample_idx,
            output_file=output_file,
            show=show,
            title=f"Saved Multivariate Time Series (Sample {sample_idx})",
        )
        logger.info(f"Saved plot for sample {sample_idx} to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save plot for sample {sample_idx}: {e}")
        raise


if __name__ == "__main__":
    # Visualize synthetic samples
    visualize_batch_samples()

    # Load and visualize the first sample from a saved batch
    visualize_saved_batch()
