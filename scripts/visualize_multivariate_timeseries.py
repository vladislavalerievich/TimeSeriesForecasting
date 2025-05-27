import logging
import os

import numpy as np
import torch

from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.synthetic_generation.kernel_generator_wrapper import (
    KernelGeneratorParams,
    KernelGeneratorWrapper,
)
from src.synthetic_generation.lmc_generator_wrapper import (
    LMCGeneratorParams,
    LMCGeneratorWrapper,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_batch_samples(
    batch_size: int = 2,
    history_length: int = 128,
    target_length: int = 64,
    num_channels: int = 5,
    output_dir: str = "outputs/plots",
    global_seed: int = 42,
) -> None:
    """
    Visualize the first sample in a batch of synthetic multivariate time series from both LMC and Kernel generators.
    Also plot artificial predictions for demonstration.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    # --- LMC Generator ---
    lmc_params = LMCGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
    )
    lmc_gen = LMCGeneratorWrapper(lmc_params)
    lmc_batch = lmc_gen.generate_batch(batch_size=batch_size)
    logger.info(f"[LMC] Batch history values shape: {lmc_batch.history_values.shape}")
    logger.info(f"[LMC] Batch target values shape: {lmc_batch.target_values.shape}")

    # Artificial predictions: add noise to the true target values
    lmc_pred = lmc_batch.target_values[0].detach().cpu().numpy() + np.random.normal(
        0, 0.2, lmc_batch.target_values[0].shape
    )

    output_file = os.path.join(output_dir, f"lmc_sample_0.png")
    plot_from_container(
        ts_data=lmc_batch,
        sample_idx=0,
        predicted_values=lmc_pred,
        output_file=output_file,
        show=False,
        title="LMC Synthetic Multivariate Time Series (Sample 0)",
    )
    logger.info(f"Saved LMC plot for sample 0 to {output_file}")

    # --- Kernel Generator ---
    kernel_params = KernelGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
    )
    kernel_gen = KernelGeneratorWrapper(kernel_params)
    kernel_batch = kernel_gen.generate_batch(batch_size=batch_size)
    logger.info(
        f"[Kernel] Batch history values shape: {kernel_batch.history_values.shape}"
    )
    logger.info(
        f"[Kernel] Batch target values shape: {kernel_batch.target_values.shape}"
    )

    # Artificial predictions: add noise to the true target values
    kernel_pred = kernel_batch.target_values[
        0
    ].detach().cpu().numpy() + np.random.normal(
        0, 0.2, kernel_batch.target_values[0].shape
    )

    output_file = os.path.join(output_dir, f"kernel_sample_0.png")
    plot_from_container(
        ts_data=kernel_batch,
        sample_idx=0,
        predicted_values=kernel_pred,
        output_file=output_file,
        show=False,
        title="Kernel Synthetic Multivariate Time Series (Sample 0)",
    )
    logger.info(f"Saved Kernel plot for sample 0 to {output_file}")


def visualize_saved_batch(
    batch_path: str = "data/synthetic_val_data_lmc_75_kernel_25_batches_10_batch_size_64/batch_00000.pt",
    output_dir: str = "outputs/plots/",
    sample_idx: int = 0,
    show: bool = False,
) -> None:
    """
    Load a saved batch file and visualize a specific sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")

    try:
        logger.info(f"Loading batch from {batch_path}")
        batch = torch.load(batch_path)
        logger.info(f"Loaded batch with {batch.history_values.shape[0]} samples")
    except Exception as e:
        logger.error(f"Failed to load batch from {batch_path}: {e}")
        raise

    output_file = os.path.join(output_dir, f"saved_sample_{sample_idx:03d}.png")
    try:
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
    # Visualize synthetic samples from both generators
    visualize_batch_samples()

    # Optionally, load and visualize the first sample from a saved batch
    visualize_saved_batch()
