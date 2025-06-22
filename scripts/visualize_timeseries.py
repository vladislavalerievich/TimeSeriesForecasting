import logging
import os

import numpy as np
import torch

from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
    ForecastPFNGeneratorWrapper,
)
from src.synthetic_generation.generator_params import (
    ForecastPFNGeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    LMCGeneratorParams,
)
from src.synthetic_generation.gp_prior.gp_generator_wrapper import (
    GPGeneratorWrapper,
)
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorWrapper,
)
from src.synthetic_generation.lmc_synth.lmc_generator_wrapper import (
    LMCGeneratorWrapper,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_batch_sample(
    generator,
    batch_size: int = 2,
    output_dir: str = "outputs/plots",
    sample_idx: int = 0,
    add_artificial_predictions: bool = False,
) -> None:
    """
    Visualize a sample from a batch of synthetic multivariate time series from any generator.
    Also plot artificial predictions for demonstration if requested.

    Args:
        generator: Any generator wrapper (LMC, Kernel, GP, etc.)
        batch_size: Number of samples to generate in the batch
        output_dir: Directory to save plots
        sample_idx: Index of the sample to visualize
        add_artificial_predictions: Whether to add artificial predictions to the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    generator_name = generator.__class__.__name__
    logger.info(f"[{generator_name}] Generating batch of size {batch_size}")

    # Generate batch
    batch = generator.generate_batch(batch_size=batch_size)
    logger.info(
        f"[{generator_name}] Batch history values shape: {batch.history_values.shape}"
    )
    logger.info(
        f"[{generator_name}] Batch future values shape: {batch.future_values.shape}"
    )

    # Create artificial predictions if requested
    predicted_values = None
    if add_artificial_predictions:
        predicted_values = batch.future_values[
            sample_idx
        ].detach().cpu().numpy() + np.random.normal(
            0, 0.2, batch.future_values[sample_idx].shape
        )

    # Generate output filename based on generator name and parameters
    num_channels = batch.history_values.shape[-1]
    series_type = "multivariate" if num_channels > 1 else "univariate"
    filename = f"{generator_name.lower().replace('generatorwrapper', '')}_{series_type}_sample_{sample_idx}.png"
    output_file = os.path.join(output_dir, filename)

    # Create title
    title = f"{generator_name.replace('GeneratorWrapper', '')} Synthetic {series_type.title()} Time Series (Sample {sample_idx})"

    plot_from_container(
        ts_data=batch,
        sample_idx=sample_idx,
        predicted_values=predicted_values,
        output_file=output_file,
        show=False,
        title=title,
    )
    logger.info(
        f"[{generator_name}] Saved plot for sample {sample_idx} to {output_file}"
    )
    logger.info("--------------------------------")


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
    # Configuration
    batch_size = 2
    history_length = 128
    future_length = 64
    output_dir = "outputs/plots"
    global_seed = 2025

    logger.info(f"Saving plots to {output_dir}")

    # Create generators
    lmc_params = LMCGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=5,
    )
    lmc_gen = LMCGeneratorWrapper(lmc_params)

    kernel_params_univariate = KernelGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=1,
    )
    kernel_gen_univariate = KernelGeneratorWrapper(kernel_params_univariate)

    kernel_params_multivariate = KernelGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=5,
    )
    kernel_gen_multivariate = KernelGeneratorWrapper(kernel_params_multivariate)

    gp_params_univariate = GPGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=1,
    )
    gp_gen_univariate = GPGeneratorWrapper(gp_params_univariate)

    gp_params_multivariate = GPGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=5,
    )
    gp_gen_multivariate = GPGeneratorWrapper(gp_params_multivariate)

    forecast_pfn_univariate_params = ForecastPFNGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=1,
    )
    forecast_pfn_univariate_gen = ForecastPFNGeneratorWrapper(
        forecast_pfn_univariate_params
    )

    forecast_pfn_multivariate_params = ForecastPFNGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=5,
    )

    forecast_pfn_multivariate_gen = ForecastPFNGeneratorWrapper(
        forecast_pfn_multivariate_params
    )

    # Visualize samples from all generators
    visualize_batch_sample(lmc_gen, batch_size=batch_size, output_dir=output_dir)
    visualize_batch_sample(
        kernel_gen_univariate, batch_size=batch_size, output_dir=output_dir
    )
    visualize_batch_sample(
        kernel_gen_multivariate, batch_size=batch_size, output_dir=output_dir
    )
    visualize_batch_sample(
        gp_gen_univariate, batch_size=batch_size, output_dir=output_dir
    )
    visualize_batch_sample(
        gp_gen_multivariate, batch_size=batch_size, output_dir=output_dir
    )
    visualize_batch_sample(
        forecast_pfn_univariate_gen, batch_size=batch_size, output_dir=output_dir
    )
    visualize_batch_sample(
        forecast_pfn_multivariate_gen, batch_size=batch_size, output_dir=output_dir
    )

    # Optionally, load and visualize the first sample from a saved batch
    # visualize_saved_batch()
