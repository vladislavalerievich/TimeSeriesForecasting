import logging
import os

import numpy as np

from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.synthetic_generation.generator_params import KernelGeneratorParams
from src.synthetic_generation.kernel_synth.kernel_generator_wrapper import (
    KernelGeneratorWrapper,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_batch(
    generator,
    batch_size: int = 10,
    output_dir: str = "outputs/plots",
    add_artificial_predictions: bool = False,
) -> None:
    """
    Visualize all samples from a batch of synthetic multivariate time series from any generator.
    Also plot artificial predictions for demonstration if requested.

    Args:
        generator: Any generator wrapper (LMC, Kernel, GP, etc.)
        batch_size: Number of samples to generate in the batch
        output_dir: Directory to save plots
        add_artificial_predictions: Whether to add artificial predictions to the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    use_gpytorch = generator.params.use_gpytorch
    gpytorch_str = "gpytorch" if use_gpytorch else "sklearn"
    generator_name = f"{generator.__class__.__name__}_{gpytorch_str}"

    logger.info(f"[{generator_name}] Generating batch of size {batch_size}")

    # Generate batch
    batch = generator.generate_batch(batch_size=batch_size)
    logger.info(
        f"[{generator_name}] Batch history values shape: {batch.history_values.shape}"
    )
    logger.info(
        f"[{generator_name}] Batch future values shape: {batch.future_values.shape}"
    )

    for sample_idx in range(batch_size):
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
        filename = f"kernel_{gpytorch_str}_{series_type}_sample_{sample_idx}.png"
        output_file = os.path.join(output_dir, filename)

        # Create title
        title = f"KernelSynth ({gpytorch_str.upper()}) Synthetic {series_type.title()} Time Series (Sample {sample_idx})"

        plot_from_container(
            batch=batch,
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


if __name__ == "__main__":
    # Configuration
    batch_size = 10
    history_length = 256
    future_length = 128
    output_dir = "outputs/plots/gpytorch_comparison"
    global_seed = 2025

    logger.info(f"Saving plots to {output_dir}")

    # --- Sklearn (use_gpytorch=False) ---
    logger.info("--- Generating samples with scikit-learn backend ---")
    kernel_params_sklearn_multi = KernelGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=5,
        use_gpytorch=False,
    )
    kernel_gen_sklearn_multi = KernelGeneratorWrapper(kernel_params_sklearn_multi)

    visualize_batch(
        kernel_gen_sklearn_multi, batch_size=batch_size, output_dir=output_dir
    )

    # --- GPyTorch (use_gpytorch=True) ---
    logger.info("--- Generating samples with GPyTorch backend ---")
    kernel_params_gpytorch_multi = KernelGeneratorParams(
        global_seed=global_seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=5,
        use_gpytorch=True,
    )
    kernel_gen_gpytorch_multi = KernelGeneratorWrapper(kernel_params_gpytorch_multi)

    visualize_batch(
        kernel_gen_gpytorch_multi, batch_size=batch_size, output_dir=output_dir
    )
