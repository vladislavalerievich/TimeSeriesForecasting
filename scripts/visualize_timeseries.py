import logging
import os

import numpy as np

from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.synthetic_generation.forecast_pfn_prior.forecast_pfn_generator_wrapper import (
    ForecastPFNGeneratorWrapper,
)
from src.synthetic_generation.generator_params import (
    ForecastPFNGeneratorParams,
    GPGeneratorParams,
    KernelGeneratorParams,
    LMCGeneratorParams,
    SineWaveGeneratorParams,
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
from src.synthetic_generation.sine_waves.sine_wave_generator_wrapper import (
    SineWaveGeneratorWrapper,
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
    add_artificial_predictions: bool = True,
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
    logger.info(f"[{generator_name}] Batch start: {batch.start}")
    logger.info(f"[{generator_name}] Batch frequency: {batch.frequency}")

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
    batch_size = 1
    total_length = 1536
    future_length = 512
    output_dir = "outputs/plots"
    global_seed = 2025

    logger.info(f"Saving plots to {output_dir}")

    # Create generators
    lmc_params = LMCGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=5,
    )
    lmc_gen = LMCGeneratorWrapper(lmc_params)

    kernel_params_univariate = KernelGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=1,
    )
    kernel_gen_univariate = KernelGeneratorWrapper(kernel_params_univariate)

    kernel_params_multivariate = KernelGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=5,
    )
    kernel_gen_multivariate = KernelGeneratorWrapper(kernel_params_multivariate)

    gp_params_univariate = GPGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=1,
    )
    gp_gen_univariate = GPGeneratorWrapper(gp_params_univariate)

    gp_params_multivariate = GPGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=5,
    )
    gp_gen_multivariate = GPGeneratorWrapper(gp_params_multivariate)

    forecast_pfn_univariate_params = ForecastPFNGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=1,
    )
    forecast_pfn_univariate_gen = ForecastPFNGeneratorWrapper(
        forecast_pfn_univariate_params
    )

    forecast_pfn_multivariate_params = ForecastPFNGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=5,
    )

    forecast_pfn_multivariate_gen = ForecastPFNGeneratorWrapper(
        forecast_pfn_multivariate_params
    )

    sine_wave_params = SineWaveGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=1,
    )
    sine_wave_univariate_gen = SineWaveGeneratorWrapper(sine_wave_params)

    sine_wave_params_multivariate = SineWaveGeneratorParams(
        global_seed=global_seed,
        total_length=total_length,
        future_length=future_length,
        num_channels=5,
    )
    sine_wave_gen_multivariate = SineWaveGeneratorWrapper(sine_wave_params_multivariate)

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
    visualize_batch_sample(
        sine_wave_univariate_gen, batch_size=batch_size, output_dir=output_dir
    )
    visualize_batch_sample(
        sine_wave_gen_multivariate, batch_size=batch_size, output_dir=output_dir
    )
