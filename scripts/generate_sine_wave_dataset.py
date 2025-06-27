import argparse
import logging
import os

import torch

from src.synthetic_generation.generator_params import SineWaveGeneratorParams
from src.synthetic_generation.sine_waves.sine_wave_generator_wrapper import (
    SineWaveGeneratorWrapper,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sine_wave_params(
    history_length: int = 256,
    future_length: int = 128,
    num_channels: int = 1,
    seed: int = 42,
    period_range: tuple = (10.0, 100.0),
    amplitude_range: tuple = (0.5, 3.0),
    noise_level: float = 0.1,
) -> SineWaveGeneratorParams:
    """Create SineWaveGeneratorParams with fixed parameters."""
    return SineWaveGeneratorParams(
        global_seed=seed,
        history_length=history_length,
        future_length=future_length,
        num_channels=num_channels,
        period_range=period_range,
        amplitude_range=amplitude_range,
        noise_level=noise_level,
    )


def generate_sine_wave_dataset(
    output_dir: str,
    num_batches: int,
    batch_size: int,
    params: SineWaveGeneratorParams,
    dataset_name: str = "dataset",
    save_as_single_file: bool = True,
) -> None:
    """
    Generate a sine wave dataset.

    Parameters
    ----------
    output_dir : str
        Directory to save the dataset.
    num_batches : int
        Number of batches to generate.
    batch_size : int
        Number of time series per batch.
    params : SineWaveGeneratorParams
        Parameters for the sine wave generator.
    dataset_name : str, optional
        Name of the dataset (default: "dataset").
    save_as_single_file : bool, optional
        If True, save all batches in a single file (default: True).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create sine wave generator wrapper
    generator = SineWaveGeneratorWrapper(params)

    batches = []
    logger.info(f"Generating {num_batches} batches with batch size {batch_size}...")

    for batch_idx in range(num_batches):
        seed = params.global_seed + batch_idx
        batch = generator.generate_batch(batch_size=batch_size, seed=seed)
        batches.append(batch)

        if not save_as_single_file:
            # Save individual batch files
            batch_path = os.path.join(output_dir, f"batch_{batch_idx:05d}.pt")
            torch.save(batch, batch_path)
            logger.debug(f"Saved batch {batch_idx} to {batch_path}")

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Generated {batch_idx + 1}/{num_batches} batches")

    if save_as_single_file:
        # Save all batches in a single file
        dataset_path = os.path.join(output_dir, f"{dataset_name}.pt")
        torch.save(batches, dataset_path)
        logger.info(
            f"Saved {dataset_name} with {num_batches} batches to {dataset_path}"
        )

    logger.info(f"Dataset generation completed for {dataset_name}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate sine wave datasets for debugging time series models."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sine_wave_dataset",
        help="Directory to save the generated datasets",
    )

    parser.add_argument(
        "--train_batches",
        type=int,
        default=100,
        help="Number of batches for the training dataset",
    )

    parser.add_argument(
        "--val_batches",
        type=int,
        default=32,
        help="Number of batches for the validation dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of time series per batch",
    )

    parser.add_argument(
        "--history_length",
        type=int,
        default=256,
        help="Fixed history length for all time series",
    )

    parser.add_argument(
        "--future_length",
        type=int,
        default=128,
        help="Fixed future length for all time series",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="Number of channels (features) per time series",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )

    parser.add_argument(
        "--period_min",
        type=float,
        default=10.0,
        help="Minimum period for sine waves",
    )

    parser.add_argument(
        "--period_max",
        type=float,
        default=100.0,
        help="Maximum period for sine waves",
    )

    parser.add_argument(
        "--amplitude_min",
        type=float,
        default=0.5,
        help="Minimum amplitude for sine waves",
    )

    parser.add_argument(
        "--amplitude_max",
        type=float,
        default=3.0,
        help="Maximum amplitude for sine waves",
    )

    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="Noise level as a fraction of amplitude",
    )

    parser.add_argument(
        "--save_as_single_file",
        action="store_true",
        default=True,
        help="Save all batches in a single file per dataset",
    )

    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Generate only training dataset",
    )

    parser.add_argument(
        "--val_only",
        action="store_true",
        help="Generate only validation dataset",
    )

    return parser.parse_args()


def main():
    """Main function to generate sine wave datasets."""
    args = parse_args()

    # Create sine wave parameters
    params = create_sine_wave_params(
        history_length=args.history_length,
        future_length=args.future_length,
        num_channels=args.num_channels,
        seed=args.seed,
        period_range=(args.period_min, args.period_max),
        amplitude_range=(args.amplitude_min, args.amplitude_max),
        noise_level=args.noise_level,
    )

    logger.info("Sine Wave Dataset Generation Parameters:")
    logger.info(f"  History Length: {args.history_length}")
    logger.info(f"  Future Length: {args.future_length}")
    logger.info(f"  Number of Channels: {args.num_channels}")
    logger.info(f"  Period Range: ({args.period_min}, {args.period_max})")
    logger.info(f"  Amplitude Range: ({args.amplitude_min}, {args.amplitude_max})")
    logger.info(f"  Noise Level: {args.noise_level}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Random Seed: {args.seed}")

    if not args.val_only:
        # Generate training dataset
        train_dir = os.path.join(args.output_dir, "train")
        logger.info(f"Generating training dataset with {args.train_batches} batches...")
        generate_sine_wave_dataset(
            output_dir=train_dir,
            num_batches=args.train_batches,
            batch_size=args.batch_size,
            params=params,
            dataset_name="train_dataset",
            save_as_single_file=args.save_as_single_file,
        )

    if not args.train_only:
        # Generate validation dataset with different seed
        val_params = create_sine_wave_params(
            history_length=args.history_length,
            future_length=args.future_length,
            num_channels=args.num_channels,
            seed=args.seed + 10000,  # Different seed for validation
            period_range=(args.period_min, args.period_max),
            amplitude_range=(args.amplitude_min, args.amplitude_max),
            noise_level=args.noise_level,
        )

        val_dir = os.path.join(args.output_dir, "val")
        logger.info(f"Generating validation dataset with {args.val_batches} batches...")
        generate_sine_wave_dataset(
            output_dir=val_dir,
            num_batches=args.val_batches,
            batch_size=args.batch_size,
            params=val_params,
            dataset_name="val_dataset",
            save_as_single_file=args.save_as_single_file,
        )

    logger.info("All datasets generated successfully!")
    logger.info(f"Datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
