import argparse
import logging
import os

from src.synthetic_generation.dataset_composer import DefaultSyntheticComposer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic time series datasets for training and validation."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic_validation_dataset",
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
        default=20,
        help="Number of batches for the validation dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of time series per batch",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )

    parser.add_argument(
        "--history_length",
        type=str,
        default="256",
        help="History length range (min,max) or fixed value",
    )

    parser.add_argument(
        "--target_length",
        type=str,
        default="64",
        help="Target length range (min,max) or fixed value",
    )

    parser.add_argument(
        "--num_channels",
        type=str,
        default="1,8",
        help="Number of channels range (min,max) or fixed value",
    )
    parser.add_argument(
        "--save_as_single_file",
        action="store_true",
        default=True,
        help="Save all batches in a single file per dataset",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full_dataset",
        action="store_true",
        default=False,
        help="Generate both training and validation datasets",
    )
    group.add_argument(
        "--validation_dataset",
        action="store_true",
        default=True,
        help="Generate only a validation dataset",
    )

    return parser.parse_args()


def parse_range_or_value(value_str: str):
    """Parse a string as either a range (min,max) or a fixed value."""
    parts = value_str.split(",")
    if len(parts) == 1:
        return int(parts[0])
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError(f"Invalid format: {value_str}. Expected 'value' or 'min,max'")


def main():
    """Main function to generate synthetic datasets."""
    args = parse_args()

    # Parse range arguments
    history_length = parse_range_or_value(args.history_length)
    target_length = parse_range_or_value(args.target_length)
    num_channels = parse_range_or_value(args.num_channels)
    generator_proportions = {
        "lmc": 0.65,
        "kernel": 0.15,
        "gp": 0.15,
        "forecast_pfn": 0.05,
    }

    # Create dataset composer
    composer = DefaultSyntheticComposer(
        seed=args.seed,
        history_length=history_length,
        target_length=target_length,
        num_channels=num_channels,
        generator_proportions=generator_proportions,
    ).composer

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.full_dataset:
        logger.info(
            f"Generating and saving full dataset with {args.train_batches} training batches and {args.val_batches} validation batches with batch size {args.batch_size}"
        )
        composer.create_train_validation_datasets(
            output_dir=args.output_dir,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            batch_size=args.batch_size,
            save_as_single_file=args.save_as_single_file,
        )
        logger.info("Full dataset generation completed successfully!")
        logger.info(f"Saved datasets to {args.output_dir}")
    elif args.validation_dataset:
        logger.info(
            f"Generating and saving {args.val_batches} validation batches with batch size {args.batch_size}"
        )
        composer.save_dataset(
            output_dir=args.output_dir,
            num_batches=args.val_batches,
            batch_size=args.batch_size,
            save_as_single_file=args.save_as_single_file,
        )
        logger.info("Validation dataset generation completed successfully!")
        logger.info(f"Saved validation dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
