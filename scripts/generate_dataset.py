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
        default=1000,
        help="Number of batches for the training dataset",
    )

    parser.add_argument(
        "--val_batches",
        type=int,
        default=100,
        help="Number of batches for the validation dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Number of time series per batch",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )

    parser.add_argument(
        "--range_proportions",
        type=str,
        default=None,
        help='JSON string for range proportions (e.g., \'{"short": 0.34, "medium": 0.33, "long": 0.33}\')',
    )
    parser.add_argument(
        "--generator_proportions",
        type=str,
        default=None,
        help='JSON string for generator proportions (e.g., \'{"short": {...}, "medium": {...}, "long": {...}}\')',
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
        default=True,
        help="Generate both training and validation datasets",
    )
    group.add_argument(
        "--validation_dataset",
        action="store_true",
        default=False,
        help="Generate only a validation dataset",
    )

    return parser.parse_args()


def main():
    """Main function to generate synthetic datasets."""
    args = parse_args()

    range_proportions = {
        "short": 0.34,
        "medium": 0.33,
        "long": 0.33,
    }

    generator_proportions = {
        "short": {
            "forecast_pfn": 0.00,
            "gp": 0.0,
            "kernel": 0.00,
            "lmc": 0.00,
            "sine_wave": 0.10,
        },
        "medium": {
            "forecast_pfn": 0.00,
            "gp": 0.0,
            "kernel": 0.00,
            "lmc": 0.00,
            "sine_wave": 0.10,
        },
        "long": {
            "forecast_pfn": 0.00,
            "gp": 0.0,
            "kernel": 0.00,
            "lmc": 0.00,
            "sine_wave": 0.10,
        },
    }

    # Create dataset composer
    composer = DefaultSyntheticComposer(
        seed=args.seed,
        range_proportions=range_proportions,
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
