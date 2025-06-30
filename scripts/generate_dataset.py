import argparse
import json
import logging
import os

from src.synthetic_generation.dataset_composer import DatasetComposer

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
        default=1,
        help="Number of batches for the training dataset",
    )

    parser.add_argument(
        "--val_batches",
        type=int,
        default=1,
        help="Number of batches for the validation dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of time series per batch",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility",
    )

    parser.add_argument(
        "--generator_proportions",
        type=str,
        default=None,
        help='JSON string for generator proportions (e.g., \'{"lmc": 0.43, "gp": 0.27, "kernel": 0.24, "forecast_pfn": 0.04, "sine_wave": 0.02}\')',
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

    # Parse generator proportions from command line or use defaults
    if args.generator_proportions:
        generator_proportions = json.loads(args.generator_proportions)
    else:
        generator_proportions = {
            "forecast_pfn": 0.00,
            "gp": 0.00,
            "kernel": 0.00,
            "lmc": 0.00,
            "sine_wave": 1.00,
        }

    # Create dataset composer
    composer = DatasetComposer(
        generator_proportions=generator_proportions,
        global_seed=args.seed,
    )

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
