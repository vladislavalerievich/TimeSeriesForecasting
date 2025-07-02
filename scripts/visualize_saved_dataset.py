import argparse
import logging
import os
from typing import List, Union

from src.data_handling.data_loaders import SyntheticDataset
from src.plotting.plot_multivariate_timeseries import plot_from_container

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_dataset_batches(
    dataset_base_path: str = "data/synthetic_validation_dataset",
    output_base_dir: str = "outputs/plots/synthetic_validation_dataset/",
    num_batches: int = 10,
    sample_indices: Union[List[int], str, int] = 0,
    show: bool = False,
    single_file: bool = True,
) -> None:
    """
    Load train and validation datasets and visualize samples from the first n batches.

    Args:
        dataset_base_path: Base path to the dataset directory containing train/ and val/ subdirs
        output_base_dir: Base directory for saving plots
        num_batches: Number of batches to visualize from each dataset
        sample_indices: Sample indices to visualize within each batch. Can be:
                       - int: single sample index (e.g., 0)
                       - list: multiple sample indices (e.g., [0, 1, 2])
                       - str: 'all' to visualize all samples in each batch
        show: Whether to display plots interactively
        single_file: Whether datasets are saved as single files or individual batch files
    """
    # Define paths
    train_data_path = os.path.join(dataset_base_path, "train")
    val_data_path = os.path.join(dataset_base_path, "val")

    if single_file:
        train_data_path = os.path.join(train_data_path, "dataset.pt")
        val_data_path = os.path.join(val_data_path, "dataset.pt")

    train_output_dir = os.path.join(output_base_dir, "train")
    val_output_dir = os.path.join(output_base_dir, "val")

    # Create output directories
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # Process training dataset
    logger.info(f"Processing training dataset from {train_data_path}")
    try:
        train_dataset = SyntheticDataset(
            data_path=train_data_path, device=None, single_file=single_file
        )

        batches_to_process = min(num_batches, len(train_dataset))
        logger.info(f"Visualizing {batches_to_process} batches from training dataset")

        for batch_idx in range(batches_to_process):
            batch = train_dataset[batch_idx]

            # Determine sample indices for this batch
            batch_size = batch.history_values.shape[0]
            if sample_indices == "all":
                sample_indices_for_batch = list(range(batch_size))
            elif isinstance(sample_indices, list):
                # Validate and filter sample indices
                valid_indices = []
                for idx in sample_indices:
                    if 0 <= idx < batch_size:
                        valid_indices.append(idx)
                    else:
                        logger.warning(
                            f"Sample index {idx} is out of range for batch {batch_idx} "
                            f"(batch size: {batch_size}). Skipping this index."
                        )
                sample_indices_for_batch = valid_indices
            elif isinstance(sample_indices, int):
                if 0 <= sample_indices < batch_size:
                    sample_indices_for_batch = [sample_indices]
                else:
                    logger.warning(
                        f"Sample index {sample_indices} is out of range for batch {batch_idx} "
                        f"(batch size: {batch_size}). Using sample 0 instead."
                    )
                    sample_indices_for_batch = [0] if batch_size > 0 else []
            else:
                raise ValueError("Invalid format for sample_indices")

            for current_sample_idx in sample_indices_for_batch:
                output_file = os.path.join(
                    train_output_dir,
                    f"train_batch_{batch_idx:03d}_sample_{current_sample_idx:03d}.png",
                )

                try:
                    print(
                        f"Training Dataset - Batch {batch_idx}, Sample {current_sample_idx}, frequency: {batch.frequency}"
                    )
                    plot_from_container(
                        batch=batch,
                        sample_idx=current_sample_idx,
                        output_file=output_file,
                        show=show,
                        title=f"Training Dataset - Batch {batch_idx}, Sample {current_sample_idx}",
                    )
                    logger.info(
                        f"Saved training plot for batch {batch_idx}, sample {current_sample_idx} to {output_file}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save training plot for batch {batch_idx}, sample {current_sample_idx}: {e}"
                    )

    except Exception as e:
        logger.error(f"Failed to process training dataset: {e}")
        raise

    # Process validation dataset
    logger.info(f"Processing validation dataset from {val_data_path}")
    try:
        val_dataset = SyntheticDataset(
            data_path=val_data_path, device=None, single_file=single_file
        )

        batches_to_process = min(num_batches, len(val_dataset))
        logger.info(f"Visualizing {batches_to_process} batches from validation dataset")

        for batch_idx in range(batches_to_process):
            batch = val_dataset[batch_idx]

            # Determine sample indices for this batch
            batch_size = batch.history_values.shape[0]
            if sample_indices == "all":
                sample_indices_for_batch = list(range(batch_size))
            elif isinstance(sample_indices, list):
                # Validate and filter sample indices
                valid_indices = []
                for idx in sample_indices:
                    if 0 <= idx < batch_size:
                        valid_indices.append(idx)
                    else:
                        logger.warning(
                            f"Sample index {idx} is out of range for batch {batch_idx} "
                            f"(batch size: {batch_size}). Skipping this index."
                        )
                sample_indices_for_batch = valid_indices
            elif isinstance(sample_indices, int):
                if 0 <= sample_indices < batch_size:
                    sample_indices_for_batch = [sample_indices]
                else:
                    logger.warning(
                        f"Sample index {sample_indices} is out of range for batch {batch_idx} "
                        f"(batch size: {batch_size}). Using sample 0 instead."
                    )
                    sample_indices_for_batch = [0] if batch_size > 0 else []
            else:
                raise ValueError("Invalid format for sample_indices")

            for current_sample_idx in sample_indices_for_batch:
                output_file = os.path.join(
                    val_output_dir,
                    f"val_batch_{batch_idx:03d}_sample_{current_sample_idx:03d}.png",
                )

                try:
                    print(
                        f"Validation Dataset - Batch {batch_idx}, Sample {current_sample_idx}, frequency: {batch.frequency}"
                    )
                    plot_from_container(
                        batch=batch,
                        sample_idx=current_sample_idx,
                        output_file=output_file,
                        show=show,
                        title=f"Validation Dataset - Batch {batch_idx}, Sample {current_sample_idx}",
                    )
                    logger.info(
                        f"Saved validation plot for batch {batch_idx}, sample {current_sample_idx} to {output_file}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save validation plot for batch {batch_idx}, sample {current_sample_idx}: {e}"
                    )

    except Exception as e:
        logger.error(f"Failed to process validation dataset: {e}")
        raise

    logger.info("Dataset visualization completed successfully!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize samples from training and validation datasets."
    )

    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="data/synthetic_validation_dataset",
        help="Base path to the dataset directory containing train/ and val/ subdirs",
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="outputs/plots/synthetic_validation_dataset/",
        help="Base directory for saving plots",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to visualize from each dataset",
    )

    parser.add_argument(
        "--sample_indices",
        type=str,
        default="0",
        help="Sample indices to visualize within each batch. Can be: "
        "- int: single sample index (e.g., 0) "
        "- list: multiple sample indices (e.g., 0,1,2) "
        "- str: 'all' to visualize all samples in each batch",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    parser.add_argument(
        "--single_file",
        action="store_true",
        default=True,
        help="Whether datasets are saved as single files (dataset.pt) or individual batch files",
    )

    parser.add_argument(
        "--individual_files",
        action="store_true",
        help="Use this flag if datasets are saved as individual batch files instead of single files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle the single_file flag
    single_file = (
        not args.individual_files if args.individual_files else args.single_file
    )

    # Convert sample_indices to the correct format
    if args.sample_indices == "all":
        sample_indices = "all"
    else:
        sample_indices = [int(i) for i in args.sample_indices.split(",")]

    visualize_dataset_batches(
        dataset_base_path=args.dataset_base_path,
        output_base_dir=args.output_base_dir,
        num_batches=args.num_batches,
        sample_indices=sample_indices,
        show=args.show,
        single_file=single_file,
    )
