import logging
import os

import torch

from src.plotting.plot_multivariate_timeseries import plot_from_container

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def visualize_saved_batch(
    batch_path: str = "data/synthetic_validation_dataset/batch_00001.pt",
    output_dir: str = "outputs/plots/synthetic_validation_dataset/",
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
        batch = torch.load(batch_path, weights_only=False)
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
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00000.pt", sample_idx=0
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00001.pt", sample_idx=1
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00002.pt", sample_idx=2
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00003.pt", sample_idx=3
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00004.pt", sample_idx=4
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00005.pt", sample_idx=5
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00006.pt", sample_idx=6
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00007.pt", sample_idx=7
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00008.pt", sample_idx=8
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00009.pt", sample_idx=9
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00010.pt", sample_idx=10
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00011.pt", sample_idx=11
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00012.pt", sample_idx=12
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00013.pt", sample_idx=13
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00014.pt", sample_idx=14
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00015.pt", sample_idx=15
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00016.pt", sample_idx=16
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00017.pt", sample_idx=17
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00018.pt", sample_idx=18
    )
    visualize_saved_batch(
        batch_path="data/synthetic_validation_dataset/batch_00019.pt", sample_idx=19
    )
