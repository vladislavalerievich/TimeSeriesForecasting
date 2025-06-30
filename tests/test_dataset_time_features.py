#!/usr/bin/env python3
"""
Test script to identify batches causing timestamp overflow errors in time feature generation.
"""

import logging
import sys
import traceback
from pathlib import Path

from src.data_handling.data_loaders import SyntheticDataset
from src.data_handling.time_features import compute_batch_time_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_dataset_time_features.log"),
    ],
)
logger = logging.getLogger(__name__)


def test_batch_time_features(
    batch, batch_idx, dataset_type, max_history=1024, max_prediction=900
):
    """Test time feature generation for a specific batch."""
    try:
        logger.info(f"=== {dataset_type.upper()} BATCH {batch_idx} ===")
        logger.info(f"History shape: {batch.history_values.shape}")
        logger.info(f"Future shape: {batch.future_values.shape}")
        logger.info(f"Frequency: {batch.frequency}")
        logger.info(f"Start times (first 3): {batch.start[:3]}")

        total_length = batch.history_length + batch.future_length
        logger.info(f"Total length: {total_length}")

        history_feats, target_feats = compute_batch_time_features(
            batch.start,
            max_history,
            max_prediction,
            batch.batch_size,
            batch.frequency,
            K_max=6,
            time_feature_config={},
        )

        logger.info(
            f"✓ Success - History features: {history_feats.shape}, Target features: {target_feats.shape}"
        )
        return True

    except Exception as e:
        logger.error(f"✗ FAILED - {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Log additional debug info for problematic batches
        logger.error("=== DEBUG INFO ===")
        try:
            # Check start time range
            start_timestamps = batch.start[:5]
            logger.error(f"Start timestamps (first 5): {start_timestamps}")

            # Check years - convert to string to see the years
            start_years = [str(ts)[:4] for ts in start_timestamps]
            logger.error(f"Start years: {start_years}")

            # Check for problematic dates
            for i, ts in enumerate(start_timestamps):
                logger.error(f"  Sample {i}: {ts} (type: {type(ts)})")

        except Exception as debug_e:
            logger.error(f"Could not extract debug info: {debug_e}")

        return False


def main():
    """Test all batches in both datasets."""
    logger.info("Starting dataset time feature test...")

    # Paths
    train_path = "data/synthetic_validation_dataset/train/dataset.pt"
    val_path = "data/synthetic_validation_dataset/val/dataset.pt"

    if not Path(train_path).exists():
        logger.error(f"Train dataset not found: {train_path}")
        return 1
    if not Path(val_path).exists():
        logger.error(f"Val dataset not found: {val_path}")
        return 1

    failed_batches = []

    # Test validation dataset first (since error occurred during validation)
    logger.info("Testing validation dataset...")
    try:
        logger.info(f"Loading dataset from {val_path}")
        val_dataset = SyntheticDataset(
            data_path=val_path,
            device=None,  # Don't move to device yet
            single_file=True,
        )

        logger.info(f"Validation dataset loaded with {len(val_dataset)} batches")

        # Test all validation batches
        for batch_idx in range(len(val_dataset)):
            logger.info(f"Processing validation batch {batch_idx}...")
            try:
                batch = val_dataset[batch_idx]
                success = test_batch_time_features(batch, batch_idx, "validation")
                if not success:
                    failed_batches.append(f"validation_batch_{batch_idx}")
            except Exception as e:
                logger.error(f"Failed to load validation batch {batch_idx}: {e}")
                failed_batches.append(f"validation_batch_{batch_idx}")

    except Exception as e:
        logger.error(f"Failed to load validation dataset: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

    # Test some training batches
    logger.info("Testing training dataset...")
    try:
        logger.info(f"Loading dataset from {train_path}")
        train_dataset = SyntheticDataset(
            data_path=train_path,
            device=None,
            single_file=True,
        )

        logger.info(f"Training dataset loaded with {len(train_dataset)} batches")

        # Test first 20 training batches
        max_train_batches = min(20, len(train_dataset))
        for batch_idx in range(max_train_batches):
            logger.info(f"Processing training batch {batch_idx}...")
            try:
                batch = train_dataset[batch_idx]
                success = test_batch_time_features(batch, batch_idx, "training")
                if not success:
                    failed_batches.append(f"training_batch_{batch_idx}")
            except Exception as e:
                logger.error(f"Failed to load training batch {batch_idx}: {e}")
                failed_batches.append(f"training_batch_{batch_idx}")

    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

    # Summary
    logger.info("=" * 60)
    if failed_batches:
        logger.error(f"FAILED batches: {failed_batches}")
        logger.error("Some batches have timestamp overflow issues!")
        return 1
    else:
        logger.info("ALL BATCHES PASSED - No timestamp issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
