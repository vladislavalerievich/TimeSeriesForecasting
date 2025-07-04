import argparse
import logging
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchmetrics
import wandb
import yaml
from linear_operator.utils.cholesky import NumericalWarning
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_handling.data_loaders import (
    SyntheticTrainDataLoader,
    SyntheticValidationDataLoader, NanAugmenter,
)
from src.gift_eval.data import Dataset
from src.gift_eval.evaluator import GiftEvaluator
from src.models.unified_model import TimeSeriesModel
from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.utils.utils import (
    device,
    effective_rank,
    generate_descriptive_model_name,
    seed_everything,
)

warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for GIFT-Eval datasets
SHORT_DATASETS = [
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "ett1/W",
    "ett2/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
]
MED_LONG_DATASETS = ["bizitobs_l2c/H"]
ALL_DATASETS = list(set(SHORT_DATASETS + MED_LONG_DATASETS))


def find_consecutive_nan_lengths(series: np.ndarray) -> list[int]:
    """Finds the lengths of all consecutive NaN blocks in a 1D array."""
    if series.ndim > 1:
        # For multivariate series, flatten to treat it as one long sequence
        series = series.flatten()

    is_nan = np.isnan(series)
    padded_is_nan = np.concatenate(([False], is_nan, [False]))
    diffs = np.diff(padded_is_nan.astype(int))

    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0]

    return (end_indices - start_indices).tolist()


def analyze_datasets_for_augmentation(gift_eval_path_str: str) -> dict:
    """
    Analyzes all datasets to derive statistics needed for NaN augmentation.
    This version collects the full distribution of NaN ratios.
    """
    logger.info("--- Starting Dataset Analysis for Augmentation (Full Distribution) ---")
    path = Path(gift_eval_path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Provided raw data path for augmentation analysis does not exist: {gift_eval_path_str}")

    dataset_names = []
    for dataset_dir in path.iterdir():
        if dataset_dir.name.startswith(".") or not dataset_dir.is_dir():
            continue
        freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        if freq_dirs:
            for freq_dir in freq_dirs:
                dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
        else:
            dataset_names.append(dataset_dir.name)

    total_series_count = 0
    series_with_nans_count = 0
    nan_ratio_distribution = []
    all_consecutive_nan_lengths = Counter()

    for ds_name in sorted(dataset_names):
        try:
            ds = Dataset(name=ds_name, term="short", to_univariate=False)
            for series_data in ds.training_dataset:
                total_series_count += 1
                target = np.atleast_1d(series_data['target'])
                num_nans = np.isnan(target).sum()

                if num_nans > 0:
                    series_with_nans_count += 1
                    # <<< MODIFIED: Collect the ratio for the distribution >>>
                    nan_ratio = num_nans / target.size
                    nan_ratio_distribution.append(float(nan_ratio))

                    nan_lengths = find_consecutive_nan_lengths(target)
                    all_consecutive_nan_lengths.update(nan_lengths)
        except Exception as e:
            logger.warning(f"Could not process {ds_name} for augmentation analysis: {e}")

    if total_series_count == 0:
        raise ValueError("No series were found during augmentation analysis. Check dataset path.")

    p_series_has_nan = series_with_nans_count / total_series_count if total_series_count > 0 else 0

    logger.info("--- Augmentation Analysis Complete ---")
    # Print summary statistics
    logger.info(f"Total series analyzed: {total_series_count}")
    logger.info(f"Series with NaNs: {series_with_nans_count} ({p_series_has_nan:.4f})")
    logger.info(f"NaN ratio distribution: {Counter(nan_ratio_distribution)}")
    logger.info(f"Consecutive NaN lengths distribution: {all_consecutive_nan_lengths}")
    logger.info("--- End of Dataset Analysis for Augmentation ---")
    return {
        "p_series_has_nan": p_series_has_nan,
        "nan_ratio_distribution": nan_ratio_distribution,
        "nan_length_distribution": all_consecutive_nan_lengths,
    }


class TrainingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.initial_epoch = 0
        self.gift_evaluator = None
        self.log_interval = self.config.get("log_interval", 10)

        # Gradient accumulation parameters
        self.gradient_accumulation_enabled = self.config.get(
            "gradient_accumulation_enabled", False
        )
        self.accumulation_steps = self.config.get("accumulation_steps", 1)
        if self.gradient_accumulation_enabled:
            logger.info(
                f"Gradient accumulation enabled with {self.accumulation_steps} steps"
            )

        logger.info("Initializing training pipeline...")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Config: {yaml.dump(config)}")
        self._setup()

    def _setup(self) -> None:
        """Setup model, optimizer, scheduler, and data loaders."""
        seed_everything(self.config["seed"])

        # Initialize model
        self.model = TimeSeriesModel(
            **self.config["TimeSeriesModel"],
        ).to(self.device)

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Load checkpoint if continuing training
        self.config["model_name"] = generate_descriptive_model_name(self.config)
        self._load_checkpoint()

        self.augmenter = None
        aug_config = self.config.get("data_augmentation", {}).get("nan_augmentation", {})
        if aug_config.get("enabled", False):
            logger.info("Initializing NaN data augmentation...")
            raw_data_path = aug_config.get("raw_data_path")
            if not raw_data_path:
                raise ValueError("`raw_data_path` must be provided in config for nan_augmentation.")

            # We need to import analyze_datasets_for_augmentation and NanAugmenter in trainer.py as well
            stats = analyze_datasets_for_augmentation(raw_data_path)
            self.augmenter = NanAugmenter(**stats)
            logger.info("NaN data augmentation enabled and configured.")

        # --- Load pre-generated synthetic datasets from disk ---
        logger.info("Loading pre-generated synthetic datasets from disk...")

        # Training data loader (load from disk)
        train_data_path = self.config.get(
            "train_data_path",
            "data/synthetic_validation_dataset_full_mix/train/dataset.pt",
        )
        logger.info(f"Training data path: {train_data_path}")
        self.train_loader = SyntheticTrainDataLoader(
            data_path=train_data_path,
            num_batches_per_epoch=self.config["num_training_iterations_per_epoch"],
            device=self.device,
            single_file=True,
            shuffle=False,
            augmenter=self.augmenter,  # <<< MODIFIED: Pass the initialized augmenter here
        )

        # Validation data loader (load from disk)
        val_data_path = self.config.get(
            "val_data_path",
            "data/synthetic_validation_dataset_full_mix/val/dataset.pt",
        )
        logger.info(f"Validation data path: {val_data_path}")
        self.val_loader = SyntheticValidationDataLoader(
            data_path=val_data_path,  # TODO: change back to val_data_path
            device=self.device,
            single_file=True,
        )

        # --- Setup GIFT evaluator ---
        # Use a reasonable default max context length for GIFT evaluation
        # This prevents memory issues with very long sequences while allowing dynamic lengths
        max_context_length = self.config.get("gift_eval_max_context_length", 2048)
        self.gift_evaluator = GiftEvaluator(
            model=self.model,
            device=self.device,
            max_context_length=max_context_length,
        )

        # Setup loss function, metrics, wandb
        self._setup_metrics()
        self._setup_wandb()

    def _setup_optimizer(self) -> None:
        """Configure optimizer and learning rate scheduler."""
        if self.config["lr_scheduler"] == "cosine":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.config["initial_lr"]
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
                eta_min=self.config["learning_rate"],
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.config["learning_rate"]
            )

    def _setup_metrics(self) -> None:
        """Initialize training and validation metrics."""
        self.train_metrics = {
            "mape": torchmetrics.MeanAbsolutePercentageError().to(self.device),
            "mse": torchmetrics.MeanSquaredError().to(self.device),
            "smape": torchmetrics.SymmetricMeanAbsolutePercentageError().to(
                self.device
            ),
        }
        self.val_metrics = {
            "mape": torchmetrics.MeanAbsolutePercentageError().to(self.device),
            "mse": torchmetrics.MeanSquaredError().to(self.device),
            "smape": torchmetrics.SymmetricMeanAbsolutePercentageError().to(
                self.device
            ),
        }

    def _setup_wandb(self):
        """Initialize wandb if enabled."""
        if self.config["wandb"]:
            try:
                self.run = wandb.init(
                    project="TimeSeriesForecasting",
                    config=self.config,
                    name=self.config["model_name"],
                    resume="allow",
                )
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}")
                self.config["wandb"] = False

    def _load_checkpoint(self) -> None:
        """Load model checkpoint if available and continuing training."""
        checkpoint_path = f"{self.config['model_path']}/{self.config['model_name']}.pth"
        if self.config["continue_training"] and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.config["lr_scheduler"] == "cosine":
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.initial_epoch = ckpt["epoch"]
        else:
            logger.info("No previous training states found, starting fresh")

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.config["lr_scheduler"] == "cosine"
            else None,
            "epoch": epoch,
        }
        torch.save(
            ckpt,
            f"{self.config['model_path']}/{self.config['model_name']}.pth",
        )
        logger.info(f"Checkpoint saved at epoch {epoch}")

    def _inverse_scale(
            self, scaled_data: torch.Tensor, scale_statistics: dict
    ) -> torch.Tensor:
        """
        Apply inverse scaling to convert predictions back to original scale.

        Args:
            scaled_data: Scaled predictions [batch_size, pred_len, num_channels]
            scale_statistics: Scaling statistics from forward pass

        Returns:
            Data in original scale
        """
        return self.model.scaler.inverse_scale(scaled_data, scale_statistics)

    def _compute_normalized_loss(self, output: dict, target: torch.Tensor) -> tuple:
        """
        Compute normalized loss for training.

        Args:
            output: Model output dictionary containing 'result' and 'scale_statistics'
            target: Ground truth future values

        Returns:
            Tuple of (loss, inverse_scaled_predictions)
        """
        # Compute loss on scaled values for stable training
        loss = self.model.compute_loss(target, output)

        # Get inverse scaled predictions for logging/metrics
        inv_scaled_output = self._inverse_scale(
            output["result"], output["scale_statistics"]
        )

        return loss, inv_scaled_output

    def _plot_validation_examples(
            self,
            epoch: int,
            avg_val_loss: float,
            plot_indices: List[int] = [0, 1, 2, 3, 4],
            plot_all: bool = True,
    ) -> None:
        """
        Plot validation examples and log to WandB.
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx > 0:
                    break
                batch.to_device(self.device)
                with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    output = self.model(batch)
                    inv_scaled_output = self._inverse_scale(
                        output["result"], output["scale_statistics"]
                    )

                # Convert the full prediction tensor to numpy
                pred_future = inv_scaled_output.cpu().numpy()

                # Determine which indices to plot
                batch_size = batch.history_values.size(0)
                if plot_all:
                    indices_to_plot = list(range(batch_size))
                else:
                    indices_to_plot = [i for i in plot_indices if i < batch_size]

                for i in indices_to_plot:
                    # The plotting function will now handle extracting the quantiles
                    fig = plot_from_container(
                        batch=batch,
                        sample_idx=i,
                        predicted_values=pred_future,  # Pass the full (batched) prediction tensor
                        model_quantiles=self.model.quantiles if self.model.loss_type == 'quantile' else None,
                        title=f"Epoch {epoch} - Val Batch {batch_idx + 1}, Sample {i} (Val Loss: {avg_val_loss:.4f})",
                        output_file=None,
                        show=False,
                    )

                    wandb.log(
                        {
                            f"synthetic_val_plots/batch{batch_idx + 1}_sample{i}": wandb.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)

    def _update_metrics(
            self, metrics: Dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update metric calculations for multivariate data."""
        # Handle quantile predictions for point metrics
        if self.model.loss_type == 'quantile':
            # Select the median prediction for point-based metrics like MAPE, MSE
            try:
                # Find the index of the 0.5 quantile
                median_idx = self.model.quantiles.index(0.5)
                # Slice the predictions to get the median forecast
                predictions = predictions[..., median_idx]
            except (ValueError, AttributeError):
                raise ValueError("Could not find median (0.5) in model's quantiles list for metric calculation.")

        predictions = predictions.contiguous()
        targets = targets.contiguous()

        # Ensure predictions and targets have the same shape
        # Predictions might be truncated to original length already
        if predictions.shape != targets.shape:
            # If shapes don't match, truncate predictions to target shape
            if predictions.shape[1] > targets.shape[1]:
                predictions = predictions[:, : targets.shape[1], :]
            elif targets.shape[1] > predictions.shape[1]:
                # This shouldn't happen, but handle it just in case
                targets = targets[:, : predictions.shape[1], :]

        # Both predictions and targets should be [batch_size, pred_len, num_channels]
        # For metrics, we flatten the channel dimension to compute across all channels
        if predictions.dim() == 3 and targets.dim() == 3:
            # Flatten to [batch_size * num_channels, pred_len] for metric computation
            batch_size, pred_len, num_channels = predictions.shape
            predictions = (
                predictions.permute(0, 2, 1)
                .contiguous()
                .view(batch_size * num_channels, pred_len)
            )
            targets = (
                targets.permute(0, 2, 1)
                .contiguous()
                .view(batch_size * num_channels, pred_len)
            )
        elif predictions.dim() != targets.dim():
            raise ValueError(
                f"Prediction and target dimensions don't match: {predictions.shape} vs {targets.shape}"
            )

        for metric in metrics.values():
            metric.update(predictions, targets)

    def _prepare_computed_metrics(self, metrics_dict: Dict) -> Dict[str, float]:
        """Helper method to compute and extract metric values."""
        return {name: metric.compute().item() for name, metric in metrics_dict.items()}

    def _log_metrics(
            self,
            metrics_dict: Dict,
            metric_type: str,
            epoch: int,
            step: int = None,
            extra_info: str = "",
    ) -> None:
        """
        Generic method to log metrics to both logger and wandb.
        """
        # Format metrics for logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])

        # Always log to Python logger
        log_msg = f"[{metric_type.upper()}] Epoch {epoch + 1}"
        if step is not None:
            log_msg += f", Step {step + 1}"
        if extra_info:
            log_msg += f" - {extra_info}"
        log_msg += f" | {metrics_str}"
        logger.info(log_msg)

        # Optionally log to wandb
        if self.config["wandb"]:
            wandb_dict = {f"{metric_type}/{k}": v for k, v in metrics_dict.items()}

            if metric_type == "train" and step is not None:
                wandb.log(wandb_dict)
            else:
                wandb.log({**wandb_dict, "epoch": epoch})

    def _log_training_metrics(
            self, epoch: int, step_idx: int, running_loss: float, avg_grad_norm: float
    ) -> None:
        """Log training metrics during training."""
        avg_loss = running_loss / self.log_interval

        # Prepare metrics dictionary
        computed_metrics = self._prepare_computed_metrics(self.train_metrics)
        num_heads = self.model.initial_hidden_state.shape[1]
        effective_ranks = {
            f"eff_rank_{head_idx}": effective_rank(
                self.model.initial_hidden_state[0, head_idx]
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            for head_idx in range(num_heads)
        }
        train_metrics = {
            "loss": avg_loss,
            "gradient_norm": avg_grad_norm,
            "init_norm": self.model.initial_hidden_state.norm().item(),
            **computed_metrics,
            **effective_ranks,
        }

        # Calculate global step
        if self.gradient_accumulation_enabled:
            global_step = (
                    epoch
                    * (
                            self.config["num_training_iterations_per_epoch"]
                            // self.accumulation_steps
                    )
                    + step_idx
            )
            extra_info = f"Effective Step: {step_idx + 1}"
        else:
            global_step = (
                    epoch * self.config["num_training_iterations_per_epoch"] + step_idx
            )
            extra_info = f"Batch: {step_idx + 1}"

        self._log_metrics(train_metrics, "train", epoch, global_step, extra_info)

    def _log_validation_metrics(self, epoch: int, val_loss: float) -> None:
        """Log validation metrics from synthetic validation data."""
        computed_metrics = self._prepare_computed_metrics(self.val_metrics)
        val_metrics = {
            "loss": val_loss,
            **computed_metrics,
        }

        self._log_metrics(val_metrics, "val", epoch, extra_info="Synthetic Validation")

    def _log_gift_eval_to_wandb(self, epoch: int, gift_eval_metrics: Dict) -> None:
        """Log GIFT evaluation metrics to WandB."""
        if not self.config["wandb"]:
            return

        wandb_metrics = {"epoch": epoch}

        for dataset_key, metrics in gift_eval_metrics.items():
            clean_dataset_name = dataset_key.replace("/", "_").replace(" ", "_")
            for metric_name, value in metrics.items():
                wandb_metrics[
                    f"gift_eval_metrics/{clean_dataset_name}/{metric_name}"
                ] = value

        wandb.log(wandb_metrics)

    def _log_gift_eval_to_console(self, epoch: int, gift_metrics: Dict) -> None:
        """Log GIFT evaluation metrics to console."""
        logger.info("=" * 80)
        logger.info(f"GIFT-EVAL RESULTS - EPOCH {epoch + 1}")
        logger.info("=" * 80)

        for dataset_name, metrics in gift_metrics.items():
            logger.info(f"Dataset: {dataset_name}")
            formatted_metrics = [
                f"{name}: {value:.4f}" for name, value in metrics.items()
            ]
            logger.info(f"  Metrics: {', '.join(formatted_metrics)}")
            logger.info("")

        logger.info("=" * 80)

    def _log_gift_eval_metrics(self, epoch: int, gift_eval_metrics: Dict) -> None:
        """Log GIFT evaluation metrics to both WandB and console."""
        if not gift_eval_metrics:
            return

        self._log_gift_eval_to_wandb(epoch, gift_eval_metrics)
        self._log_gift_eval_to_console(epoch, gift_eval_metrics)

    def _log_epoch_summary(
            self, epoch: int, train_loss: float, val_loss: float, epoch_time: float
    ) -> None:
        """Log comprehensive epoch summary."""
        # Prepare computed metrics
        train_computed = self._prepare_computed_metrics(self.train_metrics)
        val_computed = self._prepare_computed_metrics(self.val_metrics)

        # Prepare train metrics for logging
        train_summary_metrics = {
            "loss": train_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_time_minutes": epoch_time / 60,
            **train_computed,
        }

        # Prepare validation metrics for logging
        val_summary_metrics = {
            "loss": val_loss,
            **val_computed,
        }

        # Log detailed console summary (custom format for readability)
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch + 1} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Synthetic Validation Loss: {val_loss:.4f}")
        logger.info(
            f"Training MAPE: {train_computed['mape']:.4f}, MSE: {train_computed['mse']:.4f}, SMAPE: {train_computed['smape']:.4f}"
        )
        logger.info(
            f"Synthetic Validation MAPE: {val_computed['mape']:.4f}, MSE: {val_computed['mse']:.4f}, SMAPE: {val_computed['smape']:.4f}"
        )
        logger.info(f"Learning Rate: {train_summary_metrics['learning_rate']:.8f}")
        logger.info(
            f"Epoch Time: {train_summary_metrics['epoch_time_minutes']:.2f} minutes"
        )
        logger.info("=" * 80)

        # Log train and validation metrics separately
        self._log_metrics(
            train_summary_metrics, "train", epoch, extra_info="Epoch Summary"
        )
        self._log_metrics(val_summary_metrics, "val", epoch, extra_info="Epoch Summary")

    def _validate_epoch(self, epoch: int) -> float:
        """Validate model on all fixed synthetic validation batches."""
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch.to_device(self.device)
                output = self.model(batch)

                # Compute normalized loss and get inverse scaled output
                loss, inv_scaled_output = self._compute_normalized_loss(
                    output, batch.future_values
                )
                total_val_loss += loss.item()

                # Update metrics with inverse scaled predictions
                self._update_metrics(
                    self.val_metrics, inv_scaled_output, batch.future_values
                )
                num_batches += 1

        avg_val_loss = total_val_loss / max(1, num_batches)

        # Log validation metrics (always to logger, optionally to wandb)
        self._log_validation_metrics(epoch, avg_val_loss)

        # Plot examples if wandb is enabled
        if self.config["wandb"]:
            self._plot_validation_examples(epoch, avg_val_loss, plot_all=False)

        # --- GIFT-eval validation ---
        if self.config["evaluate_on_gift_eval"]:
            logger.info("Running GIFT-eval validation...")
            gift_eval_metrics = self.gift_evaluator.evaluate_datasets(
                datasets_to_eval=ALL_DATASETS,
                term="short",
                epoch=epoch,
                plot=False,  # Only plot if wandb is enabled
            )
            # Log GIFT eval metrics (always to logger, optionally to wandb)
            self._log_gift_eval_metrics(epoch, gift_eval_metrics)
            logger.info("GIFT-eval validation finished.")

        return avg_val_loss

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with per-batch performance timing."""
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        gradient_norms = []
        accumulated_loss = 0.0

        # Initialize gradients for the first accumulation step
        self.optimizer.zero_grad()

        # Start timer for the first batch's data loading
        batch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= self.config["num_training_iterations_per_epoch"]:
                break
            # --- Measure Data Loading and Preparation ---
            batch.to_device(self.device)

            # --- Measure Forward Pass ---
            output = self.model(batch)

            # --- Measure Loss Computation ---
            loss, inv_scaled_output = self._compute_normalized_loss(
                output, batch.future_values
            )

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_enabled:
                loss = loss / self.accumulation_steps

            # --- Measure Backward Pass ---
            loss.backward()
            accumulated_loss += loss.item()

            # Check if we should update weights
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = (
                    batch_idx + 1 >= self.config["num_training_iterations_per_epoch"]
            )

            if (
                    not self.gradient_accumulation_enabled
                    or is_accumulation_step
                    or is_last_batch
            ):
                # --- Measure Optimizer Step (includes grad clipping) ---
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("gradient_clip_val", 1.0),
                )
                gradient_norms.append(grad_norm.item())

                self.optimizer.step()
                self.optimizer.zero_grad()

                # --- End Measure Optimizer Step ---

                # Log accumulated loss
                if self.gradient_accumulation_enabled:
                    running_loss += accumulated_loss * self.accumulation_steps
                    epoch_loss += accumulated_loss * self.accumulation_steps
                    accumulated_loss = 0.0
                else:
                    running_loss += loss.item()
                    epoch_loss += loss.item()

            # --- Measure Metrics Update ---
            self._update_metrics(
                self.train_metrics, inv_scaled_output, batch.future_values
            )

            # Log progress based on effective updates
            effective_step = (
                batch_idx // self.accumulation_steps
                if self.gradient_accumulation_enabled
                else batch_idx
            )
            if self.gradient_accumulation_enabled:
                if (
                        is_accumulation_step
                        and (effective_step + 1) % self.log_interval == 0
                ):
                    avg_grad_norm = np.mean(gradient_norms[-self.log_interval:])
                    self._log_training_metrics(
                        epoch, effective_step, running_loss, avg_grad_norm
                    )
                    running_loss = 0.0
            else:
                if (batch_idx + 1) % self.log_interval == 0:
                    avg_grad_norm = np.mean(gradient_norms[-self.log_interval:])
                    self._log_training_metrics(
                        epoch, batch_idx, running_loss, avg_grad_norm
                    )
                    running_loss = 0.0

        # The end-of-epoch summary is removed in this version.

        # Calculate average epoch loss
        total_batches = min(
            batch_idx + 1, self.config["num_training_iterations_per_epoch"]
        )
        if self.gradient_accumulation_enabled:
            effective_updates = (
                                        total_batches + self.accumulation_steps - 1
                                ) // self.accumulation_steps
            return epoch_loss / max(1, effective_updates)
        else:
            return epoch_loss / max(1, total_batches)

    def train(self) -> None:
        """Execute the training pipeline."""
        logger.info("Starting training...")
        logger.info("=" * 80)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config['model_name']}")
        logger.info(f"Epochs: {self.config['num_epochs']}")
        if self.gradient_accumulation_enabled:
            logger.info(
                f"Gradient Accumulation: Enabled ({self.accumulation_steps} steps)"
            )
        else:
            logger.info("Gradient Accumulation: Disabled")
        logger.info(f"Learning Rate: {self.config['learning_rate']}")
        logger.info(f"Scaler: {self.config['scaler']}")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"WandB Logging: {'Enabled' if self.config['wandb'] else 'Disabled'}"
        )
        model_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Model Parameters: {model_params:,}")
        logger.info("=" * 80)

        for epoch in range(self.initial_epoch, self.config["num_epochs"]):
            start_time = time.time()

            # Training
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                train_loss = self._train_epoch(epoch)

                # Validation
                val_loss = self._validate_epoch(epoch)

            # Epoch summary
            epoch_time = time.time() - start_time
            self._log_epoch_summary(epoch, train_loss, val_loss, epoch_time)

            # Reset metrics
            for metric_dict in [self.train_metrics, self.val_metrics]:
                for metric in metric_dict.values():
                    metric.reset()

            # Update scheduler
            if self.config["lr_scheduler"] == "cosine":
                self.scheduler.step()

            # Save checkpoint
            if epoch % 5 == 4 or epoch == self.config["num_epochs"] - 1:
                self._save_checkpoint(epoch)

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        if self.config["wandb"]:
            logger.info("Finishing WandB logging...")
            wandb.finish()
            logger.info("WandB logging finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/train.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    os.environ["WANDB_MODE"] = "online" if config["wandb"] else "offline"
    pipeline = TrainingPipeline(config)
    pipeline.train()
