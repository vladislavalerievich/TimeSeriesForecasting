import argparse
import logging
import os
import time
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from linear_operator.utils.cholesky import NumericalWarning
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from src.data_handling.data_loaders import (
    SyntheticTrainDataLoader,
    SyntheticValidationDataLoader,
)
from src.gift_eval.evaluator import GiftEvaluator
from src.models.models import MultiStepModel
from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.utils.utils import (
    device,
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
            logger.info(
                f"Effective batch size: {self.config['batch_size'] * self.accumulation_steps}"
            )

        logger.info("Initializing training pipeline...")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Config: {yaml.dump(config)}")
        self._setup()

    def _setup(self) -> None:
        """Setup model, optimizer, scheduler, and data loaders."""
        seed_everything(self.config["seed"])

        # Initialize model
        self.model = MultiStepModel(
            base_model_config=self.config["BaseModelConfig"],
            encoder_config=self.config["EncoderConfig"],
            scaler=self.config["scaler"],
            time_feature_config=self.config.get("time_feature_config", {}),
            **self.config["MultiStepModel"],
        ).to(self.device)

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Load checkpoint if continuing training
        self.config["model_name"] = generate_descriptive_model_name(self.config)
        self._load_checkpoint()

        # --- Load pre-generated synthetic datasets from disk ---
        logger.info("Loading pre-generated synthetic datasets from disk...")

        # Training data loader (load from disk)
        train_data_path = self.config.get(
            "train_data_path",
            "data/synthetic_validation_dataset/train/dataset.pt",
        )
        logger.info(f"Training data path: {train_data_path}")
        self.train_loader = SyntheticTrainDataLoader(
            data_path=train_data_path,
            num_batches_per_epoch=self.config["num_training_iterations_per_epoch"],
            device=self.device,
            single_file=True,
            shuffle=False,
        )

        # Validation data loader (load from disk)
        val_data_path = self.config.get(
            "val_data_path",
            "data/synthetic_validation_dataset/val/dataset.pt",
        )
        logger.info(f"Validation data path: {val_data_path}")
        self.val_loader = SyntheticValidationDataLoader(
            data_path=val_data_path,
            device=self.device,
            single_file=True,
        )

        # --- Setup GIFT evaluator ---
        self.gift_evaluator = GiftEvaluator(
            model=self.model,
            device=self.device,
            max_context_length=self.config["max_history_length"],
        )

        # Setup loss function, metrics, wandb
        self._setup_loss_function()
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

    def _setup_loss_function(self) -> None:
        """Configure loss function based on config."""
        if self.config["loss"] == "mae":
            self.criterion = nn.L1Loss().to(self.device)
        elif self.config["loss"] == "mse":
            self.criterion = nn.MSELoss().to(self.device)
        else:
            raise ValueError("Loss function not supported")

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

    def _plot_fixed_examples(self, epoch: int, avg_val_loss: float) -> None:
        """Plot selected series from every validation batch and log to WandB."""
        # Selected indices to plot
        plot_indices = [0, 63]

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch.to_device(self.device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    output = self.model(batch)
                    inv_scaled_output = self._inverse_scale(
                        output["result"], output["scale_statistics"]
                    )
                pred_future = inv_scaled_output.cpu().numpy()

                # Plot each selected index and log to wandb
                for i in plot_indices:
                    # Check if this index exists in the current batch
                    if i >= batch.history_values.size(0):
                        continue

                    fig = plot_from_container(
                        ts_data=batch,
                        sample_idx=i,
                        predicted_values=pred_future,  # Pass full batch predictions
                        title=f"Epoch {epoch} - Val Batch {batch_idx + 1}, Sample {i} (Val Loss: {avg_val_loss:.4f})",
                        output_file=None,
                        show=False,
                    )

                    # Log each plot individually with unique names (synthetic validation prefix)
                    wandb.log(
                        {
                            f"synthetic_val/batch{batch_idx + 1}_sample{i}": wandb.Image(
                                fig
                            )
                        }
                    )
                    plt.close(fig)

    def _update_metrics(
        self, metrics: Dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update metric calculations for multivariate data."""
        predictions = predictions.contiguous()
        targets = targets.contiguous()

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

        Args:
            metrics_dict: Dictionary of metrics to log
            metric_type: Type of metrics (e.g., 'train', 'val', 'gift_eval')
            epoch: Current epoch
            step: Optional step number
            extra_info: Additional info to include in log message
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
            wandb_dict["epoch"] = epoch
            if step is not None:
                wandb_dict["step"] = step
            wandb.log(wandb_dict)

    def _log_training_metrics(
        self, epoch: int, step_idx: int, running_loss: float, avg_grad_norm: float
    ) -> None:
        """Log training metrics during training."""
        avg_loss = running_loss / self.log_interval

        # Prepare metrics dictionary
        train_metrics = {
            "loss": avg_loss,
            "gradient_norm": avg_grad_norm,
            "mape": self.train_metrics["mape"].compute().item(),
            "mse": self.train_metrics["mse"].compute().item(),
            "smape": self.train_metrics["smape"].compute().item(),
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
        val_metrics = {
            "loss": val_loss,
            "mape": self.val_metrics["mape"].compute().item(),
            "mse": self.val_metrics["mse"].compute().item(),
            "smape": self.val_metrics["smape"].compute().item(),
        }

        self._log_metrics(val_metrics, "val", epoch, extra_info="Synthetic Validation")

    def _log_gift_eval_metrics(self, epoch: int, gift_metrics: Dict) -> None:
        """Log GIFT evaluation metrics."""
        # Flatten nested gift eval metrics for logging
        flattened_metrics = {}
        for dataset_name, metrics in gift_metrics.items():
            for metric_name, value in metrics.items():
                flattened_metrics[f"{dataset_name}_{metric_name}"] = value

        if flattened_metrics:
            self._log_metrics(
                flattened_metrics,
                "gift_eval",
                epoch,
                extra_info="Real Dataset Evaluation",
            )

    def _log_epoch_summary(
        self, epoch: int, train_loss: float, val_loss: float, epoch_time: float
    ) -> None:
        """Log comprehensive epoch summary."""
        # Prepare epoch summary metrics
        epoch_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mape": self.train_metrics["mape"].compute().item(),
            "train_mse": self.train_metrics["mse"].compute().item(),
            "train_smape": self.train_metrics["smape"].compute().item(),
            "val_mape": self.val_metrics["mape"].compute().item(),
            "val_mse": self.val_metrics["mse"].compute().item(),
            "val_smape": self.val_metrics["smape"].compute().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch_time_minutes": epoch_time / 60,
        }

        # Log comprehensive summary
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch + 1} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Synthetic Validation Loss: {val_loss:.4f}")
        logger.info(
            f"Training MAPE: {epoch_metrics['train_mape']:.4f}, MSE: {epoch_metrics['train_mse']:.4f}, SMAPE: {epoch_metrics['train_smape']:.4f}"
        )
        logger.info(
            f"Synthetic Validation MAPE: {epoch_metrics['val_mape']:.4f}, MSE: {epoch_metrics['val_mse']:.4f}, SMAPE: {epoch_metrics['val_smape']:.4f}"
        )
        logger.info(f"Learning Rate: {epoch_metrics['learning_rate']:.8f}")
        logger.info(f"Epoch Time: {epoch_metrics['epoch_time_minutes']:.2f} minutes")
        logger.info("=" * 80)

        # Log to wandb
        if self.config["wandb"]:
            # Log structured epoch metrics
            wandb.log(
                {
                    "epoch_summary/train_loss": train_loss,
                    "epoch_summary/val_loss": val_loss,
                    "epoch_summary/train_mape": epoch_metrics["train_mape"],
                    "epoch_summary/train_mse": epoch_metrics["train_mse"],
                    "epoch_summary/train_smape": epoch_metrics["train_smape"],
                    "epoch_summary/val_mape": epoch_metrics["val_mape"],
                    "epoch_summary/val_mse": epoch_metrics["val_mse"],
                    "epoch_summary/val_smape": epoch_metrics["val_smape"],
                    "epoch_summary/learning_rate": epoch_metrics["learning_rate"],
                    "epoch_summary/epoch_time_minutes": epoch_metrics[
                        "epoch_time_minutes"
                    ],
                    "epoch": epoch,
                }
            )

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
            self._plot_fixed_examples(epoch, avg_val_loss)

        # --- GIFT-eval validation ---

        logger.info("Running GIFT-eval validation...")
        gift_eval_metrics = self.gift_evaluator.evaluate_datasets(
            datasets_to_eval=ALL_DATASETS,
            term="short",
            epoch=epoch,
            plot=self.config["wandb"],  # Only plot if wandb is enabled
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

            logger.info(f"--- Batch {batch_idx} Timing Analysis ---")

            # --- Measure Data Loading and Preparation ---
            data_loading_end_time = time.time()
            logger.info(
                f"  - Data Loading & Prep : {data_loading_end_time - batch_start_time:.4f}s"
            )
            batch.to_device(self.device)
            logger.info(
                f"  - History Shape: {batch.history_values.shape}, Future Shape: {batch.future_values.shape}"
            )

            # --- Measure Forward Pass ---
            forward_start_time = time.time()
            output = self.model(batch)
            forward_end_time = time.time()
            logger.info(
                f"  - Forward Pass        : {forward_end_time - forward_start_time:.4f}s"
            )

            # --- Measure Loss Computation ---
            loss_start_time = time.time()
            loss, inv_scaled_output = self._compute_normalized_loss(
                output, batch.future_values
            )
            loss_end_time = time.time()
            logger.info(
                f"  - Loss Computation    : {loss_end_time - loss_start_time:.4f}s"
            )

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_enabled:
                loss = loss / self.accumulation_steps

            # --- Measure Backward Pass ---
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            logger.info(
                f"  - Backward Pass       : {backward_end_time - backward_start_time:.4f}s"
            )

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
                optimizer_step_start_time = time.time()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("gradient_clip_val", 1.0),
                )
                gradient_norms.append(grad_norm.item())

                self.optimizer.step()
                self.optimizer.zero_grad()
                optimizer_step_end_time = time.time()
                logger.info(
                    f"  - Optimizer Step      : {optimizer_step_end_time - optimizer_step_start_time:.4f}s"
                )
                # --- End Measure Optimizer Step ---

                # Log accumulated loss
                if self.gradient_accumulation_enabled:
                    running_loss += accumulated_loss * self.accumulation_steps
                    epoch_loss += accumulated_loss * self.accumulation_steps
                    accumulated_loss = 0.0
                else:
                    running_loss += loss.item()
                    epoch_loss += loss.item()
            else:
                # If optimizer doesn't step, log a placeholder
                logger.info(
                    "  - Optimizer Step      : 0.0000s (Accumulating Gradients)"
                )

            # --- Measure Metrics Update ---
            metrics_update_start_time = time.time()
            self._update_metrics(
                self.train_metrics, inv_scaled_output, batch.future_values
            )
            metrics_update_end_time = time.time()
            logger.info(
                f"  - Metrics Update      : {metrics_update_end_time - metrics_update_start_time:.4f}s"
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
                    avg_grad_norm = np.mean(gradient_norms[-self.log_interval :])
                    self._log_training_metrics(
                        epoch, effective_step, running_loss, avg_grad_norm
                    )
                    running_loss = 0.0
            else:
                if (batch_idx + 1) % self.log_interval == 0:
                    avg_grad_norm = np.mean(gradient_norms[-self.log_interval :])
                    self._log_training_metrics(
                        epoch, batch_idx, running_loss, avg_grad_norm
                    )
                    running_loss = 0.0

            # --- Reset batch timer for the next iteration's data loading ---
            batch_start_time = time.time()

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
        logger.info(f"Loss Function: {self.config['loss']}")
        logger.info(f"Scaler: {self.config['scaler']}")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"WandB Logging: {'Enabled' if self.config['wandb'] else 'Disabled'}"
        )
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
