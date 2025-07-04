import argparse
import logging
import os
import time
import warnings
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import torchmetrics
import yaml
from linear_operator.utils.cholesky import NumericalWarning
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb
from src.data_handling.data_loaders import CyclicGiftEvalDataLoader, GiftEvalDataLoader
from src.gift_eval.evaluator import GiftEvaluator
from src.models.unified_model import TimeSeriesModel
from src.utils.utils import (
    device,
    effective_rank,
    generate_descriptive_model_name,
    seed_everything,
)

warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging with better formatting
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GiftEvalTrainingPipeline:
    """Training pipeline that uses GIFT-eval datasets for training and validation."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.initial_epoch = 0
        self.gift_evaluator = None
        self.log_interval = self.config.get("log_interval", 10)

        # Performance tracking
        self.best_val_loss = float("inf")
        self.best_gift_eval_metrics = {}
        self.epoch_times = []

        # Global step tracking for consistent wandb logging
        self.global_step = 0

        # Gradient accumulation parameters
        self.gradient_accumulation_enabled = self.config.get(
            "gradient_accumulation_enabled", False
        )
        self.accumulation_steps = self.config.get("accumulation_steps", 1)
        if self.gradient_accumulation_enabled:
            logger.info(
                f"🔄 Gradient accumulation enabled with {self.accumulation_steps} steps"
            )
            effective_batch_size = self.config["batch_size"] * self.accumulation_steps
            logger.info(f"📊 Effective batch size: {effective_batch_size}")

        # Metrics scope control - whether to reset metrics at each logging interval
        self.reset_metrics_at_log_interval = self.config.get(
            "reset_metrics_at_log_interval", True
        )
        if self.reset_metrics_at_log_interval:
            logger.info(
                "📊 Metrics will be reset at each logging interval (same scope as loss)"
            )
        else:
            logger.info(
                "📊 Metrics will accumulate throughout the epoch (different scope from loss)"
            )

        logger.info("🚀 Initializing GIFT-eval training pipeline...")
        logger.info(f"🔧 Using device: {self.device}")
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

        # --- Setup GIFT-eval data loaders ---
        logger.info("📊 Setting up GIFT-eval data loaders...")

        # max_context_length represents total window length (context + forecast)
        # e.g., if max_context_length=2048 and prediction_length=720,
        # then effective max context length = 2048 - 720 = 1328
        max_context_length = self.config.get("max_context_length", 2048)

        # Window configuration
        max_training_windows = self.config.get("max_training_windows", 20)
        max_evaluation_windows = self.config.get("max_evaluation_windows", 1)

        logger.info(
            f"🔧 Window configuration: max_training_windows={max_training_windows}, max_evaluation_windows={max_evaluation_windows}"
        )

        # Get datasets to use (if specified in config, otherwise use all)
        datasets_to_use = self.config.get("datasets_to_use", None)

        # Training data loader (uses specified datasets or all datasets by default)
        self.train_loader = GiftEvalDataLoader(
            mode="train",
            batch_size=self.config["batch_size"],
            device=self.device,
            shuffle=True,
            to_univariate=self.config.get("to_univariate", False),
            max_context_length=max_context_length,
            max_windows=max_training_windows,
            skip_datasets_with_nans=self.config.get("skip_datasets_with_nans", True),
            datasets_to_use=datasets_to_use,
        )

        # Validate that we have data
        if len(self.train_loader) == 0:
            raise RuntimeError(
                "No training data available! Check your dataset configuration."
            )

        # Wrap train loader with CyclicGiftEvalDataLoader for fixed iterations per epoch
        num_training_iterations = self.config.get(
            "num_training_iterations_per_epoch", 100
        )
        self.train_loader = CyclicGiftEvalDataLoader(
            self.train_loader, num_training_iterations
        )
        logger.info(f"🔧 Training iterations per epoch: {num_training_iterations}")

        # Validation data loader (uses specified datasets or all datasets by default)
        self.val_loader = GiftEvalDataLoader(
            mode="validation",
            batch_size=self.config["batch_size"],
            device=self.device,
            shuffle=False,
            to_univariate=self.config.get("to_univariate", False),
            max_context_length=max_context_length,
            max_windows=max_evaluation_windows,
            skip_datasets_with_nans=self.config.get("skip_datasets_with_nans", True),
            datasets_to_use=datasets_to_use,
        )

        # Validate that we have validation data
        if len(self.val_loader) == 0:
            logger.warning(
                "No validation data available! Training will continue without validation."
            )

        # --- Setup GIFT evaluator for test evaluation ---
        self.gift_evaluator = GiftEvaluator(
            model=self.model,
            device=self.device,
            max_context_length=max_context_length,
            max_windows=max_evaluation_windows,
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
                    name=f"{self.config['model_name']}_gift_eval",
                    resume="allow",
                    tags=["gift-eval", "training"],
                )
                # Log model architecture
                wandb.watch(self.model, log="all", log_freq=100)
                logger.info("✅ WandB initialized successfully")
            except Exception as e:
                logger.error(f"❌ WandB initialization failed: {e}")
                self.config["wandb"] = False

    def _load_checkpoint(self) -> None:
        """Load model checkpoint if available and continuing training."""
        checkpoint_path = (
            f"{self.config['model_path']}/{self.config['model_name']}_gift_eval.pth"
        )
        if self.config["continue_training"] and os.path.exists(checkpoint_path):
            logger.info(f"📁 Loading checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.config["lr_scheduler"] == "cosine":
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.initial_epoch = ckpt["epoch"]
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.global_step = ckpt.get("global_step", 0)
            logger.info(
                f"✅ Resumed from epoch {self.initial_epoch + 1}, global step {self.global_step}"
            )
        else:
            logger.info("🆕 Starting fresh training (no checkpoint found)")

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.config["lr_scheduler"] == "cosine"
            else None,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }

        # Regular checkpoint
        checkpoint_path = (
            f"{self.config['model_path']}/{self.config['model_name']}_gift_eval.pth"
        )
        torch.save(ckpt, checkpoint_path)

        # Best model checkpoint
        if is_best:
            best_path = f"{self.config['model_path']}/{self.config['model_name']}_gift_eval_best.pth"
            torch.save(ckpt, best_path)
            logger.info(f"🏆 New best model saved! Val Loss: {val_loss:.4f}")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_cached_gb": torch.cuda.memory_reserved() / 1e9,
            }
        return {}

    def _inverse_scale(
        self, scaled_data: torch.Tensor, scale_statistics: dict
    ) -> torch.Tensor:
        """Apply inverse scaling to convert predictions back to original scale."""
        return self.model.scaler.inverse_scale(scaled_data, scale_statistics)

    def _compute_normalized_loss(self, output: dict, target: torch.Tensor) -> tuple:
        """Compute normalized loss for training."""
        # Compute loss on scaled values for stable training
        loss = self.model.compute_loss(target, output)

        # Get inverse scaled predictions for logging/metrics
        inv_scaled_output = self._inverse_scale(
            output["result"], output["scale_statistics"]
        )

        return loss, inv_scaled_output

    def _update_metrics(
        self, metrics: Dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update metric calculations for multivariate data."""
        predictions = predictions.contiguous()
        targets = targets.contiguous()

        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            if predictions.shape[1] > targets.shape[1]:
                predictions = predictions[:, : targets.shape[1], :]
            elif targets.shape[1] > predictions.shape[1]:
                targets = targets[:, : predictions.shape[1], :]

        # Flatten channel dimension for metric computation
        if predictions.dim() == 3 and targets.dim() == 3:
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

    def _log_metrics_to_wandb(
        self, metrics_dict: Dict, prefix: str, epoch: int, step: int = None
    ) -> None:
        """Enhanced WandB logging with better organization."""
        if not self.config["wandb"]:
            return

        wandb_dict = {}

        # Core metrics
        for name, value in metrics_dict.items():
            wandb_dict[f"{prefix}/metrics/{name}"] = value

        # Training diagnostics
        if prefix == "train":
            memory_stats = self._get_memory_usage()
            for name, value in memory_stats.items():
                wandb_dict[f"system/{name}"] = value

        # Add step and epoch info
        wandb_dict[f"{prefix}/epoch"] = epoch
        if step is not None:
            wandb_dict[f"{prefix}/step"] = step

        wandb.log(wandb_dict, step=step)

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch on GIFT-eval data."""
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        gradient_norms = []
        accumulated_loss = 0.0

        # Separate tracking for progress bar display
        display_loss = 0.0
        batches_processed = 0

        # Initialize gradients for the first accumulation step
        self.optimizer.zero_grad()

        # Progress bar for training
        train_pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"🏋️  Epoch {epoch + 1}/{self.config['num_epochs']} [TRAIN]",
            leave=False,
        )

        # Start timer for the first batch's data loading
        batch_start_time = time.time()

        for batch_idx, batch in train_pbar:
            logger.info(f"--- 🔍 Batch {batch_idx + 1} Timing Analysis ---")

            # --- Measure Data Loading and Preparation ---
            data_loading_end_time = time.time()
            data_loading_time = data_loading_end_time - batch_start_time
            logger.info(f"  📥 Data Loading & Prep : {data_loading_time:.4f}s")

            batch.to_device(self.device)
            logger.info(
                f"  📊 History Shape: {batch.history_values.shape}, Future Shape: {batch.future_values.shape}"
            )

            # --- Measure Forward Pass ---
            forward_start_time = time.time()
            output = self.model(batch)
            forward_end_time = time.time()
            forward_time = forward_end_time - forward_start_time
            logger.info(f"  🚀 Forward Pass        : {forward_time:.4f}s")

            # --- Measure Loss Computation ---
            loss_start_time = time.time()
            loss, inv_scaled_output = self._compute_normalized_loss(
                output, batch.future_values
            )
            loss_end_time = time.time()
            loss_time = loss_end_time - loss_start_time
            logger.info(f"  📈 Loss Computation    : {loss_time:.4f}s")

            # Track original loss for display (before scaling)
            original_loss = loss.item()
            display_loss += original_loss
            batches_processed += 1

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_enabled:
                loss = loss / self.accumulation_steps

            # --- Measure Backward Pass ---
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            backward_time = backward_end_time - backward_start_time
            logger.info(f"  ⬅️  Backward Pass       : {backward_time:.4f}s")

            accumulated_loss += loss.item()

            # Check if we should update weights
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = batch_idx + 1 >= len(self.train_loader)

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

                # Increment global step counter when optimizer actually steps
                self.global_step += 1

                optimizer_step_end_time = time.time()
                optimizer_time = optimizer_step_end_time - optimizer_step_start_time
                logger.info(f"  ⚙️  Optimizer Step      : {optimizer_time:.4f}s")

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
                    "  ⚙️  Optimizer Step      : 0.0000s (Accumulating Gradients)"
                )

            # --- Measure Metrics Update ---
            metrics_update_start_time = time.time()
            self._update_metrics(
                self.train_metrics, inv_scaled_output, batch.future_values
            )
            metrics_update_end_time = time.time()
            metrics_time = metrics_update_end_time - metrics_update_start_time
            logger.info(f"  📊 Metrics Update      : {metrics_time:.4f}s")

            # Update progress bar with timing info - use proper average for display
            current_loss = display_loss / batches_processed
            total_batch_time = forward_time + backward_time + loss_time + metrics_time
            if (
                not self.gradient_accumulation_enabled
                or is_accumulation_step
                or is_last_batch
            ):
                total_batch_time += optimizer_time

            train_pbar.set_postfix(
                {
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "grad_norm": f"{gradient_norms[-1] if gradient_norms else 0:.2f}",
                    "batch_time": f"{total_batch_time:.3f}s",
                }
            )

            # Log progress - only when we actually performed an optimizer step
            if (
                not self.gradient_accumulation_enabled
                or is_accumulation_step
                or is_last_batch
            ) and self.global_step % self.log_interval == 0:
                avg_grad_norm = np.mean(gradient_norms[-self.log_interval :])
                self._log_training_metrics(
                    epoch, running_loss, avg_grad_norm, self.log_interval
                )
                running_loss = 0.0

                # Reset metrics if configured to match loss scope
                if self.reset_metrics_at_log_interval:
                    for metric in self.train_metrics.values():
                        metric.reset()

            # --- Reset batch timer for the next iteration's data loading ---
            batch_start_time = time.time()

        # Calculate average epoch loss
        total_batches = len(self.train_loader)
        if self.gradient_accumulation_enabled:
            effective_updates = (
                total_batches + self.accumulation_steps - 1
            ) // self.accumulation_steps
            return epoch_loss / max(1, effective_updates)
        else:
            return epoch_loss / max(1, total_batches)

    def _log_training_metrics(
        self,
        epoch: int,
        running_loss: float,
        avg_grad_norm: float,
        log_interval: int,
    ) -> None:
        """Log training metrics during training.

        Note: Loss represents average over last log_interval optimizer steps.
        Metrics scope depends on reset_metrics_at_log_interval config:
        - If True: metrics represent values over last log_interval optimizer steps (same as loss)
        - If False: metrics represent cumulative values since epoch start (different from loss)
        """
        # Calculate average loss per optimizer step (not per batch when gradient accumulation is used)
        if self.gradient_accumulation_enabled:
            avg_loss = running_loss / log_interval  # Average loss per optimizer step
            loss_description = "avg_loss_per_opt_step"
        else:
            avg_loss = (
                running_loss / log_interval
            )  # Average loss per batch (same as optimizer step)
            loss_description = "avg_loss_per_batch"

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
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **computed_metrics,
            **effective_ranks,
        }

        # Add metadata about what the metrics represent
        train_metrics["loss_type"] = loss_description
        train_metrics["metrics_scope"] = (
            "interval" if self.reset_metrics_at_log_interval else "epoch"
        )
        if self.gradient_accumulation_enabled:
            train_metrics["accumulation_steps"] = self.accumulation_steps

        # Log to WandB with better organization using global step
        self._log_metrics_to_wandb(train_metrics, "train", epoch, self.global_step)

    def _validate_epoch(self, epoch: int) -> float:
        """Validate model on GIFT-eval validation datasets."""
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        # Progress bar for validation
        val_pbar = tqdm(
            self.val_loader,
            desc=f"🔍 Epoch {epoch + 1}/{self.config['num_epochs']} [VAL]  ",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                batch.to_device(self.device)

                output = self.model(batch)
                loss, inv_scaled_output = self._compute_normalized_loss(
                    output, batch.future_values
                )

                total_val_loss += loss.item()
                self._update_metrics(
                    self.val_metrics, inv_scaled_output, batch.future_values
                )

                num_batches += 1

                # Update progress bar
                current_val_loss = total_val_loss / num_batches
                val_pbar.set_postfix(
                    {
                        "val_loss": f"{current_val_loss:.4f}",
                    }
                )

        avg_val_loss = total_val_loss / max(1, num_batches)

        # Log validation metrics
        computed_metrics = self._prepare_computed_metrics(self.val_metrics)
        val_metrics = {
            "loss": avg_val_loss,
            **computed_metrics,
        }

        # Log to WandB
        self._log_metrics_to_wandb(val_metrics, "validation", epoch, self.global_step)

        # --- Test evaluation on held-out test data ---
        if self.config.get("evaluate_on_test", True):
            gift_eval_start_time = time.time()
            logger.info("🧪 Running GIFT-eval test evaluation...")

            # Evaluate on all terms automatically
            all_gift_eval_metrics = {}

            # Progress bar for GIFT evaluation
            gift_eval_pbar = tqdm(
                GiftEvalDataLoader.TERMS,
                desc="📊 GIFT Evaluation",
                leave=False,
            )

            term_times = {}
            for term in gift_eval_pbar:
                term_start_time = time.time()
                gift_eval_pbar.set_description(f"📊 GIFT Eval [{term}]")
                gift_eval_metrics = self.gift_evaluator.evaluate_datasets(
                    datasets_to_eval=self.train_loader.base_loader.dataset_names,
                    term=term,
                    epoch=epoch,
                    plot=self.config["wandb"],
                )
                term_end_time = time.time()
                term_times[term] = term_end_time - term_start_time

                # Add term suffix to dataset names to avoid conflicts
                for dataset_key, metrics in gift_eval_metrics.items():
                    all_gift_eval_metrics[f"{dataset_key}_{term}"] = metrics

            gift_eval_end_time = time.time()
            total_gift_eval_time = gift_eval_end_time - gift_eval_start_time

            # Log GIFT evaluation timing to stdout only
            logger.info("⏱️  GIFT EVALUATION TIMING SUMMARY")
            for term, term_time in term_times.items():
                logger.info(
                    f"   📊 {term.upper()} term:    {term_time:.2f}s ({term_time / total_gift_eval_time * 100:.1f}%)"
                )
            logger.info(f"   🕐 Total GIFT Eval:     {total_gift_eval_time:.2f}s")

            self._log_gift_eval_metrics(epoch, all_gift_eval_metrics)
            logger.info("✅ GIFT-eval test evaluation finished.")

        return avg_val_loss

    def _log_gift_eval_metrics(self, epoch: int, gift_eval_metrics: Dict) -> None:
        """Log GIFT evaluation metrics to both WandB and console with better organization."""
        if not gift_eval_metrics:
            return

        # Calculate aggregated metrics
        aggregated_metrics = {}
        metric_names = set()
        for dataset_metrics in gift_eval_metrics.values():
            metric_names.update(dataset_metrics.keys())

        for metric_name in metric_names:
            values = [
                metrics[metric_name]
                for metrics in gift_eval_metrics.values()
                if metric_name in metrics
            ]
            if values:
                aggregated_metrics[f"avg_{metric_name}"] = np.mean(values)
                aggregated_metrics[f"std_{metric_name}"] = np.std(values)

        # Log to WandB with better organization
        if self.config["wandb"]:
            wandb_metrics = {"epoch": epoch}

            # Individual dataset metrics
            for dataset_key, metrics in gift_eval_metrics.items():
                clean_dataset_name = dataset_key.replace("/", "_").replace(" ", "_")
                for metric_name, value in metrics.items():
                    wandb_metrics[
                        f"gift_eval/datasets/{clean_dataset_name}/{metric_name}"
                    ] = value

            # Aggregated metrics
            for metric_name, value in aggregated_metrics.items():
                wandb_metrics[f"gift_eval/aggregated/{metric_name}"] = value

            wandb.log(wandb_metrics, step=self.global_step)

        # Enhanced console logging
        logger.info("=" * 100)
        logger.info(f"🏆 GIFT-EVAL TEST RESULTS - EPOCH {epoch + 1}")
        logger.info("=" * 100)

        # Log aggregated metrics first
        logger.info("📊 AGGREGATED METRICS:")
        for metric_name, value in aggregated_metrics.items():
            logger.info(f"   {metric_name}: {value:.4f}")

        logger.info("\n📋 INDIVIDUAL DATASET RESULTS:")
        for dataset_name, metrics in gift_eval_metrics.items():
            logger.info(f"   📁 {dataset_name}:")
            formatted_metrics = [
                f"{name}: {value:.4f}" for name, value in metrics.items()
            ]
            logger.info(f"      {', '.join(formatted_metrics)}")
        logger.info("=" * 100)

    def _log_epoch_summary(
        self, epoch: int, train_loss: float, val_loss: float, epoch_time: float
    ) -> None:
        """Log comprehensive epoch summary with performance tracking."""
        # Track performance
        self.epoch_times.append(epoch_time)
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        # Prepare computed metrics
        train_computed = self._prepare_computed_metrics(self.train_metrics)
        val_computed = self._prepare_computed_metrics(self.val_metrics)

        # Enhanced console summary
        logger.info("=" * 100)
        logger.info(
            f"📈 EPOCH {epoch + 1}/{self.config['num_epochs']} SUMMARY {'🏆 NEW BEST!' if is_best else ''}"
        )
        logger.info("=" * 100)
        logger.info(f"🔹 Training Loss:   {train_loss:.6f}")
        logger.info(
            f"🔹 Validation Loss: {val_loss:.6f} {'⬇️' if is_best else '⬆️'} (Best: {self.best_val_loss:.6f})"
        )
        logger.info(f"🔹 Learning Rate:   {self.config['learning_rate']:.2e}")
        logger.info(
            f"🔹 Epoch Time:      {epoch_time / 60:.2f} min (Avg: {np.mean(self.epoch_times) / 60:.2f} min)"
        )

        # ETA calculation
        if len(self.epoch_times) > 1:
            remaining_epochs = self.config["num_epochs"] - epoch - 1
            eta_seconds = np.mean(self.epoch_times) * remaining_epochs
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            if eta_hours >= 1:
                logger.info(f"🔹 ETA:             {eta_hours:.1f} hours")
            else:
                logger.info(f"🔹 ETA:             {eta_minutes:.1f} minutes")

        logger.info(
            f"🔹 Memory Usage:    {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            if torch.cuda.is_available()
            else "🔹 Memory Usage:    CPU mode"
        )
        logger.info("=" * 100)

        # Enhanced WandB logging
        if self.config["wandb"]:
            epoch_summary = {
                "epoch_summary/train_loss": train_loss,
                "epoch_summary/val_loss": val_loss,
                "epoch_summary/best_val_loss": self.best_val_loss,
                "epoch_summary/learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_summary/epoch_time_minutes": epoch_time / 60,
                "epoch_summary/avg_epoch_time_minutes": np.mean(self.epoch_times) / 60,
                "epoch_summary/is_best": is_best,
                **{f"epoch_summary/train_{k}": v for k, v in train_computed.items()},
                **{f"epoch_summary/val_{k}": v for k, v in val_computed.items()},
            }

            # Add memory stats
            memory_stats = self._get_memory_usage()
            epoch_summary.update(
                {f"epoch_summary/{k}": v for k, v in memory_stats.items()}
            )

            wandb.log(epoch_summary, step=self.global_step)

        # Save checkpoint (always save, but mark if best)
        self._save_checkpoint(epoch, val_loss, is_best)

    def train(self) -> None:
        """Execute the enhanced training pipeline."""
        logger.info("🚀 Starting GIFT-eval training...")
        logger.info("=" * 100)
        logger.info("⚙️  GIFT-EVAL TRAINING CONFIGURATION")
        logger.info("=" * 100)
        logger.info(f"🔹 Model:           {self.config['model_name']}")
        logger.info(f"🔹 Epochs:          {self.config['num_epochs']}")
        logger.info(f"🔹 Batch Size:      {self.config['batch_size']}")
        if self.gradient_accumulation_enabled:
            logger.info(
                f"🔹 Gradient Accum:  Enabled ({self.accumulation_steps} steps)"
            )
        else:
            logger.info("🔹 Gradient Accum:  Disabled")
        logger.info(f"🔹 Learning Rate:   {self.config['learning_rate']:.2e}")
        logger.info(f"🔹 Scaler:          {self.config['scaler']}")
        logger.info(f"🔹 Device:          {self.device}")
        logger.info(
            f"🔹 WandB:           {'Enabled ✅' if self.config['wandb'] else 'Disabled ❌'}"
        )
        logger.info(
            f"🔹 Training Windows: {self.config.get('max_training_windows', 30)} ({'All available' if self.config.get('max_training_windows', 30) == -1 else 'Limited'})"
        )
        logger.info(
            f"🔹 Eval Windows:    {self.config.get('max_evaluation_windows', 1)}"
        )

        model_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"🔹 Model Params:    {model_params:,}")
        logger.info(
            f"🔹 Datasets:        {len(self.train_loader.base_loader.dataset_names)} datasets"
        )
        logger.info(f"🔹 Terms:           {GiftEvalDataLoader.TERMS}")
        logger.info("=" * 100)

        # Main training loop with progress bar
        epoch_pbar = tqdm(
            range(self.initial_epoch, self.config["num_epochs"]),
            desc="🏋️  Training Progress",
            initial=self.initial_epoch,
            total=self.config["num_epochs"],
        )

        for epoch in epoch_pbar:
            start_time = time.time()

            # Training and validation with mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                train_loss = self._train_epoch(epoch)
                val_loss = self._validate_epoch(epoch)

            # Update main progress bar
            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "best_val": f"{self.best_val_loss:.4f}",
                }
            )

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

        logger.info("=" * 100)
        logger.info("🎉 GIFT-EVAL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"🏆 Best Validation Loss: {self.best_val_loss:.6f}")
        logger.info("=" * 100)

        if self.config["wandb"]:
            logger.info("📊 Finishing WandB logging...")
            wandb.finish()
            logger.info("✅ WandB logging finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/train_gift_eval.yaml",
        help="Path to config file for GIFT-eval training",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    os.environ["WANDB_MODE"] = "online" if config["wandb"] else "offline"
    pipeline = GiftEvalTrainingPipeline(config)
    pipeline.train()
