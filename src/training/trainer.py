import argparse
import logging
import os
import time
import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from linear_operator.utils.cholesky import NumericalWarning
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.gift_eval.evaluator import GiftEvaluator
from src.models.models import MultiStepModel
from src.plotting.plot_multivariate_timeseries import plot_from_container
from src.synthetic_generation.data_loaders import (
    SyntheticTrainDataLoader,
    SyntheticValidationDataLoader,
)
from src.synthetic_generation.dataset_composer import (
    DefaultSyntheticComposer,
    OnTheFlyDatasetGenerator,
)
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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

        # --- Synthetic training data generation setup ---
        composer = DefaultSyntheticComposer(
            seed=self.config["seed"],
            range_proportions=self.config.get("range_proportions", None),
            generator_proportions=self.config.get("generator_proportions", None),
        ).composer

        # On-the-fly training data loader
        on_the_fly_gen = OnTheFlyDatasetGenerator(
            composer=composer,
            batch_size=self.config["batch_size"],
            buffer_size=2,
            global_seed=self.config["seed"],
        )
        self.train_loader = SyntheticTrainDataLoader(
            generator=on_the_fly_gen,
            num_batches_per_epoch=self.config["num_training_iterations_per_epoch"],
            device=self.device,
        )

        # Fixed synthetic validation data loader (load all batches from disk)
        val_data_path = self.config.get(
            "val_data_path",
            "data/synthetic_validation_dataset/dataset.pt",
        )
        self.val_loader = SyntheticValidationDataLoader(
            data_path=val_data_path,
            batch_size=1,
            device=self.device,
            single_file=True,
        )

        # --- Setup GIFT evaluator ---
        self.gift_evaluator = GiftEvaluator(
            model=self.model,
            device=self.device,
            max_context_length=self.config["max_context_length"],
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
                    id=self.config["model_name"],
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

    def _inverse_scale(self, output: torch.Tensor, scale_params: Tuple) -> torch.Tensor:
        """Inverse scale multivariate model predictions using scaler's built-in method."""
        # Use the model's scaler inverse_transform method for consistency
        return self.model.scaler.inverse_transform(output, scale_params)

    def _prepare_batch(
        self, batch: BatchTimeSeriesContainer
    ) -> Tuple[BatchTimeSeriesContainer, torch.Tensor]:
        """Prepare batch data for model input."""
        batch.to_device(self.device)
        return batch, batch.future_values

    def _plot_fixed_examples(self, epoch: int, avg_val_loss: float) -> None:
        """Plot selected series from every validation batch and log to WandB."""
        # Selected indices to plot
        plot_indices = [0, 63]

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch.to_device(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                    output = self.model(batch)
                    inv_scaled_output = self._inverse_scale(
                        output["result"], output["scale_params"]
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

                    # Log each plot individually with unique names
                    wandb.log(
                        {f"val_plot_batch{batch_idx + 1}_sample{i}": wandb.Image(fig)}
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

    def _log_training_progress(
        self, epoch: int, batch_idx: int, running_loss: float
    ) -> None:
        """Log training progress."""
        avg_loss = running_loss / self.log_interval
        logger.info(
            f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, "
            f"Batch Len: {self.config['batch_size']}, Sc. Loss: {avg_loss:.4f}"
        )
        if self.config["wandb"]:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/mape": self.train_metrics["mape"].compute(),
                    "train/mse": self.train_metrics["mse"].compute(),
                    "train/smape": self.train_metrics["smape"].compute(),
                    "epoch": epoch,
                    "step": epoch * self.config["num_training_iterations_per_epoch"]
                    + batch_idx,
                }
            )

    def _validate_epoch(self, epoch: int) -> float:
        """Validate model on all fixed synthetic validation batches."""
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                data, target = self._prepare_batch(batch)
                with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                    output = self.model(data)
                    loss = self.model.compute_loss(target, output)

                total_val_loss += loss.item()

                # Inverse transform predictions for metrics calculation
                inv_scaled_output = self.model.scaler.inverse_transform(
                    output["result"], output["scale_params"]
                )
                self._update_metrics(self.val_metrics, inv_scaled_output, target)
                num_batches += 1

        avg_val_loss = total_val_loss / max(1, num_batches)
        if self.config["wandb"]:
            wandb.log(
                {
                    "val/sc_loss": avg_val_loss,
                    "val/mape": self.val_metrics["mape"].compute(),
                    "val/mse": self.val_metrics["mse"].compute(),
                    "val/smape": self.val_metrics["smape"].compute(),
                    "epoch": epoch,
                }
            )
            self._plot_fixed_examples(epoch, avg_val_loss)

        # --- GIFT-eval validation ---
        if self.config["wandb"] and epoch % 5 == 4:
            logger.info("Running GIFT-eval validation...")
            gift_eval_metrics = self.gift_evaluator.evaluate_datasets(
                datasets_to_eval=["us_births/D", "saugeenday/W", "ett1/W"],
                term="short",
                epoch=epoch,
                plot=True,
            )
            wandb.log({"gift_eval": gift_eval_metrics, "epoch": epoch})
            logger.info("GIFT-eval validation finished.")

        logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= self.config["num_training_iterations_per_epoch"]:
                break

            data, target = self._prepare_batch(batch)
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                output = self.model(data)
                loss = self.model.compute_loss(target, output)

            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

            # Update metrics with inverse transformed predictions
            inv_scaled_output = self.model.scaler.inverse_transform(
                output["result"], output["scale_params"]
            )
            self._update_metrics(self.train_metrics, inv_scaled_output, target)

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (batch_idx + 1) % self.log_interval == 0:
                self._log_training_progress(epoch, batch_idx, running_loss)
                running_loss = 0.0

        return epoch_loss / min(
            batch_idx + 1, self.config["num_training_iterations_per_epoch"]
        )

    def train(self) -> None:
        """Execute the training pipeline."""
        logger.info("Starting training...")

        for epoch in range(self.initial_epoch, self.config["num_epochs"]):
            start_time = time.time()

            # Training
            train_loss = self._train_epoch(epoch)

            # Validation
            val_loss = self._validate_epoch(epoch)

            # Logging
            if self.config["wandb"]:
                wandb.log(
                    {
                        "epoch_metrics": {
                            "train": {
                                "sc_loss": train_loss,
                                "mape": self.train_metrics["mape"].compute(),
                                "smape": self.train_metrics["smape"].compute(),
                            },
                            "val": {
                                "sc_loss": val_loss,
                                "mape": self.val_metrics["mape"].compute(),
                                "smape": self.val_metrics["smape"].compute(),
                            },
                        },
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Epoch summary
            epoch_time = time.time() - start_time
            logger.info(f"Epoch: {epoch + 1}, Time: {epoch_time / 60:.2f} mins")

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

        if self.config["wandb"]:
            wandb.finish()
        logger.info("Training completed successfully.")


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
