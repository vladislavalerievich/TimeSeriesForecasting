import argparse
import logging
import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset

import wandb
from plotting.plot_multivariate_timeseries import plot_from_container
from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.models.models import MultiStepModel
from src.synthetic_generation.multivariate_time_series_generator import (
    MultivariateTimeSeriesGenerator,
)
from src.utils.utils import (
    device,
    generate_model_save_name,
    seed_everything,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntheticDataset(IterableDataset):
    def __init__(
        self,
        generator,
        batch_size,
        num_batches,
        history_length,
        target_length,
        num_channels,
        max_target_channels,
    ):
        self.generator = generator
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.history_length = history_length
        self.target_length = target_length
        self.num_channels = num_channels
        self.max_target_channels = max_target_channels

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.generator.generate_batch(
                batch_size=self.batch_size,
                history_length=self.history_length,
                target_length=self.target_length,
                num_channels=self.num_channels,
                max_target_channels=self.max_target_channels,
            )


class TrainingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.initial_epoch = 0

        logger.info("Initializing training pipeline...")
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
            **self.config["MultiStepModel"],
        ).to(self.device)

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Load checkpoint if continuing training
        self.config["model_save_name"] = generate_model_save_name(self.config)
        self._load_checkpoint()

        # Initialize the synthetic data generator
        self.generator = MultivariateTimeSeriesGenerator(
            global_seed=self.config["seed"],
            history_length=96,
            target_length=96,
            num_channels=160,
            max_target_channels=self.config["max_target_channels"],
        )

        # Generate fixed validation batch
        self.val_data = self.generator.generate_batch(
            batch_size=self.config["batch_size"],
            seed=self.config["seed"],
            history_length=96,
            target_length=96,
            num_channels=160,
            max_target_channels=self.config["max_target_channels"],
        )

        # Setup training data loader with synthetic data
        synthetic_dataset = SyntheticDataset(
            self.generator,
            batch_size=self.config["batch_size"],
            num_batches=self.config["num_training_iterations_per_epoch"],
            history_length=96,
            target_length=96,
            num_channels=160,
            max_target_channels=self.max_target_channels,
        )
        self.train_loader = DataLoader(synthetic_dataset, batch_size=None)

        # Setup loss function
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
                    name=self.config["model_save_name"],
                )
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}")
                self.config["wandb"] = False

    def _load_checkpoint(self) -> None:
        """Load model checkpoint if available and continuing training."""
        checkpoint_path = (
            f"{self.config['model_save_dir']}/{self.config['model_save_name']}.pth"
        )
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
            f"{self.config['model_save_dir']}/{self.config['model_save_name']}.pth",
        )
        logger.info(f"Checkpoint saved at epoch {epoch}")

    def _scale_target(
        self, target: torch.Tensor, output: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale target values based on configured scaler."""
        if self.config["scaler"] == "min_max":
            max_scale = output["scale_params"][0]  # [batch_size, 1, num_channels]
            min_scale = output["scale_params"][1]
            target_index = output["target_index"]  # [batch_size, 1]
            max_targets = torch.gather(max_scale, 2, target_index).expand(
                -1, target.shape[1], -1
            )
            min_targets = torch.gather(min_scale, 2, target_index).expand(
                -1, target.shape[1], -1
            )
            scaled_target = (target - min_targets) / (max_targets - min_targets)
            return scaled_target, max_targets, min_targets
        else:
            mean = output["scale_params"][0]  # [batch_size, 1, num_channels]
            std = output["scale_params"][1]
            target_index = output["target_index"]  # [batch_size, 1]
            mean_targets = torch.gather(mean, 2, target_index).expand(
                -1, target.shape[1], -1
            )
            std_targets = torch.gather(std, 2, target_index).expand(
                -1, target.shape[1], -1
            )
            scaled_target = (target - mean_targets) / std_targets
            return scaled_target, mean_targets, std_targets

    def _inverse_scale(self, output: torch.Tensor, scale_params: Tuple) -> torch.Tensor:
        """Inverse scale model predictions."""
        if self.config["scaler"] == "min_max":
            max_targets, min_targets = scale_params
            return (output * (max_targets - min_targets)) + min_targets
        else:
            mean_targets, std_targets = scale_params
            return (output * std_targets) + mean_targets

    def _prepare_batch(
        self, batch: BatchTimeSeriesContainer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for model input."""
        batch.to_device(self.device)
        return batch, batch.target_values

    def _plot_fixed_examples(self, epoch: int, avg_val_loss: float) -> None:
        """Plot fixed validation examples and log to WandB."""
        fixed_data = self.val_data
        with torch.no_grad():
            fixed_data.to_device(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                fixed_output = self.model(fixed_data, fixed_data.target_values.size(1))
                _, *scale_params = self._scale_target(
                    fixed_data.target_values, fixed_output
                )
                inv_scaled_fixed_output = self._inverse_scale(
                    fixed_output["result"], scale_params
                )

            for i in range(
                min(
                    self.config["num_val_samples_to_plot"],
                    fixed_data.history_values.shape[0],
                )
            ):
                pred_future = inv_scaled_fixed_output[i].cpu().numpy()
                fig = plot_from_container(
                    ts_data=fixed_data,
                    sample_idx=i,
                    predicted_values=pred_future,
                    title=f"Epoch {epoch} - Example {i + 1} (Val Loss: {avg_val_loss:.4f})",
                    output_file=None,
                    show=False,
                )
                if self.config["wandb"]:
                    wandb.log({f"val_plot_{i}": wandb.Image(fig)})
                plt.close(fig)
                logger.info(f"Plotted validation sample {i + 1} for epoch {epoch}")

    def _update_metrics(
        self, metrics: Dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update metric calculations."""
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        for metric in metrics.values():
            metric.update(predictions, targets)

    def _log_training_progress(
        self, epoch: int, batch_idx: int, running_loss: float
    ) -> None:
        """Log training progress."""
        avg_loss = running_loss / 10
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
                output = self.model(
                    data_container=data, training=True, drop_enc_allow=False
                )
                scaled_target, *scale_params = self._scale_target(target, output)
                loss = self.criterion(output["result"], scaled_target.half())

            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

            # Update metrics
            inv_scaled_output = self._inverse_scale(output["result"], scale_params)
            self._update_metrics(self.train_metrics, inv_scaled_output, target)

            running_loss += loss.item()
            epoch_loss += loss.item()

            if batch_idx % 10 == 9:
                self._log_training_progress(epoch, batch_idx, running_loss)
                running_loss = 0.0

        return epoch_loss / min(
            batch_idx + 1, self.config["num_training_iterations_per_epoch"]
        )

    def _validate_epoch(self, epoch: int) -> float:
        """Validate model on the fixed synthetic validation batch."""
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            data, target = self._prepare_batch(self.val_data)
            pred_len = target.size(1)
            with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                output = self.model(data, pred_len)
                scaled_target, *scale_params = self._scale_target(target, output)
                val_loss = self.criterion(output["result"], scaled_target.half()).item()
            total_val_loss = val_loss  # Single batch
            inv_scaled_output = self._inverse_scale(output["result"], scale_params)
            self._update_metrics(self.val_metrics, inv_scaled_output, target)

        avg_val_loss = total_val_loss  # Since it's one batch
        self._plot_fixed_examples(epoch, avg_val_loss)
        logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

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
        default="./configs/training/train.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    os.environ["WANDB_MODE"] = "online" if config["wandb"] else "offline"
    pipeline = TrainingPipeline(config)
    pipeline.train()
