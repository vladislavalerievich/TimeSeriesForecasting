import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from src.models.models import MultiStepModel
from src.synthetic_generation.sine_wave import generate_sine_batch, train_val_loader
from src.utils.utils import (
    SMAPEMetric,
    avoid_constant_inputs,
    device,
    generate_model_save_name,
    seed_everything,
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
        self.val_loader = None
        self.initial_epoch = 0
        self.fixed_val_data = generate_sine_batch(
            batch_size=5,
            seq_len=256,
            pred_len=128,
            sine_config=None,
        )

        self._setup()

    def _setup(self) -> None:
        """Setup model, optimizer, scheduler, and data loaders."""
        seed_everything(self.config["seed"])

        # Determine CPU usage
        available_cpus = 1 if self.config["debugging"] else os.cpu_count()

        # Initialize model
        self.model = MultiStepModel(
            **self.config["BaseModelConfig"],
            **self.config["MultiStepModel"],
            scaler=self.config["scaler"],
        ).to(self.device)

        # Load checkpoint if continuing training
        self.config["model_save_name"] = generate_model_save_name(self.config)
        self._load_checkpoint()

        # Load data
        self.train_loader, self.val_loader = train_val_loader(
            config=self.config, cpus_available=available_cpus
        )

        # Setup optimizer and scheduler
        self._setup_optimizer()

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
            "smape": SMAPEMetric().to(self.device),
        }
        self.val_metrics = {
            "mape": torchmetrics.MeanAbsolutePercentageError().to(self.device),
            "mse": torchmetrics.MeanSquaredError().to(self.device),
            "smape": SMAPEMetric().to(self.device),
        }

    def _setup_wandb(self):
        # Initialize wandb if enabled
        if self.config["wandb"]:
            self.run = wandb.init(
                project="Sine Wave Experiments",
                config=self.config,
                name=self.config["model_save_name"],
            )

    def _load_checkpoint(self) -> None:
        """Load model checkpoint if available and continuing training."""
        checkpoint_path = (
            f"{self.config['model_save_dir']}/{self.config['model_save_name']}.pth"
        )
        if self.config["continue_training"] and os.path.exists(checkpoint_path):
            print(
                f"Loading previous training states from: {self.config['model_save_name']}"
            )
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.config["lr_scheduler"] == "cosine":
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.initial_epoch = ckpt["epoch"]
        else:
            print("No previous training states found, starting fresh")

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

    def _scale_target(
        self, target: torch.Tensor, output: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale target values based on configured scaler."""
        if self.config["scaler"] == "min_max":
            max_scale = output["scale"][0].squeeze(-1)
            min_scale = output["scale"][1].squeeze(-1)
            scaled_target = (target - min_scale) / (max_scale - min_scale)
            return scaled_target, max_scale, min_scale
        else:
            mean = output["scale"][0].squeeze(-1)
            std = output["scale"][1].squeeze(-1)
            scaled_target = (target - mean) / std
            return scaled_target, mean, std

    def _inverse_scale(self, output: torch.Tensor, scale_params: Tuple) -> torch.Tensor:
        """Inverse scale model predictions."""
        if self.config["scaler"] == "min_max":
            max_scale, min_scale = scale_params
            return (output * (max_scale - min_scale)) + min_scale
        else:
            mean, std = scale_params
            return (output * std) + mean

    def _plot_fixed_examples(self, epoch: int, avg_val_loss: float) -> None:
        """Plot fixed validation examples and log to WandB."""
        fixed_data, fixed_target = self._prepare_batch(self.fixed_val_data)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                fixed_output = self.model(fixed_data, fixed_target.size(1))
                _, *scale_params = self._scale_target(fixed_target, fixed_output)
                inv_scaled_fixed_output = self._inverse_scale(
                    fixed_output["result"], scale_params
                )

            # Generate plots for each example in the fixed batch
            for i in range(self.fixed_val_data["history"].shape[0]):
                history = fixed_data["history"][i].cpu().numpy()
                true_future = fixed_target[i].cpu().numpy()
                pred_future = inv_scaled_fixed_output[i].cpu().numpy()

                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 5))
                plt.plot(range(len(history)), history, label="History", color="blue")
                plt.plot(
                    range(len(history), len(history) + len(true_future)),
                    true_future,
                    label="True Future",
                    color="green",
                )
                plt.plot(
                    range(len(history), len(history) + len(pred_future)),
                    pred_future,
                    label="Predicted Future",
                    color="red",
                )
                plt.title(
                    f"Epoch {epoch} - Example {i + 1} (Val Loss: {avg_val_loss:.4f})"
                )
                plt.legend()
                wandb.log({f"val_plot_{i}": wandb.Image(plt)})
                plt.close()

    def _prepare_batch(self, batch: Dict) -> Tuple[Dict, torch.Tensor]:
        """Prepare batch data for processing."""
        data = {k: v.to(self.device) for k, v in batch.items() if k != "target_values"}
        target = batch["target_values"].to(self.device)
        avoid_constant_inputs(data["history"], target)
        return data, target

    def _update_metrics(
        self, metrics: Dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Update metric calculations."""
        for metric in metrics.values():
            metric.update(predictions, targets)

    def _log_training_progress(
        self, epoch: int, batch_idx: int, running_loss: float
    ) -> None:
        """Log training progress."""
        avg_loss = running_loss / 10
        print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Sc. Loss: {avg_loss}")
        if self.config["wandb"]:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/mape": self.train_metrics["mape"].compute(),
                    "train/mse": self.train_metrics["mse"].compute(),
                    "train/smape": self.train_metrics["smape"].compute(),
                    "epoch": epoch,
                    "step": epoch * self.config["training_rounds"] + batch_idx,
                }
            )

    def _maybe_sample_prediction(
        self, target: torch.Tensor, data: Dict
    ) -> Tuple[int, torch.Tensor, Dict]:
        """Randomly sample prediction length if configured."""
        pred_len = target.size(1)
        if self.config["sample_multi_pred"] > np.random.rand():
            pred_limits = np.random.randint(4, pred_len + 1, 2)
            start_pred, end_pred = min(pred_limits), max(pred_limits)
            if end_pred == start_pred:
                start_pred, end_pred = (
                    (start_pred - 1, end_pred)
                    if start_pred == pred_len
                    else (start_pred, end_pred + 1)
                )
            pred_len = end_pred - start_pred
            target = target[:, start_pred:end_pred].contiguous()
            for key in ["target_dates", "complete_target", "task"]:
                data[key] = data[key][:, start_pred:end_pred].contiguous()
        return pred_len, target, data

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= self.config["training_rounds"]:
                break

            data, target = self._prepare_batch(batch)
            pred_len, target, data = self._maybe_sample_prediction(target, data)

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=True):
                output = self.model(
                    data,
                    pred_len,
                    training=True,
                    drop_enc_allow=(
                        self.config["sample_multi_pred"] <= np.random.rand()
                    ),
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

        return epoch_loss / min(batch_idx + 1, self.config["training_rounds"])

    def _validate_epoch(self, epoch) -> float:
        """Validate model for one epoch."""
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= self.config["validation_rounds"]:
                    break

                data, target = self._prepare_batch(batch)
                pred_len = target.size(1)

                with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
                    output = self.model(data, pred_len)
                    scaled_target, *scale_params = self._scale_target(target, output)
                    val_loss = self.criterion(
                        output["result"], scaled_target.half()
                    ).item()

                total_val_loss += val_loss
                inv_scaled_output = self._inverse_scale(output["result"], scale_params)
                self._update_metrics(self.val_metrics, inv_scaled_output, target)

        avg_val_loss = total_val_loss / self.config["validation_rounds"]
        self._plot_fixed_examples(epoch, avg_val_loss)
        return avg_val_loss

    def train(self) -> None:
        """Execute the training pipeline."""
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

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch + 1}, Time: {epoch_time / 60:.2f} mins")

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
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    os.environ["WANDB_MODE"] = "online" if config["wandb"] else "offline"
    pipeline = TrainingPipeline(config)
    pipeline.train()

    if config["wandb"]:
        wandb.finish()

    print("Training completed successfully.")
