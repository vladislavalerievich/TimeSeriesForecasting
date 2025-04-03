import argparse
import os
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from src.models.models import MultiStepModel
from src.synthetic_generation.sine_wave import train_val_loader
from src.utils.utils import (
    SMAPEMetric,
    avoid_constant_inputs,
    device,
    generate_model_save_name,
    seed_everything,
)

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def train_model(config):
    seed_everything(config["seed"])

    print("config:")
    print(pprint.pformat(config))

    print(
        f"cuda device usage (before model load): {torch.cuda.memory_allocated() / 2**20}"
    )

    # loading the datasets as dataloaders
    if config["debugging"]:
        available_cpus = 1
    else:
        available_cpus = os.cpu_count()

    # Load the model
    model = MultiStepModel(
        **config["BaseModelConfig"], **config["MultiStepModel"], scaler=config["scaler"]
    ).to(device)

    # Assuming your train_loader and test_loader are already defined
    if config["lr_scheduler"] == "cosine":
        optimizer = optim.AdamW(model.parameters(), lr=config["initial_lr"])
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"], eta_min=config["learning_rate"]
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    initial_epoch = 0
    # Load state dicts if we are resuming training
    config["model_save_name"] = generate_model_save_name(config)
    if config["continue_training"] and os.path.exists(
        f"{config['model_save_dir']}/{config['model_save_name']}.pth"
    ):
        print(f"loading previous training states from: {config['model_save_name']}")
        ckpt = torch.load(
            f"{config['model_save_dir']}/{config['model_save_name']}.pth",
            map_location=device,
        )
        # load states
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if config["lr_scheduler"] == "cosine":
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        initial_epoch = ckpt["epoch"]
    else:
        print("no previous training states found, starting fresh")
        model = model.to(device)

    train_dataloader, val_dataloader = train_val_loader(
        config=config,
        cpus_available=available_cpus,
    )

    print(
        f"cuda device usage (after model load): {torch.cuda.memory_allocated() / 2**20}"
    )
    config["model_param_size"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Model size: {config['model_param_size']}")
    # wandb hyperparam init
    if config["wandb"]:
        run = wandb.init(
            project="Time Series Forecasting",
            config=config,
            name=config["model_save_name"],
        )

    if config["loss"] == "mae":
        criterion = nn.L1Loss().to(device)
    elif config["loss"] == "mse":
        criterion = nn.MSELoss().to(device)
    else:
        raise ValueError("Loss function not supported")

    # Metric initialization
    train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    train_mse = torchmetrics.MeanSquaredError().to(device)
    train_smape = SMAPEMetric().to(device)

    val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_mse = torchmetrics.MeanSquaredError().to(device)
    val_smape = SMAPEMetric().to(device)

    # Training Loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        print("============Training==================")
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_epoch_loss = 0.0
        batch_idx = 0
        for batch_id, batch in enumerate(train_dataloader):
            data, target = (
                {k: v.to(device) for k, v in batch.items() if k != "target_values"},
                batch["target_values"].to(device),
            )
            avoid_constant_inputs(data["history"], target)
            pred_len = target.size(1)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=True):
                drop_enc_allow = True
                if config["sample_multi_pred"] > np.random.rand():
                    drop_enc_allow = False
                    # randomly sample 2 numbers between 4 and pred_length (inclusive)
                    pred_limits = np.random.randint(4, pred_len + 1, 2)
                    start_pred = min(pred_limits)
                    end_pred = max(pred_limits)

                    if end_pred == start_pred:
                        if start_pred == pred_len:
                            start_pred = start_pred - 1
                        else:
                            end_pred = end_pred + 1
                    pred_len = end_pred - start_pred
                    target = target[:, start_pred:end_pred].contiguous()
                    data["target_dates"] = data["target_dates"][
                        :, start_pred:end_pred
                    ].contiguous()
                    data["complete_target"] = data["complete_target"][
                        :, start_pred:end_pred
                    ].contiguous()
                    data["task"] = data["task"][:, start_pred:end_pred].contiguous()
                output = model(
                    data, pred_len, training=True, drop_enc_allow=drop_enc_allow
                )

                print(f"Model output (batch {batch_idx}): {output['result'].shape}")

                if config["scaler"] == "min_max":
                    max_scale = output["scale"][0].squeeze(-1)
                    min_scale = output["scale"][1].squeeze(-1)
                    scaled_target = (target - min_scale) / (max_scale - min_scale)
                else:
                    scaled_target = (target - output["scale"][0].squeeze(-1)) / output[
                        "scale"
                    ][1].squeeze(-1)

                loss = criterion(output["result"], scaled_target.half())

            # Backward pass should be outside of autocast
            loss.backward()

            optimizer.step()
            torch.cuda.empty_cache()

            print(f"Train Loss for batch {batch_idx}: {loss.item()}")
            print("-" * 80)

            if config["scaler"] == "min_max":
                inv_scaled_output = (
                    output["result"] * (max_scale - min_scale)
                ) + min_scale
            else:
                inv_scaled_output = (
                    output["result"] * output["scale"][1].squeeze(-1)
                ) + output["scale"][0].squeeze(-1)

            # Update metrics
            train_mape.update(inv_scaled_output, target)
            train_mse.update(inv_scaled_output, target)
            train_smape.update(inv_scaled_output, target)
            running_loss += loss.item()
            train_epoch_loss += loss.item()

            if batch_idx == config["training_rounds"] - 1:
                train_epoch_loss = running_loss / (batch_idx % 10 + 1)

            if batch_idx % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                print(
                    f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Sc. Loss: {avg_loss} From torchmetric: {train_mse.compute()} From criterion: {loss.item()}"
                )
                if config["wandb"]:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/mape": train_mape.compute(),
                            "train/mse": train_mse.compute(),
                            "train/smape": train_smape.compute(),
                            "epoch": epoch,
                            "step": epoch * config["training_rounds"] + batch_idx,
                        }
                    )
                running_loss = 0.0

            batch_idx += 1
            # end of epoch at max training rounds
            if batch_idx == config["training_rounds"]:
                break

        if config["wandb"]:
            wandb.log(
                {"learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch}
            )

        with torch.autocast(device_type="cuda", dtype=torch.half, enabled=True):
            # Validation loop (after each epoch)
            print("============Validation==================")
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                val_batch_idx = 0
                for batch_id, batch in enumerate(val_dataloader):
                    data, target = (
                        {
                            k: v.to(device)
                            for k, v in batch.items()
                            if k != "target_values"
                        },
                        batch["target_values"].to(device),
                    )
                    avoid_constant_inputs(data["history"], target)
                    pred_len = target.size(1)

                    output = model(data, pred_len)

                    if config["scaler"] == "min_max":
                        max_scale = output["scale"][0].squeeze(-1)
                        min_scale = output["scale"][1].squeeze(-1)
                        scaled_target = (target - min_scale) / (max_scale - min_scale)
                    else:
                        scaled_target = (
                            target - output["scale"][0].squeeze(-1)
                        ) / output["scale"][1].squeeze(-1)

                    val_loss = criterion(output["result"], scaled_target.half()).item()
                    total_val_loss += val_loss

                    if batch_id % 10 == 9:
                        print(f"val loss for batch {batch_id}: {val_loss}")

                    if config["scaler"] == "min_max":
                        inv_scaled_output = (
                            output["result"] * (max_scale - min_scale)
                        ) + min_scale
                    else:
                        inv_scaled_output = (
                            output["result"] * output["scale"][1].squeeze(-1)
                        ) + output["scale"][0].squeeze(-1)
                    # Update validation metrics
                    val_mape.update(inv_scaled_output, target)
                    val_mse.update(inv_scaled_output, target)
                    val_smape.update(inv_scaled_output, target)

                    if val_batch_idx == config["validation_rounds"] - 1:
                        break

        # Compute and log validation metrics
        avg_val_loss = total_val_loss / config["validation_rounds"]
        print(
            f"Epoch: {epoch + 1}, Sc. Validation Loss: {avg_val_loss} From torchmetric: {val_mse.compute()}"
        )
        if config["wandb"]:
            wandb.log(
                {
                    "epoch_metrics": {
                        "train": {
                            "sc_loss": train_epoch_loss,
                            "mape": train_mape.compute(),
                            "smape": train_smape.compute(),
                        },
                        "val": {
                            "sc_loss": avg_val_loss,
                            "mape": val_mape.compute(),
                            "smape": val_smape.compute(),
                        },
                    },
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        epoch_time = time.time() - epoch_start_time
        print(f"Time taken for epoch: {epoch_time / 60} mins {epoch_time % 60} secs.")

        # Reset metrics for the next epoch
        train_mape.reset()
        train_mse.reset()
        train_smape.reset()
        val_mape.reset()
        val_mse.reset()
        val_smape.reset()

        if config["lr_scheduler"].startswith("cosine"):
            if (scheduler.get_last_lr()[0] == config["learning_rate"]) & (
                config["lr_scheduler"] == "cosine"
            ):
                print("Learning rate has reached the minimum value. No more steps.")
            else:
                scheduler.step()

        if epoch % 5 == 4:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                    if config["lr_scheduler"] == "cosine"
                    else None
                ),
                "epoch": epoch,
            }
            torch.save(
                ckpt, f"{config['model_save_dir']}/{config['model_save_name']}.pth"
            )

    if config["wandb"]:
        wandb.finish()

    # Save the final model
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if config["lr_scheduler"] == "cosine" else None
        ),
        "epoch": epoch,
    }
    torch.save(ckpt, f"{config['model_save_dir']}/{config['model_save_name']}.pth")


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

    # for wandb offline mode (can comment if needed):
    os.environ["WANDB_MODE"] = "online" if config["wandb"] else "offline"
    train_model(config)
