import os
import random

import numpy as np
import torch
import torchmetrics
from torch import nn

from src.data.scalers import custom_scaler_robust, identity_scaler, min_max_scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def generate_model_save_name(config):
    """
    Generate a model save name based on the configuration.
    Args:
        config: Configuration object containing model parameters.
    Returns:
        str: Generated model save name.
    """
    return (
        f"{config.model.name}_"
        f"bs{config.train.batch_size}_"
        f"lr{config.train.learning_rate}_"
        f"ep{config.train.epochs}_"
        f"seed{config.train.seed}"
    )


def position_encoding(periods: int, freqs: int):
    return np.hstack(
        [
            np.fromfunction(
                lambda i, j: np.sin(np.pi / periods * (2**j) * (i - 1)),
                (periods + 1, freqs),
            ),
            np.fromfunction(
                lambda i, j: np.cos(np.pi / periods * (2**j) * (i - 1)),
                (periods + 1, freqs),
            ),
        ]
    )


class CustomScaling(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == "custom_robust":
            self.scaler = custom_scaler_robust
        elif name == "min_max":
            self.scaler = min_max_scaler
        else:
            self.scaler = identity_scaler

    def forward(self, history_channels, epsilon):
        return self.scaler(history_channels, epsilon)


class PositionExpansion(nn.Module):
    def __init__(self, periods: int, freqs: int):
        super().__init__()
        # Channels could be ceiling(log_2(periods))
        self.periods = periods
        self.channels = freqs * 2
        self.embedding = torch.tensor(position_encoding(periods, freqs), device=device)

    def forward(self, tc: torch.Tensor):
        flat = tc.view(1, -1)
        embedded = self.embedding.index_select(0, flat.flatten().to(torch.long))
        out_shape = tc.shape
        return embedded.view(out_shape[0], out_shape[1], self.channels)


class SMAPEMetric(torchmetrics.Metric):
    def __init__(self, eps=1e-7):
        super().__init__(dist_sync_on_step=False)
        self.eps = eps
        self.add_state("total_smape", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and true labels.
        Args:
            preds (torch.Tensor): The predictions.
            target (torch.Tensor): Ground truth values.
        """
        preds = preds.float()
        target = target.float()
        diff = torch.abs((target - preds) / torch.clamp(target + preds, min=self.eps))
        smape = 200.0 * torch.mean(diff)  # Compute SMAPE for current batch
        # Multiply by batch size to prepare for mean
        self.total_smape += smape * target.numel()
        self.total_count += target.numel()

    def compute(self):
        """
        Computes the mean of the accumulated SMAPE values.
        """
        return self.total_smape / self.total_count
