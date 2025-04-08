import os
import random

import numpy as np
import torch
import torchmetrics
from torch import nn

from src.data_handling.scalers import (
    custom_scaler_robust,
    identity_scaler,
    min_max_scaler,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_model_save_name(config):
    return (
        f"{config['model_save_name_prefix']}_"
        f"batch_size_{config['batch_size']}_"
        f"num_epochs_{config['num_epochs']}_"
        f"initial_lr{config['initial_lr']}_"
        f"learning_rate_{config['learning_rate']}_"
        f"context_len{config['context_len']}_"
        f"min_seq_len{config['min_seq_len']}_"
        f"max_seq_len{config['max_seq_len']}_"
        f"pred_len{config['pred_len']}_"
        f"pred_len_min{config['pred_len_min']}_"
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


def avoid_constant_inputs(inputs, outputs):
    idx_const_in = torch.nonzero(
        torch.all(inputs == inputs[:, 0].unsqueeze(1), dim=1)
    ).squeeze(1)
    if idx_const_in.size(0) > 0:
        inputs[idx_const_in, 0] += np.random.uniform(0.1, 1)


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
