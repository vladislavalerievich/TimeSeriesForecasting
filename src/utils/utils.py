import os
import random

import numpy as np
import torch
import torchmetrics

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
    )


def avoid_constant_inputs(inputs, outputs):
    idx_const_in = torch.nonzero(
        torch.all(inputs == inputs[:, 0].unsqueeze(1), dim=1)
    ).squeeze(1)
    if idx_const_in.size(0) > 0:
        inputs[idx_const_in, 0] += np.random.uniform(0.1, 1)


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
