from dataclasses import dataclass

import torch


@dataclass
class TimeSeriesData:
    history_ts: torch.Tensor  # [batch_size, seq_len, num_ts_features]
    history_values: torch.Tensor  # [batch_size, seq_len, num_features]
    target_ts: torch.Tensor  # [batch_size, pred_len, num_ts_features]
    target_values: torch.Tensor  # [batch_size, pred_len, num_targets]
    task: torch.Tensor  # [batch_size]
