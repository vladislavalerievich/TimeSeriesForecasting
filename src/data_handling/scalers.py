from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class BaseScaler(ABC):
    """
    Abstract base class for time series scalers.

    Defines the interface for scaling multivariate time series data with support
    for masked values and channel-wise scaling.
    """

    @abstractmethod
    def compute_statistics(
            self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scaling statistics from historical data.
        """
        pass

    @abstractmethod
    def scale(
            self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply scaling transformation to data.
        """
        pass

    @abstractmethod
    def inverse_scale(
            self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse scaling transformation to recover original scale.
        """
        pass


class RobustScaler(BaseScaler):
    """
    Robust scaler using median and IQR for normalization.
    """

    def __init__(self, epsilon: float = 1e-6, min_scale: float = 1e-3):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.epsilon = epsilon
        self.min_scale = min_scale

    def compute_statistics(
            self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute median and IQR statistics from historical data with improved numerical stability.
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        medians = torch.zeros(batch_size, 1, num_channels, device=device)
        iqrs = torch.ones(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                if len(valid_data) == 0:
                    continue

                valid_data = valid_data[torch.isfinite(valid_data)]

                if len(valid_data) == 0:
                    continue

                median_val = torch.median(valid_data)
                medians[b, 0, c] = median_val

                if len(valid_data) > 1:
                    try:
                        q75 = torch.quantile(valid_data, 0.75)
                        q25 = torch.quantile(valid_data, 0.25)
                        iqr_val = q75 - q25
                        iqr_val = torch.max(
                            iqr_val, torch.tensor(self.min_scale, device=device)
                        )
                        iqrs[b, 0, c] = iqr_val
                    except Exception:
                        std_val = torch.std(valid_data)
                        iqrs[b, 0, c] = torch.max(
                            std_val, torch.tensor(self.min_scale, device=device)
                        )
                else:
                    iqrs[b, 0, c] = self.min_scale

        return {"median": medians, "iqr": iqrs}

    def scale(
            self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply robust scaling: (data - median) / (iqr + epsilon).
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        denominator = torch.max(
            iqr + self.epsilon, torch.tensor(self.min_scale, device=iqr.device)
        )
        scaled_data = (data - median) / denominator
        scaled_data = torch.clamp(scaled_data, -50.0, 50.0)

        return scaled_data

    def inverse_scale(
            self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse robust scaling, now compatible with 3D or 4D tensors.
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        denominator = torch.max(
            iqr + self.epsilon, torch.tensor(self.min_scale, device=iqr.device)
        )

        # --- MODIFICATION START ---
        # If the input is 4D (includes quantile dimension), reshape stats for broadcasting.
        if scaled_data.ndim == 4:
            denominator = denominator.unsqueeze(-1)
            median = median.unsqueeze(-1)
        # --- MODIFICATION END ---

        return scaled_data * denominator + median


class MinMaxScaler(BaseScaler):
    """
    Min-Max scaler that normalizes data to the range [-1, 1].
    """

    def __init__(self, epsilon: float = 1e-8):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon

    def compute_statistics(
            self, history_values: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute min and max statistics from historical data.
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        mins = torch.zeros(batch_size, 1, num_channels, device=device)
        maxs = torch.ones(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                channel_data = history_values[b, :, c]

                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                if len(valid_data) == 0:
                    continue

                min_val = torch.min(valid_data)
                max_val = torch.max(valid_data)

                mins[b, 0, c] = min_val
                maxs[b, 0, c] = max_val

                if torch.abs(max_val - min_val) < self.epsilon:
                    maxs[b, 0, c] = min_val + 1.0

        return {"min": mins, "max": maxs}

    def scale(
            self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply min-max scaling to range [-1, 1].
        """
        min_val = statistics["min"]
        max_val = statistics["max"]

        normalized = (data - min_val) / (max_val - min_val + self.epsilon)
        return normalized * 2.0 - 1.0

    def inverse_scale(
            self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse min-max scaling, now compatible with 3D or 4D tensors.
        """
        min_val = statistics["min"]
        max_val = statistics["max"]

        # --- MODIFICATION START ---
        # If the input is 4D (includes quantile dimension), reshape stats for broadcasting.
        if scaled_data.ndim == 4:
            min_val = min_val.unsqueeze(-1)
            max_val = max_val.unsqueeze(-1)
        # --- MODIFICATION END ---

        normalized = (scaled_data + 1.0) / 2.0
        return normalized * (max_val - min_val + self.epsilon) + min_val