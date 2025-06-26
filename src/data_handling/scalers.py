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

        Args:
            history_values: Historical observations.
                Shape: [batch_size, seq_len, num_channels]
            history_mask: Optional mask indicating valid entries (1.0 for valid, 0.0 for invalid).
                Shape: [batch_size, seq_len]
                If None, all values are considered valid.

        Returns:
            Dictionary containing scaling statistics with tensors of shape
            [batch_size, 1, num_channels] for broadcasting compatibility.
        """
        pass

    @abstractmethod
    def scale(
        self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply scaling transformation to data.

        Args:
            data: Data to scale.
                Shape: [batch_size, seq_len_or_pred_len, num_channels]
            statistics: Scaling statistics from compute_statistics.

        Returns:
            Scaled tensor with same shape as input.
        """
        pass

    @abstractmethod
    def inverse_scale(
        self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse scaling transformation to recover original scale.

        Args:
            scaled_data: Scaled data to transform back.
                Shape: [batch_size, seq_len_or_pred_len, num_channels]
            statistics: Scaling statistics from compute_statistics.

        Returns:
            Data in original scale with same shape as input.
        """
        pass


class RobustScaler(BaseScaler):
    """
    Robust scaler using median and IQR for normalization.

    Scaling formula: (data - median) / (iqr + epsilon)

    This scaler is robust to outliers by using median and interquartile range
    instead of mean and standard deviation.

    Args:
        epsilon: Small constant for numerical stability. Default: 1e-6 (increased for better stability)
        min_scale: Minimum scale value to prevent division by very small numbers. Default: 1e-3
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

        Args:
            history_values: Historical observations.
                Shape: [batch_size, seq_len, num_channels]
            history_mask: Optional mask indicating valid entries.
                Shape: [batch_size, seq_len]

        Returns:
            Dictionary with keys 'median' and 'iqr', each containing tensors
            of shape [batch_size, 1, num_channels].
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        # Initialize output tensors
        medians = torch.zeros(batch_size, 1, num_channels, device=device)
        iqrs = torch.ones(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                # Get data for this batch and channel
                channel_data = history_values[b, :, c]

                # Apply mask if provided
                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                # Skip if no valid data
                if len(valid_data) == 0:
                    continue  # Keep defaults: median=0, iqr=1

                # Remove NaN and inf values for more robust statistics
                valid_data = valid_data[torch.isfinite(valid_data)]

                if len(valid_data) == 0:
                    continue  # Keep defaults if no finite data

                # Compute median
                median_val = torch.median(valid_data)
                medians[b, 0, c] = median_val

                # Compute IQR (75th percentile - 25th percentile)
                if len(valid_data) > 1:
                    try:
                        q75 = torch.quantile(valid_data, 0.75)
                        q25 = torch.quantile(valid_data, 0.25)
                        iqr_val = q75 - q25

                        # Ensure IQR is not too small (use minimum scale)
                        iqr_val = torch.max(
                            iqr_val, torch.tensor(self.min_scale, device=device)
                        )
                        iqrs[b, 0, c] = iqr_val
                    except Exception:
                        # Fallback to standard deviation if quantile fails
                        std_val = torch.std(valid_data)
                        iqrs[b, 0, c] = torch.max(
                            std_val, torch.tensor(self.min_scale, device=device)
                        )
                # If only one data point, keep default iqr with min_scale
                else:
                    iqrs[b, 0, c] = self.min_scale

        return {"median": medians, "iqr": iqrs}

    def scale(
        self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply robust scaling: (data - median) / (iqr + epsilon).

        Args:
            data: Data to scale.
                Shape: [batch_size, seq_len_or_pred_len, num_channels]
            statistics: Dictionary containing 'median' and 'iqr' tensors.

        Returns:
            Scaled tensor with same shape as input.
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        # Ensure denominator is not too small
        denominator = torch.max(
            iqr + self.epsilon, torch.tensor(self.min_scale, device=iqr.device)
        )

        scaled_data = (data - median) / denominator

        # Clamp to reasonable range to prevent extreme values
        scaled_data = torch.clamp(scaled_data, -50.0, 50.0)

        return scaled_data

    def inverse_scale(
        self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse robust scaling: scaled_data * (iqr + epsilon) + median.

        Args:
            scaled_data: Scaled data to transform back.
                Shape: [batch_size, seq_len_or_pred_len, num_channels]
            statistics: Dictionary containing 'median' and 'iqr' tensors.

        Returns:
            Data in original scale with same shape as input.
        """
        median = statistics["median"]
        iqr = statistics["iqr"]

        # Ensure denominator consistency with scaling
        denominator = torch.max(
            iqr + self.epsilon, torch.tensor(self.min_scale, device=iqr.device)
        )

        return scaled_data * denominator + median


class MinMaxScaler(BaseScaler):
    """
    Min-Max scaler that normalizes data to the range [-1, 1].

    Scaling formula: (data - min) / (max - min + epsilon) * 2 - 1

    This scaler maps the data to a fixed range, which can be beneficial
    for neural networks and other algorithms that work well with bounded inputs.

    Args:
        epsilon: Small constant for numerical stability. Default: 1e-8
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

        Args:
            history_values: Historical observations.
                Shape: [batch_size, seq_len, num_channels]
            history_mask: Optional mask indicating valid entries.
                Shape: [batch_size, seq_len]

        Returns:
            Dictionary with keys 'min' and 'max', each containing tensors
            of shape [batch_size, 1, num_channels].
        """
        batch_size, seq_len, num_channels = history_values.shape
        device = history_values.device

        # Initialize output tensors with defaults
        mins = torch.zeros(batch_size, 1, num_channels, device=device)
        maxs = torch.ones(batch_size, 1, num_channels, device=device)

        for b in range(batch_size):
            for c in range(num_channels):
                # Get data for this batch and channel
                channel_data = history_values[b, :, c]

                # Apply mask if provided
                if history_mask is not None:
                    mask = history_mask[b, :].bool()
                    valid_data = channel_data[mask]
                else:
                    valid_data = channel_data

                # Skip if no valid data
                if len(valid_data) == 0:
                    continue  # Keep defaults: min=0, max=1

                # Compute min and max
                min_val = torch.min(valid_data)
                max_val = torch.max(valid_data)

                mins[b, 0, c] = min_val
                maxs[b, 0, c] = max_val

                # Handle case where min == max
                if torch.abs(max_val - min_val) < self.epsilon:
                    maxs[b, 0, c] = min_val + 1.0  # Set range to [min_val, min_val + 1]

        return {"min": mins, "max": maxs}

    def scale(
        self, data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply min-max scaling to range [-1, 1]: (data - min) / (max - min + epsilon) * 2 - 1.

        Args:
            data: Data to scale.
                Shape: [batch_size, seq_len_or_pred_len, num_channels]
            statistics: Dictionary containing 'min' and 'max' tensors.

        Returns:
            Scaled tensor with same shape as input, values in range [-1, 1].
        """
        min_val = statistics["min"]
        max_val = statistics["max"]

        # Scale to [0, 1] then to [-1, 1]
        normalized = (data - min_val) / (max_val - min_val + self.epsilon)
        return normalized * 2.0 - 1.0

    def inverse_scale(
        self, scaled_data: torch.Tensor, statistics: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse min-max scaling: (scaled_data + 1) / 2 * (max - min + epsilon) + min.

        Args:
            scaled_data: Scaled data to transform back (assumed to be in range [-1, 1]).
                Shape: [batch_size, seq_len_or_pred_len, num_channels]
            statistics: Dictionary containing 'min' and 'max' tensors.

        Returns:
            Data in original scale with same shape as input.
        """
        min_val = statistics["min"]
        max_val = statistics["max"]

        # Transform from [-1, 1] to [0, 1] then to original scale
        normalized = (scaled_data + 1.0) / 2.0
        return normalized * (max_val - min_val + self.epsilon) + min_val
