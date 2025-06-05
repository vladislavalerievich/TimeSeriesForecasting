from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import torch


class Frequency(Enum):
    M = "ME"  # Month End
    W = "W"  # Weekly
    D = "D"  # Daily
    H = "h"  # Hourly
    S = "s"  # Seconds
    T5 = "5min"  # 5 minutes
    T10 = "10min"  # 10 minutes
    T15 = "15min"  # 15 minutes


@dataclass
class StaticFeaturesDataContainer:
    """
    Holds computed numerical static features for a batch of time series.
    Each tensor should have shape [batch_size, num_channels].
    """

    # --- Core Statistics ---
    mean: torch.Tensor
    std: torch.Tensor
    median: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    # --- Temporal Statistics ---
    autocorr_lag1: torch.Tensor
    trend_slope: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate that all feature tensors have consistent shapes and types."""
        feature_list = [
            self.mean,
            self.std,
            self.median,
            self.min_val,
            self.max_val,
            self.autocorr_lag1,
        ]
        if self.trend_slope is not None:
            feature_list.append(self.trend_slope)

        batch_size, num_channels = feature_list[0].shape
        for i, feature in enumerate(feature_list):
            if not isinstance(feature, torch.Tensor):
                raise TypeError(
                    f"Feature at index {i} is not a torch.Tensor (got {type(feature)})"
                )
            if feature.shape != (batch_size, num_channels):
                raise ValueError(
                    f"Inconsistent shape at feature {i}: expected {(batch_size, num_channels)}, got {feature.shape}"
                )

    def get_feature_tensors(self) -> List[torch.Tensor]:
        """Returns a list of available (non-None) feature tensors."""
        feature_list = [
            self.mean,
            self.std,
            self.median,
            self.min_val,
            self.max_val,
            self.autocorr_lag1,
        ]
        if self.trend_slope is not None:
            feature_list.append(self.trend_slope)
        return feature_list

    def concatenate(self, flatten_channels: bool = True) -> Optional[torch.Tensor]:
        """
        Concatenates all numerical static features into a single tensor.

        Args:
            flatten_channels: If True, flattens the channel dimension, resulting
                              in shape [batch_size, num_channels * num_feature_types].
                              If False, keeps channel dimension separate, resulting
                              in shape [batch_size, num_channels, num_feature_types].

        Returns:
            Concatenated tensor [batch_size, num_channels * num_features] or [batch_size, num_channels, num_features].
        """
        feature_tensors = self.get_feature_tensors()
        if not feature_tensors:
            return None

        batch_size = feature_tensors[0].shape[0]
        reshaped_features = [f.unsqueeze(-1) for f in feature_tensors]
        concatenated = torch.cat(reshaped_features, dim=2)  # [B, C, num_features]

        if flatten_channels:
            return concatenated.view(batch_size, -1)
        else:
            return concatenated

    def to_device(self, device: torch.device) -> None:
        """
        Move all tensors in the StaticFeaturesDataContainer to the specified device in place.

        Args:
            device: The target device (e.g., 'cpu', 'cuda').

        Raises:
            TypeError: If any tensor attribute is not a torch.Tensor.
            RuntimeError: If device transfer fails for any tensor.
        """
        # Move required tensor attributes
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.median = self.median.to(device)
        self.min_val = self.min_val.to(device)
        self.max_val = self.max_val.to(device)
        self.autocorr_lag1 = self.autocorr_lag1.to(device)

        # Move optional tensor attribute if it exists
        if self.trend_slope is not None:
            self.trend_slope = self.trend_slope.to(device)


@dataclass
class BatchTimeSeriesContainer:
    """
    Container for a batch of multivariate time series data and their associated features.

    Attributes:
        history_values: Tensor of historical observations.
            Shape: [batch_size, seq_len, num_channels]
        target_values: Tensor of future observations to predict.
            Shape: [batch_size, pred_len]
        target_index: Tensor of target channel index.
            Shape: [batch_size]
        start: Timestamps of the first history value.
            Shape: [batch_size]
        frequency: Frequency of the time series.
            Type: Frequency enum (D=Daily, W=Weekly, H=Hourly, ME=Month End, S=Seconds)
        past_feat_dynamic_real: Optional time-varying covariates for history.
            Shape: [batch_size, seq_len, num_dynamic_features]
        future_feat_dynamic_real: Optional time-varying covariates for prediction horizon.
            Shape: [batch_size, pred_len, num_dynamic_features]
        static_features: Optional StaticFeaturesDataContainer of features constant over time.
            Shape: [batch_size, num_static_features, num_static_features_per_channel]
        history_mask: Optional boolean/float tensor indicating valid (1/True) vs padded (0/False)
            entries in history_values/history_time_features.
            Shape: [batch_size, seq_len]
        target_mask: Optional boolean/float tensor indicating valid (1/True) vs padded/missing (0/False)
            target values.
            Shape: [batch_size, pred_len]
    """

    history_values: torch.Tensor
    target_values: torch.Tensor
    target_index: torch.Tensor
    start: np.ndarray[np.datetime64]
    frequency: Frequency

    past_feat_dynamic_real: Optional[torch.Tensor] = None
    future_feat_dynamic_real: Optional[torch.Tensor] = None
    static_features: Optional[StaticFeaturesDataContainer] = None

    history_mask: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate all tensor shapes and consistency."""
        # --- Tensor Type Checks ---
        if not isinstance(self.history_values, torch.Tensor):
            raise TypeError("history_values must be a torch.Tensor")
        if not isinstance(self.target_values, torch.Tensor):
            raise TypeError("target_values must be a torch.Tensor")
        if self.target_index is not None and not isinstance(
            self.target_index, torch.Tensor
        ):
            raise TypeError("target_index must be a torch.Tensor or None")
        if not isinstance(self.start, np.ndarray):
            raise TypeError("start must be a np.ndarray")
        if not all(isinstance(s, np.datetime64) for s in self.start):
            raise TypeError("start must be a list of np.datetime64")
        if not isinstance(self.frequency, Frequency):
            raise TypeError("frequency must be a Frequency enum")

        batch_size, seq_len, num_channels = self.history_values.shape
        pred_len = self.target_values.shape[1]

        # --- Core Shape Checks ---
        if self.target_values.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between history and target_values")
        if self.target_index.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between history and target_index")

        # --- Static Features Check ---
        if self.static_features is not None:
            if not isinstance(self.static_features, StaticFeaturesDataContainer):
                raise TypeError(
                    f"static_features must be a StaticFeaturesDataContainer, got {type(self.static_features)}"
                )
            try:
                feature_tensors = self.static_features.get_feature_tensors()
                if feature_tensors:
                    if feature_tensors[0].shape[0] != batch_size:
                        raise ValueError("Batch size mismatch in static_features")
                    if feature_tensors[0].shape[1] != num_channels:
                        raise ValueError("Channel size mismatch in static_features")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid StaticFeaturesDataContainer: {e}") from e

        # --- Dynamic Features Check ---
        if self.past_feat_dynamic_real is not None:
            if not isinstance(self.past_feat_dynamic_real, torch.Tensor):
                raise TypeError("past_feat_dynamic_real must be a torch.Tensor or None")
            if self.past_feat_dynamic_real.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"past_feat_dynamic_real shape mismatch: expected [{batch_size}, {seq_len}, ...], got {self.past_feat_dynamic_real.shape}"
                )
        if self.future_feat_dynamic_real is not None:
            if not isinstance(self.future_feat_dynamic_real, torch.Tensor):
                raise TypeError(
                    "future_feat_dynamic_real must be a torch.Tensor or None"
                )
            if self.future_feat_dynamic_real.shape[:2] != (batch_size, pred_len):
                raise ValueError(
                    f"future_feat_dynamic_real shape mismatch: expected [{batch_size}, {pred_len}, ...], got {self.future_feat_dynamic_real.shape}"
                )

        # --- Optional Mask Checks ---
        if self.history_mask is not None:
            if not isinstance(self.history_mask, torch.Tensor):
                raise TypeError("history_mask must be a Tensor or None")
            if self.history_mask.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"Shape mismatch in history_mask: {self.history_mask.shape[:2]} vs {(batch_size, seq_len)}"
                )

        if self.target_mask is not None:
            if not isinstance(self.target_mask, torch.Tensor):
                raise TypeError("target_mask must be a Tensor or None")
            if not (
                self.target_mask.shape == (batch_size, pred_len)
                or self.target_mask.shape == self.target_values.shape
            ):
                raise ValueError(
                    f"Shape mismatch in target_mask: expected {(batch_size, pred_len)} or {self.target_values.shape}, got {self.target_mask.shape}"
                )

    def to_device(
        self, device: torch.device, attributes: Optional[List[str]] = None
    ) -> None:
        """
        Move specified tensors to the target device in place.

        Args:
            device: Target device (e.g., 'cpu', 'cuda').
            attributes: Optional list of attribute names to move. If None, move all tensors.

        Raises:
            ValueError: If an invalid attribute is specified or device transfer fails.
        """
        all_tensors = {
            "history_values": self.history_values,
            "target_values": self.target_values,
            "target_index": self.target_index,
            "history_mask": self.history_mask,
            "target_mask": self.target_mask,
            "past_feat_dynamic_real": self.past_feat_dynamic_real,
            "future_feat_dynamic_real": self.future_feat_dynamic_real,
        }

        if attributes is None:
            attributes = [k for k, v in all_tensors.items() if v is not None]
            if self.static_features is not None:
                self.static_features.to_device(device)

        for attr in attributes:
            if attr not in all_tensors:
                raise ValueError(f"Invalid attribute: {attr}")
            if all_tensors[attr] is not None:
                setattr(self, attr, all_tensors[attr].to(device))

    @property
    def batch_size(self) -> int:
        return self.history_values.shape[0]

    @property
    def history_length(self) -> int:
        return self.history_values.shape[1]

    @property
    def target_length(self) -> int:
        return self.target_values.shape[1]

    @property
    def num_channels(self) -> int:
        return self.history_values.shape[2]
