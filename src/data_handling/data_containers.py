from dataclasses import dataclass
from typing import List, Optional

import torch


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
class TimeSeriesDataContainer:
    """
    Container for a batch of multivariate time series data and their associated features.

    Attributes:
        history_values: Tensor of historical observations.
            Shape: [batch_size, seq_len, num_channels]
        target_values: Tensor of future observations to predict.
            Shape: [batch_size, pred_len, num_targets]
        target_channels_indices: Tensor mapping target_values columns back to
            the original channel indices in history_values.
            Shape: [batch_size, num_targets]
        history_time_features: Tensor of time-derived features for the history timestamps window.
            Shape: [batch_size, seq_len, num_time_features]
        target_time_features: Tensor of time-derived features for the prediction timestamps window.
            Shape: [batch_size, pred_len, num_time_features]
        static_features: Optional StaticFeaturesDataContainer of features constant over time.
            Shape: [batch_size, num_static_features, num_static_features_per_channel]
        history_mask: Optional boolean/float tensor indicating valid (1/True) vs padded (0/False)
            entries in history_values/history_time_features.
            Shape: [batch_size, seq_len]
        target_mask: Optional boolean/float tensor indicating valid (1/True) vs padded/missing (0/False)
            target values.
            Shape: [batch_size, pred_len] or [batch_size, pred_len, num_targets]
    """

    history_values: torch.Tensor
    target_values: torch.Tensor
    target_channels_indices: torch.Tensor

    history_time_features: Optional[torch.Tensor] = None
    target_time_features: Optional[torch.Tensor] = None

    static_features: Optional[StaticFeaturesDataContainer] = None

    history_mask: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate all tensor shapes and consistency."""
        if not isinstance(self.history_values, torch.Tensor):
            raise TypeError("history_values must be a Tensor")
        if not isinstance(self.target_values, torch.Tensor):
            raise TypeError("target_values must be a Tensor")
        if not isinstance(self.target_channels_indices, torch.Tensor):
            raise TypeError("target_channels_indices must be a Tensor")

        batch_size, seq_len, num_channels = self.history_values.shape
        pred_len = self.target_values.shape[1]

        # --- Core Shape Checks ---
        if self.target_values.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between history and target_values")
        if self.target_channels_indices.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between history and target_channels_indices"
            )
        if self.target_values.shape[2] != self.target_channels_indices.shape[1]:
            raise ValueError(
                f"Number of target features mismatch: target_values {self.target_values.shape[2]} vs indices {self.target_channels_indices.shape[1]}"
            )

        # --- Optional Time Features ---
        if self.history_time_features is not None:
            if not isinstance(self.history_time_features, torch.Tensor):
                raise TypeError("history_time_features must be a Tensor or None")
            if self.history_time_features.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"Shape mismatch in history_time_features: {self.history_time_features.shape[:2]} vs {(batch_size, seq_len)}"
                )

        if self.target_time_features is not None:
            if not isinstance(self.target_time_features, torch.Tensor):
                raise TypeError("target_time_features must be a Tensor or None")
            if self.target_time_features.shape[:2] != (batch_size, pred_len):
                raise ValueError(
                    f"Shape mismatch in target_time_features: {self.target_time_features.shape[:2]} vs {(batch_size, pred_len)}"
                )

            if self.history_time_features is not None:
                if (
                    self.history_time_features.shape[2]
                    != self.target_time_features.shape[2]
                ):
                    raise ValueError(
                        "Mismatch in num_time_features between history and target_time_features"
                    )

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

    def to_device(self, device: torch.device) -> None:
        """
        Move all tensors in the TimeSeriesDataContainer to the specified device in place.

        Args:
            device: The target device (e.g., 'cpu', 'cuda').

        Raises:
            TypeError: If any tensor attribute is not a torch.Tensor or if static_features
                is not a StaticFeaturesDataContainer.
            RuntimeError: If device transfer fails for any tensor.
        """
        # Move required tensor attributes
        self.history_values = self.history_values.to(device)
        self.target_values = self.target_values.to(device)
        self.target_channels_indices = self.target_channels_indices.to(device)
        self.history_time_features = self.history_time_features.to(device)
        self.target_time_features = self.target_time_features.to(device)

        # Move optional tensor attributes if they exist
        if self.static_features is not None:
            self.static_features.to_device(device)
        if self.history_mask is not None:
            self.history_mask = self.history_mask.to(device)
        if self.target_mask is not None:
            self.target_mask = self.target_mask.to(device)
