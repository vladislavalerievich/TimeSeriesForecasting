from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class StaticFeaturesDataContainer:
    """
    Holds computed numerical static features for a batch of time series.

    Assumes features are computed per channel unless specified otherwise.
    All attributes that are not None are expected to be torch.Tensors
    with shape [batch_size, num_channels].
    """

    # --- Core Statistics ---
    # Shape: [batch_size, num_channels] for each
    mean: torch.Tensor
    std: torch.Tensor
    median: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    # --- Temporal Statistics ---
    # Shape: [batch_size, num_channels]
    autocorr_lag1: torch.Tensor
    # Shape: [batch_size, num_channels] - Optional
    trend_slope: Optional[torch.Tensor] = None

    def get_feature_tensors(self) -> List[torch.Tensor]:
        """
        Returns a list of the contained, non-None numerical feature tensors.
        Performs basic validation checks.
        """
        feature_list = [
            self.mean,
            self.std,
            self.median,
            self.min_val,
            self.max_val,
            self.autocorr_lag1,
        ]
        if self.trend_slope is not None:
            # Ensure trend_slope is also a tensor before adding
            if isinstance(self.trend_slope, torch.Tensor):
                feature_list.append(self.trend_slope)
            else:
                raise TypeError(
                    f"trend_slope is not a torch.Tensor (got {type(self.trend_slope)})"
                )

        # --- Validation within the getter ---
        if not feature_list:
            return []  # Return empty list if no features are present (shouldn't happen with mandatory fields)

        expected_shape = None
        for i, feature in enumerate(feature_list):
            if not isinstance(feature, torch.Tensor):
                raise TypeError(
                    f"Feature at index {i} is not a torch.Tensor (got {type(feature)})"
                )
            if i == 0:
                expected_shape = feature.shape
                # Expect [batch_size, num_channels]
                if len(expected_shape) != 2:
                    raise ValueError(
                        f"Expected features to have shape [batch_size, num_channels], but got {expected_shape} for feature at index 0 ({type(feature).__name__})"
                    )
            elif feature.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent shape at index {i} ({type(feature).__name__}): expected {expected_shape}, got {feature.shape}"
                )
        return feature_list

    def concatenate(self, flatten_channels: bool = True) -> Optional[torch.Tensor]:
        """
        Concatenates all available numerical static features into a single tensor.

        Args:
            flatten_channels: If True, flattens the channel dimension, resulting
                              in shape [batch_size, num_channels * num_feature_types].
                              If False, keeps channel dimension separate, resulting
                              in shape [batch_size, num_channels, num_feature_types].

        Returns:
            A single tensor containing the concatenated numerical features, or None
            if no features were present.
        """
        try:
            feature_tensors = self.get_feature_tensors()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Validation failed before concatenation: {e}") from e

        if not feature_tensors:
            # This case might indicate an issue as some fields are mandatory
            # Return None, or perhaps raise an error depending on expected usage.
            return None

        batch_size, num_channels = feature_tensors[0].shape

        # Add a feature dimension for concatenation
        # Shape: List[[batch_size, num_channels, 1]]
        reshaped_features = [f.unsqueeze(-1) for f in feature_tensors]

        # Concatenate along the new feature dimension
        # Shape: [batch_size, num_channels, num_feature_types]
        concatenated = torch.cat(reshaped_features, dim=2)

        if flatten_channels:
            # Shape: [batch_size, num_channels * num_feature_types]
            # Use reshape for safety with non-contiguous tensors if applicable
            return concatenated.reshape(batch_size, -1)
        else:
            # Shape: [batch_size, num_channels, num_feature_types]
            return concatenated


@dataclass
class TimeSeriesDataContainer:
    """
    Container for batch data fed into a time series model.

    Attributes:
        history_values: Tensor of historical observations.
            Shape: [batch_size, seq_len, num_channels]
        target_values: Tensor of future observations to predict.
            Shape: [batch_size, pred_len, num_targets]
        target_channels_indices: Tensor mapping target_values columns back to
            the original channel indices in history_values. Essential for
            multivariate forecasting where only a subset of channels might be targets.
            Prevents the model from guessing.
            Shape: [batch_size, num_targets]
        history_time_features: Tensor of time-derived features for the history timestamps window.
            Examples: hour_of_day, day_of_week, month_of_year (often cyclically encoded).
            Crucial for models to understand temporal context and seasonality.
            Shape: [batch_size, seq_len, num_time_features]
        target_time_features: Tensor of time-derived features for the prediction timestamps window.
            These are *known* future features.
            Shape: [batch_size, pred_len, num_time_features]
        static_features: Tensor of features that are constant over time for each series
            in the batch. Examples: Base Sampling Frequency,  Statistical Properties (Mean, Std Dev), etc.
            Shape: [batch_size, num_static_features, num_static_features_per_channel]
        history_mask: Optional boolean/float tensor indicating valid (1/True) vs padded (0/False)
            entries in history_values/history_time_features. Essential for handling
            variable length sequences or missing data within the history.
            Shape: [batch_size, seq_len]
        target_mask: Optional boolean/float tensor indicating valid (1/True) vs padded/missing (0/False)
            target values.
            Shape: [batch_size, pred_len] or [batch_size, pred_len, num_targets]
        dataset_id: Optional integer tensor indicating the source dataset for each time series
            in the batch. This can be used to differentiate between synthetic data, real-world
            datasets, or multiple datasets with varying distributions.
            Shape: [batch_size]
        frequency_encoded: Optional integer tensor representing the base frequency
            (e.g., 0='15T', 1='H', 2='D') for each item in the batch. Can be used
            as input to an embedding layer if explicitly conditioning on frequency is desired.
            Shape: [batch_size, num_channels]
    """

    history_values: torch.Tensor
    target_values: torch.Tensor

    target_channels_indices: torch.Tensor

    history_time_features: Optional[torch.Tensor] = None
    target_time_features: Optional[torch.Tensor] = None  # Known future time features

    # Contains pre-computed numerical static features.
    # Use .concatenate() to combine its features or merge with other static features.
    static_features: Optional[StaticFeaturesDataContainer] = None

    history_mask: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None

    # --- Metadata (Optional) ---
    # dataset_id: Optional[torch.Tensor] = None # Shape: [batch_size]
    # frequency_encoded: Optional[torch.Tensor] = None # Shape: [batch_size, num_channels]

    def __post_init__(self):
        """Basic validation of shapes."""
        # --- Mandatory Field Checks ---
        if not isinstance(self.history_values, torch.Tensor):
            raise TypeError("history_values must be a Tensor")
        if not isinstance(self.target_values, torch.Tensor):
            raise TypeError("target_values must be a Tensor")
        if not isinstance(self.target_channels_indices, torch.Tensor):
            raise TypeError("target_channels_indices must be a Tensor")

        batch_size = self.history_values.shape[0]
        seq_len = self.history_values.shape[1]
        num_channels = self.history_values.shape[2]
        pred_len = self.target_values.shape[1]

        # Basic shape checks
        if self.target_values.shape[0] != batch_size:
            raise ValueError("Batch size mismatch: target_values")
        if self.target_channels_indices.shape[0] != batch_size:
            raise ValueError("Batch size mismatch: target_channels_indices")
        if self.target_values.shape[2] != self.target_channels_indices.shape[1]:
            raise ValueError(
                f"num_targets mismatch: target_values has {self.target_values.shape[2]}, indices map {self.target_channels_indices.shape[1]}"
            )

        # Optional Field Checks
        if self.history_time_features is not None:
            if not isinstance(self.history_time_features, torch.Tensor):
                raise TypeError("history_time_features must be None or a Tensor")
            if self.history_time_features.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"Shape mismatch: history_time_features {self.history_time_features.shape} vs history {(batch_size, seq_len)}"
                )
        if self.target_time_features is not None:
            if not isinstance(self.target_time_features, torch.Tensor):
                raise TypeError("target_time_features must be None or a Tensor")
            if self.target_time_features.shape[:2] != (batch_size, pred_len):
                raise ValueError(
                    f"Shape mismatch: target_time_features {self.target_time_features.shape} vs target {(batch_size, pred_len)}"
                )
            if (
                self.history_time_features is not None
            ):  # Check consistency only if both exist
                if (
                    self.history_time_features.shape[2]
                    != self.target_time_features.shape[2]
                ):
                    raise ValueError(
                        f"Mismatch num_time_features: history {self.history_time_features.shape[2]} vs target {self.target_time_features.shape[2]}"
                    )

        # --- Validation for static_features (holding the container) ---
        if self.static_features is not None:
            if not isinstance(self.static_features, StaticFeaturesDataContainer):
                raise TypeError(
                    f"static_features must be None or StaticFeaturesDataContainer, got {type(self.static_features)}"
                )
            # Perform validation *within* the held container by trying to access its features
            try:
                feature_tensors = self.static_features.get_feature_tensors()
                # Check batch size and channel consistency if features exist
                if feature_tensors:
                    if feature_tensors[0].shape[0] != batch_size:
                        raise ValueError(
                            f"Batch size mismatch in static_features: expected {batch_size}, got {feature_tensors[0].shape[0]}"
                        )
                    if feature_tensors[0].shape[1] != num_channels:
                        raise ValueError(
                            f"Channel mismatch in static_features: expected {num_channels}, got {feature_tensors[0].shape[1]}"
                        )
            except (ValueError, TypeError) as e:
                # Re-raise with context
                raise ValueError(
                    f"Invalid StaticFeaturesDataContainer in static_features: {e}"
                ) from e

        # Mask validation
        if self.history_mask is not None:
            if not isinstance(self.history_mask, torch.Tensor):
                raise TypeError("history_mask must be None or a Tensor")
            if self.history_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Shape mismatch: history_mask {self.history_mask.shape} vs history {(batch_size, seq_len)}"
                )
        if self.target_mask is not None:
            if not isinstance(self.target_mask, torch.Tensor):
                raise TypeError("target_mask must be None or a Tensor")
            if not (
                self.target_mask.shape == (batch_size, pred_len)
                or self.target_mask.shape == self.target_values.shape
            ):
                raise ValueError(
                    f"Shape mismatch: target_mask {self.target_mask.shape} vs target {(batch_size, pred_len)} or {self.target_values.shape}"
                )
