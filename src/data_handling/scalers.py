import torch
import torch.nn as nn


def custom_scaler_robust_multivariate(
    inputs, epsilon, q_low=0.25, q_high=0.75, clamp=None
):
    """
    Scale each channel of multivariate time series independently using median and IQR.

    Args:
        inputs: Tensor of shape [batch_size, seq_len, num_channels]
        epsilon: Small value to avoid division by zero
        q_low: Lower quantile (default: 0.25)
        q_high: Higher quantile (default: 0.75)
        clamp: Optional value to clamp scaled outputs

    Returns:
        scale_params: Tuple of (medians, iqrs) for each channel in each batch
        inputs_scaled: Scaled input tensor with same shape as input
    """
    batch_size, seq_len, num_channels = inputs.shape

    # Initialize output tensors
    medians = torch.zeros(batch_size, 1, num_channels, device=inputs.device)
    iqrs = torch.zeros(batch_size, 1, num_channels, device=inputs.device)
    inputs_scaled = torch.zeros_like(inputs)

    # Process each channel independently
    for c in range(num_channels):
        channel_values = inputs[:, :, c]

        # Compute median for each batch sample
        batch_medians = torch.nanmedian(channel_values, dim=1).values

        # Compute IQR for each batch sample
        q1 = torch.nanquantile(channel_values, q_low, dim=1)
        q3 = torch.nanquantile(channel_values, q_high, dim=1)
        batch_iqrs = q3 - q1

        # Ensure IQR is not zero
        batch_iqrs = torch.clamp(batch_iqrs, min=epsilon)

        # Store scaling parameters
        medians[:, 0, c] = batch_medians
        iqrs[:, 0, c] = batch_iqrs

        # Apply scaling
        scaled_channel = (
            channel_values - batch_medians.unsqueeze(1)
        ) / batch_iqrs.unsqueeze(1)

        # Handle NaN values
        scaled_channel = torch.where(
            torch.isnan(scaled_channel),
            torch.tensor(0.0, device=inputs.device),
            scaled_channel,
        )

        # Apply optional clamping
        if clamp is not None:
            scaled_channel = torch.clamp(scaled_channel, -clamp, clamp)

        # Store scaled values
        inputs_scaled[:, :, c] = scaled_channel

    return (medians, iqrs), inputs_scaled.to(torch.float32)


def min_max_scaler_multivariate(inputs, epsilon):
    """
    Apply min-max scaling independently to each channel of multivariate time series.

    Args:
        inputs: Tensor of shape [batch_size, seq_len, num_channels]
        epsilon: Small value to avoid division by zero

    Returns:
        scale_params: Tuple of (max_values, min_values) for each channel in each batch
        inputs_scaled: Scaled input tensor with same shape as input
    """
    batch_size, seq_len, num_channels = inputs.shape

    # Initialize output tensors
    max_values = torch.zeros(batch_size, 1, num_channels, device=inputs.device)
    min_values = torch.zeros(batch_size, 1, num_channels, device=inputs.device)
    inputs_scaled = torch.zeros_like(inputs)

    # Process each channel independently
    for c in range(num_channels):
        channel_values = inputs[:, :, c]

        # Get max and min for each batch sample
        batch_max = torch.max(channel_values, dim=1, keepdim=True)[0]
        batch_min = torch.min(channel_values, dim=1, keepdim=True)[0]

        # Ensure range is not zero
        range_values = torch.clamp(batch_max - batch_min, min=epsilon)

        # Store scaling parameters
        max_values[:, 0, c] = batch_max.squeeze(1)
        min_values[:, 0, c] = batch_min.squeeze(1)

        # Apply scaling
        scaled_channel = (channel_values - batch_min) / range_values

        # Handle NaN values
        scaled_channel = torch.where(
            torch.isnan(scaled_channel),
            torch.tensor(0.0, device=inputs.device),
            scaled_channel,
        )

        # Store scaled values
        inputs_scaled[:, :, c] = scaled_channel

    return (max_values, min_values), inputs_scaled.to(torch.float32)


def rescale_custom_robust(predictions, scale_params, target_index):
    """
    Rescale predictions using the custom robust scaling parameters.

    Args:
        predictions: Tensor of shape [batch_size, pred_len, 1]
        scale_params: Tuple of (medians, iqrs) as returned by custom_scaler_robust_multivariate
        target_index: Tensor of shape [batch_size] specifying the target channel for each batch item

    Returns:
        Rescaled predictions with shape [batch_size, pred_len, 1]
    """
    medians, iqrs = scale_params
    batch_size, pred_len, _ = predictions.shape

    # Rescale predictions
    rescaled = torch.zeros_like(predictions)

    for b in range(batch_size):
        # Get the target channel index for this batch item
        channel_idx = target_index[b].long()

        # Rescale using the appropriate parameters
        rescaled[b, :, 0] = (
            predictions[b, :, 0] * iqrs[b, 0, channel_idx] + medians[b, 0, channel_idx]
        )

    return rescaled


def rescale_min_max(predictions, scale_params, target_index):
    """
    Rescale predictions using the min-max scaling parameters.

    Args:
        predictions: Tensor of shape [batch_size, pred_len, 1]
        scale_params: Tuple of (max_values, min_values) as returned by min_max_scaler_multivariate
        target_index: Tensor of shape [batch_size] specifying the target channel for each batch item

    Returns:
        Rescaled predictions with shape [batch_size, pred_len, 1]
    """
    max_values, min_values = scale_params
    batch_size, pred_len, _ = predictions.shape

    # Rescale predictions
    rescaled = torch.zeros_like(predictions)

    for b in range(batch_size):
        # Get the target channel index for this batch item
        channel_idx = target_index[b].long()

        # Rescale using the appropriate parameters
        range_value = max_values[b, 0, channel_idx] - min_values[b, 0, channel_idx]
        rescaled[b, :, 0] = (
            predictions[b, :, 0] * range_value + min_values[b, 0, channel_idx]
        )

    return rescaled


class CustomScalingMultivariate(nn.Module):
    """
    Custom scaling module for multivariate time series data.
    """

    def __init__(self, name, clamp=None):
        super().__init__()
        self.name = name
        self.clamp = clamp

        if name == "custom_robust":
            self.scaler = lambda x, eps: custom_scaler_robust_multivariate(
                x, eps, clamp=clamp
            )
            self.rescaler = rescale_custom_robust
        elif name == "min_max":
            self.scaler = min_max_scaler_multivariate
            self.rescaler = rescale_min_max
        else:
            raise ValueError(f"Unknown scaler name: {name}")

    def forward(self, values, epsilon):
        """
        Scale input values using the configured scaler.

        Args:
            values: Input tensor of shape [batch_size, seq_len, num_channels]
            epsilon: Small value to avoid division by zero

        Returns:
            Tuple of (scale_params, scaled_values)
        """
        if values is None:
            return None, None
        return self.scaler(values, epsilon)

    def inverse_transform(self, scaled_values, scale_params, target_index=None):
        """
        Inverse transform scaled values back to original scale.

        Args:
            scaled_values: Scaled tensor of shape [batch_size, seq_len, 1] or [batch_size, seq_len]
            scale_params: Scaling parameters from forward pass
            target_index: Optional tensor of shape [batch_size] specifying target channels

        Returns:
            Rescaled tensor with same shape as scaled_values
        """
        if scaled_values is None or scale_params is None:
            return None

        # Ensure scaled_values has shape [batch_size, seq_len, 1]
        if scaled_values.dim() == 2:
            scaled_values = scaled_values.unsqueeze(-1)

        if target_index is None:
            # If no target_index provided, assume we're rescaling all channels
            target_index = torch.zeros(
                scaled_values.shape[0], device=scaled_values.device
            )
        elif target_index.dim() > 1:
            # Ensure target_index is 1D
            target_index = target_index.squeeze(-1)

        return self.rescaler(scaled_values, scale_params, target_index)
