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


def rescale_custom_robust(predictions, scale_params, target_indices):
    """
    Rescale predictions using the custom robust scaling parameters.

    Args:
        predictions: Tensor of shape [batch_size, pred_len, num_targets]
        scale_params: Tuple of (medians, iqrs) as returned by custom_scaler_robust_multivariate
        target_indices: Tensor of shape [batch_size, num_targets] mapping predictions to original channels

    Returns:
        Rescaled predictions with shape [batch_size, pred_len, num_targets]
    """
    medians, iqrs = scale_params
    batch_size, pred_len, num_targets = predictions.shape

    # Rescale predictions
    rescaled = torch.zeros_like(predictions)

    for b in range(batch_size):
        for t in range(num_targets):
            # Get the original channel index for this target
            channel_idx = target_indices[b, t].item()

            # Rescale using the appropriate parameters
            rescaled[b, :, t] = (
                predictions[b, :, t] * iqrs[b, 0, channel_idx]
                + medians[b, 0, channel_idx]
            )

    return rescaled


def rescale_min_max(predictions, scale_params, target_indices):
    """
    Rescale predictions using the min-max scaling parameters.

    Args:
        predictions: Tensor of shape [batch_size, pred_len, num_targets]
        scale_params: Tuple of (max_values, min_values) as returned by min_max_scaler_multivariate
        target_indices: Tensor of shape [batch_size, num_targets] mapping predictions to original channels

    Returns:
        Rescaled predictions with shape [batch_size, pred_len, num_targets]
    """
    max_values, min_values = scale_params
    batch_size, pred_len, num_targets = predictions.shape

    # Rescale predictions
    rescaled = torch.zeros_like(predictions)

    for b in range(batch_size):
        for t in range(num_targets):
            # Get the original channel index for this target
            channel_idx = target_indices[b, t].item()

            # Rescale using the appropriate parameters
            range_value = max_values[b, 0, channel_idx] - min_values[b, 0, channel_idx]
            rescaled[b, :, t] = (
                predictions[b, :, t] * range_value + min_values[b, 0, channel_idx]
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
        elif name == "min_max":
            self.scaler = min_max_scaler_multivariate
        else:
            raise ValueError(f"Unknown scaler name: {name}")

    def forward(self, values, epsilon):
        return self.scaler(values, epsilon)
