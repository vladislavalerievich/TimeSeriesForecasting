import torch
from torch.nn import Module


def identity_scaler(inputs, epsilon):
    return torch.ones_like(inputs[:, 0:1]), inputs


def custom_scaler_robust(inputs, epsilon, q_low=0.25, q_high=0.75, clamp=None):
    # Define a helper function to compute median and IQR excluding NaNs
    def compute_median_iqr(tensor):
        # Median
        median = tensor.nanmedian(dim=1).values

        # IQR (Interquartile Range)
        q1 = torch.nanquantile(tensor, q_low, dim=1)
        q3 = torch.nanquantile(tensor, q_high, dim=1)
        iqr = q3 - q1

        return median, iqr

    medians, iqrs = compute_median_iqr(inputs)

    # Ensure IQR is not zero to avoid division by zero
    iqrs = iqrs + epsilon

    # Reshape medians and iqrs to match the input shape for broadcasting
    medians = medians.view(-1, 1, 1)
    iqrs = iqrs.view(-1, 1, 1)
    # print(medians.shape, iqrs.shape, inputs_with_nans.shape)
    # Perform robust scaling
    inputs_scaled = (inputs - medians) / iqrs

    # Replace NaNs back to 0s if needed
    inputs_scaled = torch.where(
        torch.isnan(inputs_scaled), torch.tensor(0.0), inputs_scaled
    )
    if clamp is not None:
        inputs_scaled = torch.clamp(inputs_scaled, -clamp, clamp)
    return (medians, iqrs), inputs_scaled.to(torch.float32)


def min_max_scaler(inputs, epsilon):
    scale = [
        torch.max(inputs, dim=1, keepdim=True)[0] + epsilon,
        torch.min(inputs, dim=1, keepdim=True)[0] - epsilon,
    ]
    scale = torch.stack(scale, dim=0)
    output = (inputs - scale[1]) / (scale[0] - scale[1])
    return scale, output.to(torch.float32)


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
        batch_iqrs = batch_iqrs + epsilon

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
        batch_max = torch.max(channel_values, dim=1, keepdim=True)[0] + epsilon
        batch_min = torch.min(channel_values, dim=1, keepdim=True)[0] - epsilon

        # Store scaling parameters
        max_values[:, 0, c] = batch_max.squeeze(1)
        min_values[:, 0, c] = batch_min.squeeze(1)

        # Apply scaling
        range_values = batch_max - batch_min
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


class CustomScalingMultivariate(Module):
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
            # Identity scaler as fallback
            self.scaler = lambda x, eps: (torch.ones_like(x[:, 0:1, :]), x)

    def forward(self, values, epsilon):
        return self.scaler(values, epsilon)
