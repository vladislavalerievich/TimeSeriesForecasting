import pytest
import torch

from src.data_handling.time_series_data_structure import (
    StaticFeaturesDataContainer,
    TimeSeriesDataContainer,
)


# Helper function to create dummy StaticFeaturesDataContainer
def create_static_features(batch_size, num_channels):
    return StaticFeaturesDataContainer(
        mean=torch.randn(batch_size, num_channels),
        std=torch.rand(batch_size, num_channels),
        median=torch.randn(batch_size, num_channels),
        min_val=torch.randn(batch_size, num_channels) - 1,
        max_val=torch.randn(batch_size, num_channels) + 1,
        autocorr_lag1=torch.randn(batch_size, num_channels),
        trend_slope=torch.randn(batch_size, num_channels),
    )


@pytest.mark.parametrize("num_channels", [1, 3, 5, 10])
@pytest.mark.parametrize("use_static_features", [True, False])
@pytest.mark.parametrize("use_masks", [True, False])
def test_time_series_data_container(num_channels, use_static_features, use_masks):
    batch_size = 4
    seq_len = 20
    pred_len = 5
    num_targets = min(2, num_channels)  # make sure num_targets <= num_channels
    num_time_features = 3

    # Generate dummy history and target
    history_values = torch.randn(batch_size, seq_len, num_channels)
    target_values = torch.randn(batch_size, pred_len, num_targets)
    target_channels_indices = torch.randint(0, num_channels, (batch_size, num_targets))

    # Optional: Time features
    history_time_features = torch.randn(batch_size, seq_len, num_time_features)
    target_time_features = torch.randn(batch_size, pred_len, num_time_features)

    # Optional: Static features
    static_features = (
        create_static_features(batch_size, num_channels)
        if use_static_features
        else None
    )

    # Optional: Masks
    history_mask = torch.ones(batch_size, seq_len) if use_masks else None
    target_mask = torch.ones(batch_size, pred_len) if use_masks else None

    # Create container
    container = TimeSeriesDataContainer(
        history_values=history_values,
        target_values=target_values,
        target_channels_indices=target_channels_indices,
        history_time_features=history_time_features,
        target_time_features=target_time_features,
        static_features=static_features,
        history_mask=history_mask,
        target_mask=target_mask,
    )

    # --- Assertions ---

    # Validate base tensors
    assert container.history_values.shape == (batch_size, seq_len, num_channels)
    assert container.target_values.shape == (batch_size, pred_len, num_targets)
    assert container.target_channels_indices.shape == (batch_size, num_targets)

    # Validate optional fields
    if use_static_features:
        assert container.static_features is not None
        concatenated_static = container.static_features.concatenate(
            flatten_channels=True
        )
        assert concatenated_static.shape[0] == batch_size
        # Should be batch_size x (num_channels * num_features)
        assert concatenated_static.shape[1] == num_channels * len(
            container.static_features.get_feature_tensors()
        )

    if use_masks:
        assert container.history_mask.shape == (batch_size, seq_len)
        assert container.target_mask.shape == (batch_size, pred_len)

    if container.history_time_features is not None:
        assert container.history_time_features.shape == (
            batch_size,
            seq_len,
            num_time_features,
        )
    if container.target_time_features is not None:
        assert container.target_time_features.shape == (
            batch_size,
            pred_len,
            num_time_features,
        )
