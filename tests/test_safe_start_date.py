import numpy as np
import pandas as pd
import pytest

from src.data_handling.data_containers import Frequency
from src.data_handling.time_features import (
    _get_frequency_str,
    compute_batch_time_features,
)
from src.synthetic_generation.common.utils import select_safe_start_date


@pytest.mark.parametrize(
    "history_len, future_len, freq, description",
    [
        (443, 195, Frequency.Q, "Quarterly case from training error"),
        (100, 200, Frequency.Q, "Quarterly with long future"),
        (200, 150, Frequency.M, "Monthly with medium length"),
        (80, 50, Frequency.A, "Annual with long history/future"),
        (100, 120, Frequency.M, "Monthly"),
        (500, 600, Frequency.W, "Weekly"),
        (200, 300, Frequency.D, "Daily"),
        (1000, 500, Frequency.H, "Hourly"),
        (925, 195, Frequency.D, "Daily case from training"),
        (576, 180, Frequency.H, "Hourly case from training"),
    ],
)
def test_safe_start_date_selection(history_len, future_len, freq, description):
    rng = np.random.default_rng(42)
    K_max = 15

    start_date = select_safe_start_date(history_len, future_len, freq, rng)
    assert isinstance(start_date, np.datetime64), (
        f"Start date is not datetime64 for {description}"
    )

    history_features, future_features = compute_batch_time_features(
        start=start_date,
        history_length=history_len,
        future_length=future_len,
        frequency=freq,
        K_max=K_max,
    )
    assert history_features.shape == (history_len, K_max), (
        f"History shape mismatch for {description}"
    )
    assert future_features.shape == (future_len, K_max), (
        f"Future shape mismatch for {description}"
    )

    offset_str = _get_frequency_str(freq, for_date_range=True)
    total_length = history_len + future_len
    date_range = pd.date_range(
        start=pd.Timestamp(start_date), periods=total_length, freq=offset_str
    )
    assert len(date_range) == total_length, (
        f"Date range length mismatch for {description}"
    )


@pytest.mark.parametrize(
    "history_len, future_len, freq, description",
    [
        (150, 100, Frequency.Q, "Long quarterly series"),
        (100, 80, Frequency.A, "Long annual series"),
        (100, 6, Frequency.A, "Short annual series"),
    ],
)
def test_extreme_cases(history_len, future_len, freq, description):
    rng = np.random.default_rng(123)
    start_date = select_safe_start_date(history_len, future_len, freq, rng)
    assert isinstance(start_date, np.datetime64), (
        f"Start date is not datetime64 for {description}"
    )


@pytest.mark.parametrize(
    "history_len, future_len, freq, description",
    [
        (2000, 900, Frequency.Q, "Very long quarterly series that exceeds window"),
        (1024, 900, Frequency.A, "Very long annual series that exceeds window"),
    ],
)
def test_time_span_exceeds_window(history_len, future_len, freq, description):
    """Test that extremely long time series raise appropriate errors"""
    rng = np.random.default_rng(123)
    with pytest.raises(
        ValueError, match="Required time span.*exceeds available date window"
    ):
        select_safe_start_date(history_len, future_len, freq, rng)
