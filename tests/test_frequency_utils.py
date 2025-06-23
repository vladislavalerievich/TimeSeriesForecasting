import logging

import pytest

from src.data_handling.data_containers import Frequency
from src.data_handling.frequency_utils import get_frequency_enum


def test_standard_frequencies():
    """Test standard, single-character frequency strings."""
    assert get_frequency_enum("A") == Frequency.A
    assert get_frequency_enum("Q") == Frequency.Q
    assert get_frequency_enum("M") == Frequency.M
    assert get_frequency_enum("W") == Frequency.W
    assert get_frequency_enum("D") == Frequency.D
    assert get_frequency_enum("H") == Frequency.H
    assert get_frequency_enum("S") == Frequency.S
    assert get_frequency_enum("T") == Frequency.T1
    assert get_frequency_enum("min") == Frequency.T1


def test_pandas_style_frequencies():
    """Test pandas-style frequency strings with anchors (e.g., W-SUN)."""
    assert get_frequency_enum("A-DEC") == Frequency.A
    assert get_frequency_enum("YE-DEC") == Frequency.A
    assert get_frequency_enum("W-MON") == Frequency.W
    assert get_frequency_enum("W-TUE") == Frequency.W
    assert get_frequency_enum("W-WED") == Frequency.W
    assert get_frequency_enum("W-THU") == Frequency.W
    assert get_frequency_enum("W-FRI") == Frequency.W
    assert get_frequency_enum("W-SAT") == Frequency.W
    assert get_frequency_enum("W-SUN") == Frequency.W
    assert get_frequency_enum("QE-MAR") == Frequency.Q
    assert get_frequency_enum("Q-MAR") == Frequency.Q


def test_case_insensitivity():
    """Test that frequency strings are handled case-insensitively."""
    assert get_frequency_enum("a") == Frequency.A
    assert get_frequency_enum("q") == Frequency.Q
    assert get_frequency_enum("m") == Frequency.M
    assert get_frequency_enum("w") == Frequency.W
    assert get_frequency_enum("d") == Frequency.D
    assert get_frequency_enum("h") == Frequency.H
    assert get_frequency_enum("s") == Frequency.S


def test_minute_based_frequencies():
    """Test supported minute-based frequencies with multipliers."""
    assert get_frequency_enum("1T") == Frequency.T1
    assert get_frequency_enum("1min") == Frequency.T1
    assert get_frequency_enum("5T") == Frequency.T5
    assert get_frequency_enum("5min") == Frequency.T5
    assert get_frequency_enum("10T") == Frequency.T10
    assert get_frequency_enum("10min") == Frequency.T10
    assert get_frequency_enum("15T") == Frequency.T15
    assert get_frequency_enum("15min") == Frequency.T15


def test_unsupported_minute_fallback(caplog):
    """Test that unsupported minute frequencies fall back to T1 with a warning."""
    with caplog.at_level(logging.WARNING):
        # These should fall back to T1
        assert get_frequency_enum("2T") == Frequency.T1
        assert get_frequency_enum("30min") == Frequency.T1
        assert get_frequency_enum("90T") == Frequency.T1

        # Check for warnings
        assert "Unsupported minute frequency '2T'" in caplog.text
        assert "Falling back to '1min'" in caplog.text
        assert "Unsupported minute frequency '30min'" in caplog.text


def test_unsupported_frequencies():
    """Test that unsupported base frequencies raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        get_frequency_enum("B")  # Business day
    with pytest.raises(NotImplementedError):
        get_frequency_enum("C")  # Custom business day
    with pytest.raises(NotImplementedError):
        get_frequency_enum("L")  # Milliseconds
    with pytest.raises(NotImplementedError):
        get_frequency_enum("U")  # Microseconds
    with pytest.raises(NotImplementedError):
        get_frequency_enum("N")  # Nanoseconds


def test_aliases():
    """Test common aliases for frequencies."""
    assert get_frequency_enum("Y") == Frequency.A
    assert get_frequency_enum("y") == Frequency.A
    assert get_frequency_enum("H") == Frequency.H
    assert get_frequency_enum("h") == Frequency.H
