import logging
import re

import pandas as pd

from src.data_handling.data_containers import Frequency

# Set up logger
logger = logging.getLogger(__name__)


def get_frequency_enum(freq_str: str) -> Frequency:
    """
    Map frequency string to a Frequency enum member, making it robust to variations.
    It extracts the base frequency and an optional multiplier.

    Args:
        freq_str: The frequency string to parse (e.g., "5T", "W-SUN", "M").

    Returns:
        The corresponding Frequency enum member.

    Raises:
        NotImplementedError: If the frequency string is not supported.
    """
    # Handle minute-based frequencies BEFORE pandas standardization
    # because pandas converts "5T" to just "min", losing the multiplier
    minute_match = re.match(r"^(\d*)T$", freq_str, re.IGNORECASE) or re.match(
        r"^(\d*)min$", freq_str, re.IGNORECASE
    )
    if minute_match:
        multiplier = int(minute_match.group(1)) if minute_match.group(1) else 1
        enum_key = f"T{multiplier}"
        try:
            return Frequency[enum_key]
        except KeyError:
            logger.warning(
                f"Unsupported minute frequency '{freq_str}' (multiplier: {multiplier}). "
                f"Falling back to '1min' ({Frequency.T1.value})."
            )
            return Frequency.T1

    # Now standardize frequency string for other cases
    try:
        offset = pd.tseries.frequencies.to_offset(freq_str)
        standardized_freq = offset.name
    except Exception:
        standardized_freq = freq_str

    # Handle other frequencies by their base (e.g., 'W-SUN' -> 'W', 'A-DEC' -> 'A')
    base_freq = standardized_freq.split("-")[0].upper()

    freq_map = {
        "A": Frequency.A,
        "Y": Frequency.A,  # Alias for Annual
        "YE": Frequency.A,  # Alias for Annual
        "Q": Frequency.Q,
        "QE": Frequency.Q,  # Alias for Quarterly
        "M": Frequency.M,
        "ME": Frequency.M,  # Alias for Monthly
        "W": Frequency.W,
        "D": Frequency.D,
        "H": Frequency.H,
        "S": Frequency.S,
    }

    if base_freq in freq_map:
        return freq_map[base_freq]

    raise NotImplementedError(f"Frequency '{standardized_freq}' is not supported.")
