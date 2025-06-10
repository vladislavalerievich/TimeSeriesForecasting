from datetime import date, timedelta
from enum import Enum

DEFAULT_END_DATE = date.today()  # Use current date to define a range
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=20 * 365)  #  20 years back
BASE_START = DEFAULT_START_DATE.toordinal()
BASE_END = DEFAULT_END_DATE.toordinal() + 1


class Frequency(Enum):
    M = "ME"  # Month End
    W = "W"  # Weekly
    D = "D"  # Daily
    H = "h"  # Hourly
    S = "s"  # Seconds
    T5 = "5min"  # 5 minutes
    T10 = "10min"  # 10 minutes
    T15 = "15min"  # 15 minutes


FREQUENCY_MAPPING = {
    Frequency.S: ("min", "1", 1 / 1440),
    Frequency.T5: ("min", "5", 1 / 1440),
    Frequency.T10: ("min", "10", 1 / 1440),
    Frequency.T15: ("min", "15", 1 / 1440),
    Frequency.H: ("H", "", 1 / 24),
    Frequency.D: ("D", "", 1),
    Frequency.W: ("W", "", 7),
    Frequency.M: ("MS", "", 30),
}
