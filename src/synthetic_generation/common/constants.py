from datetime import date, timedelta

from src.data_handling.data_containers import Frequency

DEFAULT_END_DATE = date.today()  # Use current date to define a range
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=20 * 365)  #  20 years back
BASE_START = DEFAULT_START_DATE.toordinal()
BASE_END = DEFAULT_END_DATE.toordinal() + 1


FREQUENCY_MAPPING = {
    Frequency.A: ("A", "", 12),
    Frequency.Q: ("Q", "", 4),
    Frequency.M: ("M", "", 30),
    Frequency.W: ("W", "", 7),
    Frequency.D: ("D", "", 1),
    Frequency.H: ("h", "", 1 / 24),
    Frequency.S: ("s", "", 1 / 86400),  # 24*60*60
    Frequency.T1: ("min", "1", 1 / 1440),  # 24*60
    Frequency.T5: ("min", "5", 1 / 288),  # 24*60/5
    Frequency.T10: ("min", "10", 1 / 144),  # 24*60/10
    Frequency.T15: ("min", "15", 1 / 96),  # 24*60/15
}
