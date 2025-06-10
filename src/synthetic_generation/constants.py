from datetime import date, timedelta

from src.data_handling.data_containers import Frequency

DEFAULT_END_DATE = date.today()  # Use current date to define a range
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=20 * 365)  #  20 years back
BASE_START = DEFAULT_START_DATE.toordinal()
BASE_END = DEFAULT_END_DATE.toordinal() + 1


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
