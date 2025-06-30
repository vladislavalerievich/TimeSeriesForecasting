from datetime import date, timedelta

import numpy as np

from src.data_handling.data_containers import Frequency

DEFAULT_END_DATE = date.today()  # Use current date as end date
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=200 * 365)  # 200 years back
BASE_START_DATE = np.datetime64(DEFAULT_START_DATE)
BASE_END_DATE = np.datetime64(DEFAULT_END_DATE)


FREQUENCY_MAPPING = {
    Frequency.A: (
        "YE",
        "",
        365.25,
    ),  # Average days per year (accounting for leap years)
    Frequency.Q: ("Q", "", 91.3125),  # 365.25/4 - average days per quarter
    Frequency.M: ("M", "", 30.4375),  # 365.25/12 - average days per month
    Frequency.W: ("W", "", 7),
    Frequency.D: ("D", "", 1),
    Frequency.H: ("h", "", 1 / 24),
    Frequency.S: ("s", "", 1 / 86400),  # 24*60*60
    Frequency.T1: ("min", "1", 1 / 1440),  # 24*60
    Frequency.T5: ("min", "5", 1 / 288),  # 24*60/5
    Frequency.T10: ("min", "10", 1 / 144),  # 24*60/10
    Frequency.T15: ("min", "15", 1 / 96),  # 24*60/15
}
