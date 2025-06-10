from datetime import date, timedelta

DEFAULT_END_DATE = date.today()  # Use current date to define a range
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=20 * 365)  #  20 years back
BASE_START = DEFAULT_START_DATE.toordinal()
BASE_END = DEFAULT_END_DATE.toordinal() + 1

freq_dict = {
    "minute": {"freq": "min", "time_scale": 1 / 1440},
    "hourly": {"freq": "H", "time_scale": 1 / 24},
    "daily": {"freq": "D", "time_scale": 1},
    "weekly": {"freq": "W", "time_scale": 7},
    "monthly": {"freq": "MS", "time_scale": 30},
    "quarterly": {"freq": "QS", "time_scale": 90},
    "yearly": {"freq": "Y", "time_scale": 12},
}
