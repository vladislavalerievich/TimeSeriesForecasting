from datetime import date

BASE_START = date.fromisoformat("1900-01-01").toordinal()
BASE_END = date.fromisoformat("2024-12-31").toordinal() + 1


freq_dict = {
    "minute": {"freq": "min", "time_scale": 1 / 1440},
    "hourly": {"freq": "H", "time_scale": 1 / 24},
    "daily": {"freq": "D", "time_scale": 1},
    "weekly": {"freq": "W", "time_scale": 7},
    "monthly": {"freq": "MS", "time_scale": 30},
    "quarterly": {"freq": "QS", "time_scale": 90},
    "yearly": {"freq": "Y", "time_scale": 12},
}
