from datetime import date, timedelta

DEFAULT_END_DATE = date.today()  # Use current date to define a range
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=20 * 365)  #  20 years back
BASE_START_ORD = DEFAULT_START_DATE.toordinal()
BASE_END_ORD = DEFAULT_END_DATE.toordinal()
