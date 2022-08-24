import numpy as np
from datetime import datetime, timedelta
def yyyymmdd_to_datetime64(date_string: str) -> np.datetime64:
    date = np.datetime64(f"{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}")
    return date

def yyyymmdd_to_datetime(date_string: str) -> datetime:
    date = yyyymmdd_to_datetime64(date_string)
    date = date.astype(datetime)
    date = datetime.combine(date, datetime.min.time())
    return date

def hhmmss_to_timedelta64(time_string: str) -> np.timedelta64:
    time = (np.timedelta64(time_string[:2], "h")
        + np.timedelta64(time_string[2:4], "m") 
        + np.timedelta64(time_string[4:], "s"))
    return time

def hhmmss_to_timedelta(time_string: str) -> timedelta:
    time = hhmmss_to_timedelta64(time_string)
    time = time.astype(timedelta)
    return time

def datetime64_to_yyyymmdd_and_hhmmss(time: np.datetime64) -> tuple[str, str]:
    string = str(time)
    string = string.replace("-", "").replace(":", "")
    date, time = string.split("T")
    return date, time