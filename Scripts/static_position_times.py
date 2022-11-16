from datetime import datetime
from pathlib import Path
import argparse
from utils import yyyymmdd_to_datetime, hhmmss_to_timedelta, datetime_to_yyyymmdd_and_hhmmss
from read_sounding_positions import get_out_names
from timezonefinder import TimezoneFinder
import pytz
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to get UTC times for static measurement position at a specieal time every day.")
    parser.add_argument("start", type=str, help="Start date of times to setup YYYYMMDD")
    parser.add_argument("end", type=str, help="End dateof times to setup YYYYMMDD (day also included)")
    parser.add_argument("time", type=str, help="Time at each day to setup hhmmss")
    parser.add_argument("longitude", type=float, help="Longitude of station")
    parser.add_argument("latitude", type=float, help="Latitude of station")
    parser.add_argument("--out_dir", type=str, default="", help="Directory form output files with times")
    parser.add_argument("--out_name", type=str, default="", help="Name for outputfile")

    args = parser.parse_args()

    start = yyyymmdd_to_datetime(args.start)
    end = yyyymmdd_to_datetime(args.end)
    time = hhmmss_to_timedelta(args.time)

    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lng=args.longitude, lat=args.latitude)
    timezone = pytz.timezone(timezone_name)

    out_name, _ = get_out_names(args.out_name, args.start, args.end)
    out_dir = Path(args.out_dir)
    out_path = out_dir / out_name

    times_str = ""
    for date in pd.date_range(start, end):
        date = pd.to_datetime(date) + time
        date = timezone.localize(date)
        utc_datetime = date.astimezone(pytz.utc)
        utc_date, utc_time = datetime_to_yyyymmdd_and_hhmmss(utc_datetime)
        times_str += f"{utc_date},{utc_time},{utc_date},{utc_time}\n"

    with out_path.open("w") as f:
        f.write(times_str)
    print(f"Saved data to {out_path}")






