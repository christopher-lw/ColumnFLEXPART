import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import argparse
from datetime import datetime

class SatillitePositions():
    def __init__(self, xarr, start=None, stop=None):
        self.xarr = xarr
        self._start = self.format_times(start)
        self._stop = self.format_times(stop) + np.timedelta64(1, "D")
    
        self.dataframe = self.get_dataframe()

    def get_dataframe(self):
        try: 
            xarr = self.xarr.drop("source_files")[["time", "longitude", "latitude"]]
        except ValueError:
            xarr = self.xarr[["time", "longitude", "latitude"]]
        
        if self._start is not None: xarr = xarr.where(xarr.time >= self._start, drop=True)
        if self._stop is not None: xarr = xarr.where(xarr.time <= self._stop, drop=True)
        df = xarr.to_dataframe().reset_index()
        points = gpd.points_from_xy(df.longitude, df.latitude)
        gdf = gpd.GeoDataFrame(data=df["time"], geometry=points, crs="EPSG:4326")
        return gdf

    def select_country(self, country):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[["name", "geometry"]]
        world.crs = 'epsg:4326'
        country_gdf = world[world.name == country][["geometry"]]
        selection = gpd.overlay(self.dataframe, country_gdf, how="intersection")
        return selection
    
    def format_times(self, time):
        if isinstance(time, datetime) or isinstance(time, np.datetime64) or time is None:
            return time
        else:
            time = "".join(c for c in time if c.isdigit())
            time = np.datetime64(datetime(int(time[:4]), int(time[4:6]), int(time[6:])))
            return time

    def format_selection(self, selection):
        #formatting time
        datetimes = selection.time.values.astype("datetime64[s]")
        datetimes = np.datetime_as_string(datetimes, unit="s")
        dates = []
        times = []
        for d in datetimes:
            date, time = d.split("T")
            dates.append(date.replace("-", ""))
            times.append(time.replace(":", ""))
        
        #formatting positions
        lon = selection.geometry.x.values
        lat = selection.geometry.y.values

        df = pd.DataFrame(data=dict(date=dates, time=times, longitude=lon, latitude=lat))
        return df        
        
    def save_selection(self, df, dir, time_filename="times.txt", coord_filename="coords.txt"):
        times_str = ""
        coords_str = ""
        for (date, time, lon, lat) in zip(df.date.values, df.time.values, df.longitude.values, df.latitude.values):
            times_str += f"{date},{time},{date},{time}\n"
            coords_str += f"{lon},{lon},{lat},{lat}\n"

        with open(os.path.join(dir, time_filename), "w") as f:
            f.write(times_str)
        with open(os.path.join(dir, coord_filename), "w") as f:
            f.write(coords_str)
        print(f"Saved data to: {os.path.join(dir, time_filename)}")
        print(f"Saved data to: {os.path.join(dir, coord_filename)}")
    
    def preprocess(self, dir, country=None, **kwargs):
        df = None
        if country is not None:
            df = self.select_country(country)
        df = self.format_selection(df)
        self.save_selection(df, dir, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to convert acos data to time and coordinate files readable for prepare_release.py")
    parser.add_argument("input_directory", type=str, help="Directory from which all ACOS files are loaded")
    parser.add_argument("output_directory", type=str, help="Directory for output files")
    parser.add_argument("--country", type=str, default=None, help="Country to filter measurements for")
    parser.add_argument("--start", type=None, default="", help="Start of time frame to convert. (format YYYYMMDD)")
    parser.add_argument("--stop", type=None, default="", help="End of time frame to convert. (format YYYYMMDD)")
    parser.add_argument("--out_name", type=str, default="", help="Specification for name of output files. Will be time_{name}.txt and coords_{name}.txt. If not set it will be time_{start}-{stop}.txt if start and stop are set")
    
    args = parser.parse_args()

    dir_in = args.input_directory
    dir_out = args.output_directory
    
    start = args.start
    stop = args.stop

    if start is not None and stop is None:
        stop = start

    files = []
    for file in os.listdir(dir_in):
        files.append(os.path.join(dir_in, file))
    xarr = xr.open_mfdataset(files, drop_variables="source_files")

    

    extension = ".txt"
    if args.out_name == "":
        if start is not None and stop is not None:
            if start == stop:
                extension = f"_{start}{extension}"
            else:
                extension = f"_{start}-{stop}{extension}"
    else:
        extension = f"_{args.out_name}{extension}"
    time_filename = "times" + extension
    coord_filename = "coords" + extension

    data = SatillitePositions(xarr, start=start, stop=stop)
    data.preprocess(dir=dir_out, country=args.country, time_filename=time_filename, coord_filename=coord_filename)
