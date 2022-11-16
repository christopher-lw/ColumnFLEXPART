import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import argparse
from datetime import datetime
from typing import Union
import logging

class SatillitePositions():
    def __init__(self, xarr: xr.Dataset, start: str=None, stop: str=None, quality_flag: int=0, keep_co2: bool=False):
        self.logger = logging.getLogger("SatellitePositions")
        self.quality_flag = quality_flag
        #self.xarr = self.select_quality(xarr, quality_flag)
        self.xarr = xarr
        self._start = self.format_times(start)
        self._stop = self.format_times(stop) + np.timedelta64(1, "D")
    
        self.dataframe = self.get_dataframe(keep_co2)

    def select_quality(self, xarr: xr.Dataset, quality_flag: int) -> xr.Dataset:
        self.logger.debug("select_quality")
        xarr = xarr.isel(sounding_id = (xarr.xco2_quality_flag <= quality_flag))
        return xarr

    def get_dataframe(self, keep_co2:bool) -> gpd.GeoDataFrame:
        self.logger.debug("get_dataframe")
        keys = ["time", "longitude", "latitude", "xco2_quality_flag"]
        data_keys = ["time"]
        if keep_co2:
            keys.append("xco2")
            data_keys.append("xco2")
        try: 
            # xarr = self.xarr.drop("source_files")[["time", "longitude", "latitude"]]
            xarr = self.xarr.drop("source_files")[keys]
        except ValueError:
            # xarr = self.xarr[["time", "longitude", "latitude"]]
            xarr = self.xarr[keys]
        self.logger.debug("selecting times")
        if self._start is not None: xarr = xarr.where(xarr.time >= self._start, drop=True)
        if self._stop is not None: xarr = xarr.where(xarr.time <= self._stop, drop=True)
        xarr = self.select_quality(xarr, self.quality_flag)
        self.logger.debug("convertign to dataframe")
        df = xarr.to_dataframe().reset_index()
        self.logger.debug("getting coords")
        points = gpd.points_from_xy(df.longitude, df.latitude)
        self.logger.debug("converting to geodataframe")
        gdf = gpd.GeoDataFrame(data=df[data_keys], geometry=points, crs="EPSG:4326")
        return gdf

    def select_country(self, country:str) -> gpd.GeoDataFrame:
        self.logger.debug("select_country")
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[["name", "geometry"]]
        world.crs = 'epsg:4326'
        country_gdf = world[world.name == country][["geometry"]]
        selection = gpd.overlay(self.dataframe, country_gdf, how="intersection")
        return selection
    
    def format_times(self, time: Union[datetime,str]) -> np.datetime64:
        if isinstance(time, datetime) or isinstance(time, np.datetime64) or time is None:
            return time
        else:
            time = "".join(c for c in time if c.isdigit())
            time = np.datetime64(datetime(int(time[:4]), int(time[4:6]), int(time[6:])))
            return time

    def format_selection(self, selection: gpd.GeoDataFrame) -> pd.DataFrame:
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
        
    def save_selection(self, df: pd.DataFrame, dir: str, time_filename: str="times.txt", coord_filename: str="coord.txt"):
        times_str = ""
        coords_str = ""
        for (date, time, lon, lat) in zip(df.date.values, df.time.values, df.longitude.values, df.latitude.values):
            times_str += f"{date},{time},{date},{time}\n"
            coords_str += f"{lon},{lon},{lat},{lat}\n"

        with open(os.path.join(dir, time_filename), "w") as f:
            f.write(times_str)
        with open(os.path.join(dir, coord_filename), "w") as f:
            f.write(coords_str)
        logging.info(f"Saved data to: {os.path.join(dir, time_filename)}")
        logging.info(f"Saved data to: {os.path.join(dir, coord_filename)}")
    
    def preprocess(self, dir: str, country: str=None, **kwargs):
        df = None
        if country is not None:
            df = self.select_country(country)
        df = self.format_selection(df)
        self.save_selection(df, dir, **kwargs)

def get_out_names(out_name: str, start: str, stop: str) -> tuple[str, str]:
    """Constructs a filename for the output depending on arguments given.

    Args:
        out_name (str): name_to be set after "time_" if "" then start and stop are used for the filename
        start (str): start of inscluded data
        stop (str): stop of included data

    Returns:
        tuple[str, str]: name of time file, name of coords file
    """        
    extension = ".txt"
    if out_name == "":
        if start is not None and stop is not None:
            if start == stop:
                extension = f"_{start}{extension}"
            else:
                extension = f"_{start}-{stop}{extension}"
    else:
        extension = f"_{out_name}{extension}"
    time_filename = "times" + extension
    coord_filename = "coords" + extension
    return time_filename, coord_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to convert acos data to time and coordinate files readable for prepare_release.py")
    parser.add_argument("input_directory", type=str, help="Directory from which all ACOS files are loaded")
    parser.add_argument("output_directory", type=str, help="Directory for output files")
    parser.add_argument("--country", type=str, default=None, help="Country to filter measurements for")
    parser.add_argument("--start", type=None, default="", help="Start of time frame to convert. (format YYYYMMDD)")
    parser.add_argument("--stop", type=None, default="", help="End of time frame to convert. (format YYYYMMDD)")
    parser.add_argument("--out_name", type=str, default="", help="Specification for name of output files. Will be time_{name}.txt and coords_{name}.txt. If not set it will be time_{start}-{stop}.txt if start and stop are set")
    parser.add_argument("-v", action="store_true", default=False, help="Verbose mode showing debug logging")
    args = parser.parse_args()

    if args.v: 
        logging.basicConfig(format='%(name)s: %(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(name)s: %(levelname)s: %(message)s', level=logging.WARNING)

    dir_in = args.input_directory
    dir_out = args.output_directory
    
    start = args.start
    stop = args.stop

    if start is not None and stop is None:
        stop = start

    files = []
    for file in os.listdir(dir_in):
        files.append(os.path.join(dir_in, file))
    logging.debug("open_dataset...")    
    xarr = xr.open_mfdataset(files, drop_variables="source_files")
    logging.debug("done")
    time_filename, coord_filename = get_out_names(args.out_name, start, stop)

    data = SatillitePositions(xarr, start=start, stop=stop)
    data.preprocess(dir=dir_out, country=args.country, time_filename=time_filename, coord_filename=coord_filename)
