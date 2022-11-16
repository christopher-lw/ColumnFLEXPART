from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union
import numpy as np
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder
import xarray as xr

from master.utils.utils import datetime64_to_yyyymmdd_and_hhmmss

class ColumnMeasurement(ABC):
    """Class to load, hold and use column measurement data such as ACOS or TCCON data to work with flexpart output."""
    def __init__(self, path: Union[Path, str], time: datetime):
        self.data, self.path, self.id = self.load(path, time)

    @abstractmethod
    def surface_pressure(self) -> float:
        """Return surface pressure"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Union[Path, str], time) -> tuple[xr.Dataset, Path, int]:
        """Loads data of measurement"""
        raise NotImplementedError

    @abstractmethod
    def interpolate_to_levels(self, dataarray: xr.DataArray, pressure_key: str, pressure_values: Iterable) -> xr.DataArray:
        """Interpolates given DataArray to levels of measurement data"""
        raise NotImplementedError
    
    @abstractmethod
    def add_variables(self, dataarray: xr.DataArray, variables: list[str]) -> xr.Dataset:
        """Creates Dataset from dataarray and dataarrays of mesurements named in variables"""
        raise NotImplementedError
    
    @abstractmethod
    def pressure_weighted_sum(self, dataset: xr.Dataset, data_var: str, with_averaging_kernel:bool) -> xr.DataArray:
        """Carries out pressure weighted sum with additional option to use averaging kernel"""
        raise NotImplementedError

    @abstractmethod
    def surface_temperature(self) -> float:
        """Retrun surface temperature"""


class TcconMeasurement(ColumnMeasurement):
    """Class to load, hold and use ACOS data to work with flexpart output. More information on how to treat data is found here: https://tccon-wiki.caltech.edu/Main/AuxilaryData#A_Priori_Profiles_and_Column_Averaging_Kernels_for_the_40obsolete_41_GGG2009_Data"""
    def __init__(self, path: Union[Path, str], time: datetime):
        super(TcconMeasurement, self).__init__(path, time)
    
    @staticmethod
    def get_times(tccon_data: xr.Dataset) -> "np.ndarray[np.datetime64]":
        """Transfers year day and time information and combines them to np.datetime64 obejcts"""
        dates = tccon_data.year.values.astype("int").astype("str").astype("datetime64") +\
            (tccon_data.day - 1).values.astype("timedelta64[D]")+\
            np.round(tccon_data.hour.values*3600).astype("timedelta64[s]")
        return dates
        
    @staticmethod
    def get_prior_times(tccon_data: xr.Dataset) -> pd.DatetimeIndex:   
        """Reads out time values for year, month and day and combines them to np.datetime64 object. Adds 12 hour for timestamp to be in the middle of the day and transfers it to utc."""
        prior_times = tccon_data.prior_year.values.astype("int").astype("str").astype("datetime64") +\
            (tccon_data.prior_month - 1).astype("timedelta64[M]") +\
            (tccon_data.prior_day - 1).astype("timedelta64[D]")+\
            np.timedelta64(12, "h")

        tf = TimezoneFinder()
        timezone_name = tf.timezone_at(lng=tccon_data.long_deg.values[0], lat=tccon_data.lat_deg.values[0])
        timezone = pytz.timezone(timezone_name)

        prior_times = [timezone.localize(pd.to_datetime(t)).astimezone(pytz.utc) for t in prior_times.values]
        prior_times = pd.DatetimeIndex(prior_times).tz_localize(None)
        return prior_times

    def load(self, path: Union[Path, str], time: datetime) -> xr.Dataset:
        path = Path(path)
        # Selecting needed variables
        variables = [
            "long_deg", "lat_deg", "year", "day", "hour", 
            "xco2_ppm_error", "xco2_ppm", "ak_co2", "prior_co2", 
            "prior_year", "prior_month", "prior_day", "asza_deg", 
            "pout_hPa", "tout_C", "prior_h2o", "prior_Pressure"
            ]
        data: xr.Dataset = xr.load_dataset(path, decode_times=False)[variables]
        data = data.isel(time = data.year == time.year)
        # Adjusting time coordinates to times instead of indices 
        times = self.get_times(data)
        prior_times = self.get_prior_times(data)
        data = data.assign_coords(time=times, prior_date=prior_times)
        # Adjust moist to dry air apriori:
        data  = data.assign(prior_co2 = data.prior_co2/(1 - data.prior_h2o))
        # Drop now unnecessary data
        data = data.drop(["year", "day", "hour", "prior_year", "prior_month", "prior_day", "prior_h2o"])
        # Get data for needed time
        time = np.datetime64(time)
        data = data.interp(time=[time])
        if data.ak_zenith.values.min() >= float(data.asza_deg):
            data = data.sel(ak_zenith=data.ak_zenith.min())
        elif data.ak_zenith.values.max() <= float(data.asza_deg):
            data = data.sel(ak_zenith=data.ak_zenith.max())
        else:
            data = data.interp(ak_zenith=[float(data.asza_deg)])
        data = data.isel(prior_date=np.argmin(np.abs(time - data.prior_date.values)))
        # Interpolate averaging kernel to prior levels 
        data = data.interp(ak_P_hPa = data.prior_Pressure.values, kwargs = dict(fill_value="extrapolate"))
        # Replacing old coordinate
        averaging_kernel = data.ak_co2.interp(
            ak_P_hPa = data.prior_Pressure.values).assign_coords(
                ak_P_hPa = data.prior_Height.values).rename(
                    ak_P_hPa = "prior_Height")
        data = (data.drop("ak_co2").drop_dims("ak_P_hPa")).merge(averaging_kernel.to_dataset())
        #renaming for consistency
        data = data.rename(xco2_ppm = "xco2", xco2_ppm_error = "xco2_uncertainty")
        return data.squeeze(drop=True), path, None
    
    def interpolate_to_levels(self, dataarray: xr.DataArray, pressure_key: str, pressure_values: Iterable) -> xr.DataArray:
        """Interpolates given DataArray to levels of measurement data"""
        dataarray = dataarray.assign_coords({pressure_key: pressure_values}).rename({pressure_key: "pressure"})
        tccon_pressures = self.data["prior_Pressure"]
        within_sounding_range = (tccon_pressures > dataarray.pressure.values.min()) & (tccon_pressures < dataarray.pressure.values.max())
        #get respective level information
        tccon_pressures = tccon_pressures.values[within_sounding_range]
        tccon_levels = self.data["prior_Height"].values[within_sounding_range]
        #interpolation to tccon levels
        dataarray = dataarray.interp(pressure=tccon_pressures)
        #assign levels to pressures
        dataarray = dataarray.assign_coords({"pressure": tccon_levels}).rename({"pressure": "prior_Height"})
        dataarray = dataarray.compute()
        return dataarray.squeeze(drop=True)

    def add_variables(
        self, 
        dataarray: xr.DataArray, 
        variables: list[str] = ["prior_Pressure", "ak_co2", "prior_co2"]
    ) -> xr.Dataset:
        """Creates Dataset from dataarray and dataarrays of mesurements named in variables"""
        tccon_variable_data = [self.data[var].squeeze(drop=True) for var in variables]
        dataset = xr.merge([dataarray.squeeze(drop=True), *tccon_variable_data])
        return dataset
    
    def pressure_weighted_sum(self, dataset: xr.Dataset, data_var: str, with_averaging_kernel:bool) -> xr.DataArray:
        """Carries out pressure weighted sum with additional option to use averaging kernel"""
        dataarray = dataset[data_var]
        # correct 0th value if surface pressure is inconsistent
        if np.isnan(dataarray[0]).prod():
            dataarray[0] = dataarray[1]
        if with_averaging_kernel: 
            averaging_kernel = self.data.ak_co2
            not_levels = [k for k in dataarray.dims if k != "prior_Height"]
            no_data = np.isnan(dataarray).prod(not_levels).values.astype(bool)
            dataarray[no_data] = 0
            # set values of averaging kernel to 0 for these values (only use prior here)
            averaging_kernel = xr.where(no_data, 0, averaging_kernel)
            dataset = dataset.drop("ak_co2")
            dataset = dataset.assign(dict(ak_co2 = averaging_kernel))
            dataarray = dataarray * dataset.ak_co2 + dataset.prior_co2 * (1 - dataset.ak_co2)
        pressure_weights = (dataset.prior_Pressure.values - np.pad(dataset.prior_Pressure.values, (0, 1), "constant")[1:])/dataset.prior_Pressure.values[0]
        pressure_weights = xr.DataArray(pressure_weights, dict(prior_Height = dataset.prior_Height.values))
        pw_dataarray = dataarray * pressure_weights
        result = pw_dataarray.sum(dim = "prior_Height")
        return result
        
    def surface_pressure(self) -> float:
        """Return surface pressure"""
        return float(self.data.pout_hPa)
    
    def surface_temperature(self) -> float:
        """Retrun surface temperature"""
        return float(self.data.tout_C) + 273.15

class AcosMeasurement(ColumnMeasurement):
    """Class to load, hold and use ACOS data to work with flexpart output."""
    def __init__(self, path: Union[Path, str], time: datetime):
        super(AcosMeasurement, self).__init__(path, time)

    def load(self, path: Union[Path, str], time: datetime) -> xr.Dataset:
        path = Path(path)
        sounding_datetime = np.datetime64(time)
        sounding_date, _ = datetime64_to_yyyymmdd_and_hhmmss(sounding_datetime)
        sounding_date = sounding_date[2:]
        file_path = None
        for file in path.parent.iterdir():
            if path.name + sounding_date in file.name:
                file_path = file
        if file_path is None:
            raise FileNotFoundError(
                f"Could not find matching ACOS file for date {sounding_date}"
            )
        acos_data = xr.load_dataset(file_path)
        # return acos_data.time, sounding_datetime
        acos_data = acos_data.isel(
            sounding_id=acos_data.time.values.astype("datetime64[s]") == sounding_datetime.astype("datetime64[s]")
        )
        sounding_id = int(acos_data.sounding_id)
        return acos_data, file_path, sounding_id

    def surface_pressure(self) -> float:
        return float(self.data.pressure_levels.values[0,-1])
    
    def interpolate_to_levels(
        self, 
        dataarray: xr.DataArray, 
        pressure_key: str, 
        pressure_values: Iterable,
    ) -> xr.DataArray:
        dataarray = dataarray.assign_coords({pressure_key: pressure_values}).rename({pressure_key: "pressure"})
        pressure_sounding = self.data.pressure_levels.values[0]
        #find possible levels to interpolate to
        within_sounding_range = (pressure_sounding > dataarray.pressure.values.min()) & (pressure_sounding < dataarray.pressure.values.max())
        #get respective level information
        pressure_sounding = pressure_sounding[within_sounding_range]
        levels_sounding= self.data.levels.values[within_sounding_range]
        #interpolation to acos levels
        dataarray = dataarray.interp(pressure=pressure_sounding)
        #assign levels to pressures
        dataarray = dataarray.assign_coords({"pressure": levels_sounding}).rename({"pressure": "levels"})
        dataarray = dataarray.compute()
        return dataarray.squeeze(drop=True)

    def add_variables(
        self,
        dataarray: xr.DataArray, 
        variables: list[str]= ["pressure_weight", "xco2_averaging_kernel", "co2_profile_apriori"]
    ) -> xr.Dataset:
        acos_variable_data = [self.data[var].squeeze(drop=True) for var in variables]
        dataset = xr.merge([dataarray.squeeze(drop=True), *acos_variable_data])
        return dataset
    
    def pressure_weighted_sum(
        self,
        dataset: xr.Dataset, 
        data_var: str, 
        with_averaging_kernel: bool
    ) -> xr.DataArray:
        dataarray = dataset[data_var]
        if with_averaging_kernel:
            averaging_kernel = dataset.xco2_averaging_kernel
            # get levels at which there is no data (filled up with nans)
            not_levels = [k for k in dataarray.coords.keys() if k != "levels"]
            no_data = np.isnan(dataarray).prod(not_levels).values.astype(bool)
            dataarray[no_data] = 0
            # set values of averaging kernel to 0 for these values (only use prior here)
            averaging_kernel = xr.where(no_data, 0, averaging_kernel)
            dataset = dataset.drop("xco2_averaging_kernel")
            dataset = dataset.assign(dict(xco2_averaging_kernel = averaging_kernel))
            dataarray = dataarray * dataset.xco2_averaging_kernel + dataset.co2_profile_apriori * (1 - dataset.xco2_averaging_kernel)
        pw_dataarray = dataarray * dataset.pressure_weight
        result = pw_dataarray.sum(dim = "levels")
        return result

    def surface_pressure(self) -> float:
        return self.data.pressure_levels.values[0,-1]
    
    def surface_temperature(self) -> float:
        return 22 + 273.15
