from copy import deepcopy
from datetime import datetime
from functools import cache
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union
import warnings
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cf
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
from columnflexpart.classes.measurement import AcosMeasurement, ColumnMeasurement, TcconMeasurement
from columnflexpart.utils import calc_enhancement, select_extent, to_tuple


class FlexDataset:
    """Class for postprocessing FLEXPART output. Includes plotting, enhancement and background calculation and metadata readout"""    
    def __init__(self, directory: str, **kwargs):
        nc_path: str = self.get_nc_path(directory)
        self._nc_path: str = nc_path
        self._dir: str = nc_path.rsplit("/", 1)[0]
        self._kwargs: dict = dict(
            extent=[-180, 180, -90, 90],
            ct_dir=None,
            ct_name_dummy=None,
            id_key="id",
            chunks="auto",
            name=None,
            cmap="jet",
            norm=LogNorm(),
            datakey="spec001_mr",
            persist=False,
        )
        self._kwargs.update(kwargs)
        self._plot_kwargs = self._kwargs.copy()
        [
            self._plot_kwargs.pop(key)
            for key in [
                "extent",
                "ct_dir",
                "ct_name_dummy",
                "id_key",
                "chunks",
                "name",
                "datakey",
                "persist",
            ]
        ]

        self.footprint: self.Footprint = self.Footprint(self)
        self.trajectories: Optional[self.Trajectory] = None
        if os.path.exists(os.path.join(self._dir, "trajectories.pkl")):
            self.trajectories = self.Trajectories(self)
        else:
            warnings.warn("No trajectory information found.")
        self.measurement: Optional[ColumnMeasurement] = None
        self.start, self.stop, self.release = self.get_metadata()
        self._background: Optional[float] = None
        self._background_layer: Optional[list[float]] = None
        self._background_inter: Optional[float] = None
        self._enhancement: Optional[float] = None
        self._enhancement_layer: Optional[list[float]] = None
        self._enhancement_inter: Optional[float] = None
        self._total: Optional[float] = None
        self._total_layer: Optional[list[float]] = None
        self._total_inter: Optional[float] = None
        
        self._last_boundary: Optional[list[float]] = None

    @staticmethod
    def get_nc_path(directory: str) -> str:
        """Finds file in directory that contains footprint data

        Args:
            directory (str): Directory to search in 

        Returns:
            str: Path of file with footprints
        """        
        nc_file = None
        for filename in os.listdir(directory):
            if "grid_time" in filename and ".nc" in filename:
                nc_file = filename
                break
        if nc_file is None:
            raise FileNotFoundError(
                f"No NetCDF file with 'grid_time' in given directory: {directory}"
            )
        return os.path.join(directory, nc_file)

    def get_metadata(self):
        """Reads out FLEXPART output directory to extract metadata

        Returns:
            tuple[datetime, datetime, dict]: Start and stop of simulation, and information abouut releases.
        """        
        # read things from the header and RELEASES.namelist
        with open(os.path.join(self._dir, "header_txt"), "r") as f:
            lines = f.readlines()
        ibdate, ibtime, iedate, ietime = lines[1].strip().split()[:4]
        ibtime = ibtime.zfill(6)
        ietime = ietime.zfill(6)
        start = datetime.strptime(iedate + ietime, "%Y%m%d%H%M%S")
        stop = datetime.strptime(ibdate + ibtime, "%Y%m%d%H%M%S")
        with open(os.path.join(self._dir, "RELEASES.namelist"), "r") as f:
            lines = f.readlines()[5:13]

        release = dict()
        for i, line in enumerate(lines):
            lines[i] = line.split("=")[1].strip()[:-1].strip()

        release["start"] = datetime.strptime(
            lines[2] + lines[3].zfill(6), "%Y%m%d%H%M%S"
        )
        release["stop"] = datetime.strptime(
            lines[0] + lines[1].zfill(6), "%Y%m%d%H%M%S"
        )
        release["lon1"] = float(lines[4])
        release["lon2"] = float(lines[5])
        release["lat1"] = float(lines[6])
        release["lat2"] = float(lines[7])
        release["lon"] = (release["lon1"] + release["lon2"]) / 2
        release["lat"] = (release["lat1"] + release["lat2"]) / 2
        release["boundary_low"] = self.footprint.dataset.RELZZ1.values
        release["boundary_up"] = self.footprint.dataset.RELZZ2.values
        release["heights"] = np.mean(
            [release["boundary_low"], release["boundary_up"]], axis=0
        )
        return start, stop, release

    @staticmethod
    def add_map(
        ax: plt.Axes = None,
        feature_list: list[cf.NaturalEarthFeature] = [
            cf.COASTLINE,
            cf.BORDERS,
        ],
        leave_lims: bool = False,
        **grid_kwargs,
    ) -> tuple[plt.Axes]:
        """Add map to axes using cartopy.

        Args:
            ax (Axes): Axes to add map to
            feature_list (list, optional): Features of cartopy to be added. Defaults to [cf.COASTLINE, cf.BORDERS, [cf.STATES, dict(alpha=0.1)]].
            extent (list, optional): list to define region ([lon1, lon2, lat1, lat2]). Defaults to None.

        Returns:
            Axes: Axes with added map
            Gridliner: cartopy.mpl.gridliner.Gridliner for further settings
        """
        if leave_lims:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        for feature in feature_list:
            feature, kwargs = (
                feature if isinstance(feature, list) else [feature, dict()]
            )
            ax.add_feature(feature, **kwargs)
        grid = True
        gl = None
        try:
            grid = grid_kwargs["grid"]
            grid_kwargs.pop("grid", None)
        except KeyError:
            pass
        if grid:
            grid_kwargs = (
                dict(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                if not bool(grid_kwargs)
                else grid_kwargs
            )
            gl = ax.gridlines(**grid_kwargs)
            gl.top_labels = False
            gl.right_labels = False
        if leave_lims:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        return ax, gl

    @staticmethod
    def subplots(*args, **kwargs):
        """Same as matplotlib function only with projection=ccrs.PlateCarree() as default subplot_kw.

        Returns:
            Figure: figure
            Axes: axes of figure
        """
        default_kwargs = dict(subplot_kw=dict(projection=ccrs.PlateCarree()))
        for key, val in kwargs.items():
            default_kwargs[key] = val
        kwargs = default_kwargs
        fig, ax = plt.subplots(*args, **kwargs)
        return fig, ax

    @staticmethod
    def pressure_factor(
        h: Union[float, np.ndarray],
        Tb: float = 288.15,
        hb : float = 0,
        R: float = 8.3144598,
        g: float = 9.80665,
        M: float = 0.0289644,
    ) -> Union[float, np.ndarray]:
        """Calculate factor of barrometric height formula as described here: https://en.wikipedia.org/wiki/Barometric_formula

        Args:
            h (fleat): height for factor calculation [m]
            Tb (float, optional): reference temperature [K]. Defaults to 288.15.
            hb (float, optional): height of reference [m]. Defaults to 0.
            R (float, optional): universal gas constant [J/(mol*K)]. Defaults to 8.3144598.
            g (float, optional): gravitational acceleration [m/s^2]. Defaults to 9.80665.
            M (float, optional): molar mass of Earth's air [kg/mol]. Defaults to 0.0289644.

        Returns:
            float: fraction of pressure at height h compared to height hb
        """
        factor = np.exp(-g * M * (h - hb) / (R * Tb))
        return factor


    def get_result_mode(self, value: Any, key: str, allow_read: bool, force_calculation: bool, boundary: list[float, float, float, float]):
        readable = True
        try:
            self.load_result(key, boundary)
        except (KeyError, FileNotFoundError):
            readable = False
        if force_calculation:
            mode = "calculate"
        elif value is not None and (boundary == self._last_boundary or "background" not in key or self.trajectories.endpoints is None):
            mode = "from_instance"
        elif allow_read and readable:
            mode = "load"
        else:
            mode = "calculate"
        print(f"{key}: mode = {mode}")
        return mode

    def to_pointspec_dataarray(self, data: list, name: str) -> xr.DataArray:
        """Constructs an xr.DataArray from onedimensional data of lenght of "pointspec" and uses "pointspec" as dimension.

        Args:
            data (list): One dimensional data of lenght of pointspec
            name (str): Name to set for DataArray

        Returns:
            xr.DataArray: DataArray with data aligned to pointspec as coordinates for further processing with "interpolate_to_acos_levels"
        """        
        dataarray = xr.DataArray(
                    data, 
                    coords = dict(pointspec = self.footprint.dataset.pointspec.values),
                    dims = dict(pointspec = self.footprint.dataset.pointspec.values),
                    name = name
                    )
        return dataarray

    def enhancement_layer(
        self,
        ct_file: str = None,
        boundary: list[float, float, float, float] = None,
        allow_read: bool = True,
        force_calculation: bool = False,
        chunks: dict = dict(time=10)
        ) -> list[float]:
        """Calculates enhancement for each layer (release) of the footprint

        Args:
            ct_file (str, optional): String describing file names of CT2019B flux data before time stamp, e.g: '/path/to/data/CT2019B.flux1x1.'. Defaults to None.
            boundary (list[float, float, float, float], optional): boundary to calculate enhancement for [longitude left, longitude right, latitude lower, latitude upper]. Defaults to None.
            allow_read (bool, optional): Switch to allow reading the result fro the memory if possible. Defaults to True.
            force_calculation (bool, optional): Switch to force an new calculation. Defaults to False.
            chunks (dict, optional): Chucks to use for loaded dataset (dask). Defaults to dict(time=10).

        Returns:
            list[float]: Enhancements per layer
        """        
        mode = self.get_result_mode(self._enhancement_layer, "enhancement_layer", allow_read, force_calculation, boundary)
        if mode == "from_instance":
            pass
        elif mode == "load":
            self._enhancement_layer = self.load_result("enhancement_layer", boundary)
        elif mode == "calculate":
            footprint = self.footprint.dataset.compute()
            if boundary is not None:
                footprint = select_extent(footprint, *boundary)
            molefractions = calc_enhancement(
                footprint,
                ct_file,
                self.stop.date(),
                self.start.date(),
                boundary=boundary,
                chunks=chunks,
            )
            self._enhancement_layer = molefractions
            self.save_result("enhancement_layer", self._enhancement_layer, boundary)
        return self._enhancement_layer

    def background_layer(self,
        allow_read: bool = True,
        boundary: list[float, float, float, float] = None,
        force_calculation: bool = False,
    ) -> list[float]:
        """Calculates background for each layer (release) of the released particles.

        Args:
            allow_read (bool, optional): Switch to allow reading the result fro the memory if possible. Defaults to True.
            boundary (list[float, float, float, float], optional): boundary to calculate background for [longitude left, longitude right, latitude lower, latitude upper]. Defaults to None.
            force_calculation (bool, optional): Switch to force an new calculation. Defaults to False.

        Returns:
            list[float]: Backgrounds per layer
        """    
        mode = self.get_result_mode(self._background_layer, "background_layer", allow_read, force_calculation, boundary)
        if mode == "from_instance":
            pass
        elif mode == "load":
            self._background_layer = self.load_result("background_layer", boundary)
        elif mode == "calculate":
            assert (
                boundary == self._last_boundary
            ), f"No endpoints calculated for given boundary. Your input: boundary = {boundary}, boundary of current endpoints: {self._last_boundary}."
            co2_means = (
                self.trajectories.endpoints[["co2", "pointspec"]]
                .groupby("pointspec")
                .mean()
                .values.flatten()
            )
            self._background_layer = co2_means
            self.save_result("background_layer", self._background_layer, boundary)
        return self._background_layer

    def total_layer(
        self,
        ct_file: str = None,
        allow_read: bool=True,
        boundary: list[float, float, float, float] = None,
        force_calculation: bool = False,
        chunks: dict = dict(time=10)
    ) -> list[float]:
        """Calculates total CO2 value for each layer (release) of the footprint

        Args:
            ct_file (str, optional): String describing file names of CT2019B flux data before time stamp, e.g: '/path/to/data/CT2019B.flux1x1.'. Defaults to None.
            boundary (list[float, float, float, float], optional): boundary to calculate CO2 for [longitude left, longitude right, latitude lower, latitude upper]. Defaults to None.
            allow_read (bool, optional): Switch to allow reading the result fro the memory if possible. Defaults to True.
            force_calculation (bool, optional): Switch to force an new calculation. Defaults to False.
            chunks (dict, optional): Chucks to use for loaded dataset (dask). Defaults to dict(time=10).

        Returns:
            list[float]: Enhancements per layer
        """        
        mode = self.get_result_mode(self._total_layer, "total_layer", allow_read, force_calculation, boundary)
        if mode == "from_instance":
            pass
        elif mode == "load":
            self._total_layer = self.load_result("total_layer", boundary)
        elif mode == "calculate":
            background = self.background_layer(allow_read=allow_read, force_calculation=force_calculation, boundary=boundary)
            enhancement = self.enhancement_layer(ct_file, allow_read=allow_read, force_calculation=force_calculation, chunks=chunks)
            self._total_layer = list(np.array(background) + np.array(enhancement))
            self.save_result("total_layer", self._total_layer, boundary)
        return self._total_layer

    def total(self,
        ct_file: str = None,
        allow_read: bool=True,
        boundary: list[float, float, float, float] = None,
        force_calculation: bool = False,
        chunks: dict = dict(time=10),
        interpolate: bool = True
    ) -> float:
        total_layer = self.total_layer(ct_file, allow_read, boundary, force_calculation, chunks)
        if interpolate:
            mode = self.get_result_mode(self._total_inter, "total_inter", allow_read, force_calculation, boundary)
            if mode == "from_instance":
                pass
            elif mode == "load":
                self._total_inter = self.load_result("total_inter", boundary)
            elif mode == "calculate":
                total_layer = self.to_pointspec_dataarray(total_layer, "total_layer")
                pressures = self.measurement.surface_pressure() * self.pressure_factor(self.release["heights"], Tb=self.measurement.surface_temperature())
                total_layer = self.measurement.interpolate_to_levels(total_layer, "pointspec", pressures)
                total_layer = self.measurement.add_variables(total_layer)
                self._total_inter = self.measurement.pressure_weighted_sum(total_layer, "total_layer", with_averaging_kernel = True).values
                self._total_inter = float(self._total_inter)
                self.save_result("total_inter", self._total_inter, boundary)
            return self._total_inter
        else: 
            mode = self.get_result_mode(self._total, "total", allow_read, force_calculation, boundary)
            if mode == "from_instance":
                pass
            elif mode == "load":
                self._total = self.load_result("total", boundary)
            elif mode == "calculate":    
                pressure_weights = self.pressure_weights_from_height()
                self._total = (total_layer * pressure_weights).sum()
                self.save_result("total", self._total, boundary)
            return self._total
    
    def enhancement(
        self,
        ct_file: str = None,
        boundary: list[float, float, float, float] = None,
        allow_read: bool = True,
        force_calculation: bool = False,
        chunks: dict = dict(time=10),
        interpolate: bool = True,
    ) -> int:
        """Returns enhancement based on Carbon Tracker emmission data and the footprint. Is either calculated, loaded, or read from class.

        Args:
            ct_file (str, optional): Name of Carbon Tracker (only shared part not the time stamp). Defaults to None.
            p_surf (float, optional): Surface pressure in millibars. Defaults to 1013.
            allow_read (bool, optional): Flag to allow reading of result from file . Defaults to True.

        Returns:
            float: Enhancement of CO2 in ppm
        """
        molefractions = self.enhancement_layer(ct_file, boundary, allow_read, force_calculation, chunks)
        #self.save_result("enhancement_layer", molefractions, boundary)
        if interpolate:
            mode = self.get_result_mode(self._enhancement_inter, "enhancement_inter", allow_read, force_calculation, boundary)
            if mode == "from_instance":
                pass
            elif mode == "load":
                self._enhancement_inter = self.load_result("enhancement_inter", boundary)
            elif mode == "calculate":    
                molefractions = self.to_pointspec_dataarray(molefractions, "enhancement_layer")
                pressures = self.measurement.surface_pressure() * self.pressure_factor(self.release["heights"], Tb=self.measurement.surface_temperature())
                molefractions = self.measurement.interpolate_to_levels(molefractions, "pointspec", pressures)
                molefractions = self.measurement.add_variables(molefractions)
                self._enhancement_inter = self.measurement.pressure_weighted_sum(molefractions, "enhancement_layer", with_averaging_kernel = False).values
                self._enhancement_inter = float(self._enhancement_inter)
                self.save_result("enhancement_inter", self._enhancement_inter, boundary)
            return self._enhancement_inter
        
        else: 
            mode = self.get_result_mode(self._enhancement, "enhancement", allow_read, force_calculation, boundary)
            if mode == "from_instance":
                pass
            elif mode == "load":
                self._enhancement = self.load_result("enhancement", boundary)
            elif mode == "calculate":    
                pressure_weights = self.pressure_weights_from_height()
                self._enhancement = (molefractions * pressure_weights).sum()
                self.save_result("enhancement", self._enhancement, boundary)

            return self._enhancement

    def background(
        self,
        allow_read: bool = True,
        boundary: list[float, float, float, float] = None,
        interpolate: bool = True,
        force_calculation: bool = False,
    ) -> float:
        """Returns background calculation in ppm based on trajectories in trajectories.pkl file. Is either loaded from file, calculated or read from class.

        Args:
            allow_read (bool, optional): Flag to allow reading of result from file. Defaults to True.

        Returns:
            float: Background CO2 in ppm
        """
        co2_means = self.background_layer(allow_read, boundary, force_calculation)
        if interpolate:
            mode = self.get_result_mode(self._background_inter, "background_inter", allow_read, force_calculation, boundary)
            if mode == "from_instance":
                pass
            elif mode == "load":
                self._background_inter = self.load_result("background_inter", boundary)
            elif mode == "calculate":
                co2_means = self.to_pointspec_dataarray(co2_means, "background_layer")
                pressures = self.measurement.surface_pressure() * self.pressure_factor(self.release["heights"], Tb=self.measurement.surface_temperature())
                co2_means = self.measurement.interpolate_to_levels(co2_means, "pointspec", pressures)
                co2_means = self.measurement.add_variables(co2_means)
                self._background_inter = self.measurement.pressure_weighted_sum(co2_means, "background_layer", with_averaging_kernel=True).values
                self._background_inter = float(self._background_inter)
                self.save_result("background_inter", self._background_inter, boundary)
            return self._background_inter
            
        else: 
            mode = self.get_result_mode(self._background, "background", allow_read, force_calculation, boundary)
            if mode == "from_instance":
                pass
            elif mode == "load":
                self._background = self.load_result("background", boundary)
            elif mode == "calculate":
                pressure_weights = self.pressure_weights_from_height()
                self._background = (co2_means * pressure_weights).sum()
                self.save_result("background", self._background, boundary)
            return self._background

    def load_result(
        self, name: str, boundary: list[float, float, float, float] = None
    ) -> float:

        if not boundary is None:
            boundary = [float(b) for b in boundary]
        file = os.path.join(self._dir, f"{name}.json")
        with open(file) as f:
            data = json.load(f)
        result = data[str(boundary)]
        return result

    def save_result(
        self,
        name: str,
        result: Union[float, Iterable],
        boundary: list[float, float, float, float] = None,
    ):
        if not boundary is None:
            boundary = [float(b) for b in boundary]
        file = os.path.join(self._dir, f"{name}.json")
        data = {}
        if os.path.exists(file):
            with open(file) as f:
                data = json.load(f)
        try:    
            data[str(boundary)] = float(result)
        except TypeError:
            if isinstance(result, np.ndarray):
                result = result.astype(float)
            data[str(boundary)] = list(result)
        with open(file, "w") as f:
            json.dump(data, f)

    def delete_results(self, names=["enhancement", "enhancement_inter", "enhancement_layer", "background", "background_inter", "background_layer", "total", "total_inter", "total_layer"]):
        for name in names:
            file = os.path.join(self._dir, f"{name}.json")
            if os.path.exists(file):
                os.remove(file)
    
    def reset(self):
        self._background = None
        self._background_layer = None
        self._background_inter = None
        self._enhancement = None
        self._enhancement_layer = None
        self._enhancement_inter = None
        self._total = None
        self._total_layer = None
        self._total_inter = None

    def load_measurement(self, path: Union[Path, str], measurement_type: Literal["ACOS", "TCCON"]) -> xr.Dataset:
        """Loads column measurement of FLEXPART run

        Args:
            path (Union[Path, str]): Path of Acos file unitl timestamp. E.g.: '/path/to/acos/folder/acos_LtCO2_'
            measurement_type (Literal["ACOS", "TCCON"]): Type of measurement to be loaded.

        Returns:
            xr.Dataset: Dataof loadaed measurement. 
        """
        if measurement_type == "ACOS":
            measurement = AcosMeasurement
        elif measurement_type == "TCCON":
            measurement = TcconMeasurement
        else:
            raise ValueError(f"Possible values for measurement_type are: ['ACOS', 'TCCON'] not {measurement_type}")
        self.measurement = measurement(path = path, time = self.release["start"])
        return self.measurement.data
    
    def get_pressures(self, heights, lon, lat):
        pressures = self.pressure_factor(heights)
        return pressures
    
    def pressure_weights_from_height(
        self,
    ):
        pressures_low = self.pressure_factor(
            self.release["boundary_low"], self.release["lon"], self.release["lat"]
        )
        pressures_up = self.pressure_factor(
            self.release["boundary_up"], self.release["lon"], self.release["lat"]
        )
        pressure_diffs = pressures_low - pressures_up
        pressure_weights = pressure_diffs / pressure_diffs.sum()
        return pressure_weights

    class Footprint:
        def __init__(self, outer):
            self._outer: FlexDataset = outer
            self._nc_path = outer._nc_path
            self._dir = outer._dir
            self.extent = outer._kwargs["extent"]
            self.chunks = outer._kwargs["chunks"]
            self.name = outer._kwargs["name"]
            self.datakey = outer._kwargs["datakey"]

            self._plot_kwargs = outer._plot_kwargs

            self.dataset = xr.open_dataset(self._nc_path, chunks=self.chunks)
            self.dataarray = self.dataset[self.datakey]
            if self._outer._kwargs["persist"]:
                self.dataarray = self.dataarray.persist()

            self._total = None
            self._total_inter = None
            self._total_inter_time = None

        def reset(self):
            self._total = None
            self._total_inter = None
            self._total_inter_time = None
        
        def delete_results(self):
            files = ["Footprint_total.nc", "Footprint_total_inter.nc", "Footprint_total_inter_time.nc"]
            for file in files:
                if os.path.exists(os.path.join(self._dir, file)):
                    os.remove(os.path.join(self._dir, file))
        def save_total(self, interpolate: bool = True):
            """Saves Footprints

            Args:
                include_sums (bool, optional): _description_. Defaults to True.
            """
            if interpolate:
                inter_time_path = os.path.join(self._dir, "Footprint_total_inter_time.nc")
                inter_path = os.path.join(self._dir, "Footprint_inter.nc")
                self._total_inter_time.to_netcdf(inter_time_path)
                self._total_inter.to_netcdf(inter_path)
            else:
                sum_path = os.path.join(self._dir, "Footprint_total.nc")
                self._total.to_netcdf(sum_path)

        def calc_total(self, interpolate: bool = True):
            """Calculates total Footprints"""
            if interpolate:
                if self._outer.measurement is None:
                    raise ValueError("Cannot calculate total footprint with interpolation without measurement data. To interpolate first use FlexDataset2.load_measurement(). To just calculate sum over heights set interpolate=False")
                self._total_inter = None
                pressures = self._outer.measurement.surface_pressure() * self._outer.pressure_factor(self._outer.release["heights"], Tb=self._outer.measurement.surface_temperature())
                total = self._outer.measurement.interpolate_to_levels(self.dataarray, "pointspec", pressures)
                total = self._outer.measurement.add_variables(total)
                self._total_inter_time = self._outer.measurement.pressure_weighted_sum(total, self.datakey, with_averaging_kernel=False).compute()
                self._total_inter = self._total_inter_time.sum("time")
                return self._total_inter

            else:
                self._total = None
                self._total = self.dataarray.sum(dim=["time", "pointspec"]).compute()
                return self._total

        def load_total(self, interpolate: bool = True):
            """Loads Footprints from directory of DataSet data"""
            if interpolate:
                self._total_inter_time = None
                self._total_inter = None
                inter_time_path = os.path.join(self._dir, "Footprint_total_inter_time.nc")
                inter_path = os.path.join(self._dir, "Footprint_inter.nc")
                self._total_inter_time = xr.load_dataarray(inter_time_path)
                self._total_inter = xr.load_dataarray(inter_path)
            else:
                self._total = None
                path = os.path.join(self._dir, "Footprint_total.nc")
                self._total = xr.load_dataarray(path)

        def total(self, interpolate: bool = True):
            """Get footprints from either loading of calculation"""
            if self._total is None:
                try:
                    self.load_total(interpolate)
                except FileNotFoundError:
                    self.calc_total(interpolate)
                    self.save_total(interpolate)
            return self._total if not interpolate else self._total_inter

        def plot(
            self,
            ax: plt.Axes = None,
            time: list[int] = None,
            pointspec: list[int] = None,
            interpolate: bool = True,
            plot_func: str = None,
            plot_station: bool = True,
            add_map: bool = True,
            station_kwargs: dict = dict(color="black"),
            **kwargs,
        ) -> tuple[plt.Figure, plt.Axes]:

            plot_kwargs = deepcopy(self._plot_kwargs)
            plot_kwargs.update(kwargs)
            if ax is None:
                fig, ax = self._outer.subplots()
            else:
                fig = ax.get_figure()
            footprint = self.sum(time, pointspec, interpolate)
            footprint = footprint.where(footprint != 0)[:, :, ...]
            if plot_func is None:
                footprint.plot(ax=ax, **plot_kwargs)
            else:
                getattr(footprint.plot, plot_func)(ax=ax, **plot_kwargs)
            if plot_station:
                ax.scatter(
                    self._outer.release["lon1"],
                    self._outer.release["lat1"],
                    **station_kwargs,
                )
            if add_map:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                self._outer.add_map(ax)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            return fig, ax

        @to_tuple(["time", "pointspec"], [1, 2])
        @cache
        def sum(
            self, time: list[int] = None, pointspec: list[int] = None, interpolate: bool = True
        ) -> xr.DataArray:
            footprint = None
            if time is None and pointspec is None:
                footprint = self.total(interpolate)
            elif time is None:
                footprint = self.dataarray.sum(dim=["time"])
                footprint = (
                    footprint.isel(dict(pointspec=list(pointspec)))
                    .sum(dim=["pointspec"])
                    .compute()
                )
            elif pointspec is None:
                footprint = self.dataarray.sum(dim=["pointspec"])
                footprint = (
                    footprint.isel(dict(time=list(time))).sum(dim=["time"]).compute()
                )
            return footprint

    class Trajectories:
        def __init__(self, outer):
            self._outer = outer
            self._dir = outer._dir
            self._ct_dir = outer._kwargs["ct_dir"]
            self._ct_dummy = outer._kwargs["ct_name_dummy"]
            self._id_key = outer._kwargs["id_key"]
            self.dataframe = pd.read_pickle(
                os.path.join(self._dir, "trajectories.pkl")
            ).reset_index()
            self._min_time = self.dataframe.time.min()
            self._max_time = self.dataframe.time.max()
            self.endpoints = None
            self.ct_data = None

        def ct_endpoints(
            self,
            boundary: list[float, float, float, float] = None,
            ct_dir: str = None,
            ct_dummy: str = None,
        ) -> pd.DataFrame:
            if boundary is not None:
                df_outside = self.dataframe[
                    (
                        (self.dataframe.longitude < boundary[0])
                        | (self.dataframe.longitude > boundary[1])
                        | (self.dataframe.latitude < boundary[2])
                        | (self.dataframe.latitude > boundary[3])
                    )
                ]
                df_outside = (
                    df_outside.loc[df_outside.groupby(self._id_key)["time"].idxmax()]
                    .reset_index()
                    .drop(columns="index")
                )
                df_inside = self.dataframe[
                    ~self.dataframe[self._id_key].isin(df_outside[self._id_key])
                ]
                df_inside = df_inside.loc[
                    df_inside.groupby(self._id_key)["time"].idxmin()
                ]
                df_total = pd.concat([df_outside, df_inside])

            else:
                df_total = self.dataframe.loc[
                    self.dataframe.groupby(self._id_key)["time"].idxmin()
                ]

            _ = self.load_ct_data(ct_dir, ct_dummy)
            # finding cells of endpoints for time longitude and latitude
            variables = ["time", "longitude", "latitude"]
            for i, var in enumerate(variables):
                ct_vals = self.ct_data[var].values
                df_vals = df_total[var].values
                diff = np.abs(df_vals[:, None] - ct_vals[None, :])
                inds = np.argmin(diff, axis=-1)
                df_total.insert(loc=1, column=f"ct_{var}", value=inds)
            df_total.insert(
                loc=1,
                column="pressure_height",
                value=101300 * self._outer.pressure_factor(df_total.height),
            )
            # finding cells of endpoints for height
            ct_vals = self.ct_data.pressure.isel(
                time=xr.DataArray(df_total.ct_time.values),
                latitude=xr.DataArray(df_total.ct_latitude.values),
                longitude=xr.DataArray(df_total.ct_longitude.values),
            )
            df_vals = df_total.pressure_height.values
            diff = np.abs(df_vals[:, None] - ct_vals)
            inds = np.argsort(diff, axis=-1)[:, :2]
            inds = abs(inds).min(axis=1)
            df_total.insert(1, "ct_height", value=inds)
            # sorting and filtering
            self.endpoints = df_total.sort_values(self._id_key)
            self.endpoints = self.endpoints[np.sign(self.endpoints.pointspec) == 1]
            self.endpoints.attrs["boundary"] = boundary
            self.endpoints.attrs["height unit"] = "m"
            self.endpoints.attrs["pressure unit"] = "Pa"
            self.endpoints.attrs["co2 unit"] = "ppm"
            self._outer._last_boundary = boundary
            return self.endpoints

        def load_ct_data(self, ct_dir: str = None, ct_dummy: str = None) -> xr.Dataset:
            if ct_dir is not None:
                self._ct_dir = ct_dir
            if ct_dummy is not None:
                self._ct_dummy = ct_dummy
            file_list = []
            for date in np.arange(
                self._min_time,
                self._max_time + np.timedelta64(1, "D"),
                dtype="datetime64[D]",
            ):
                date = str(date)
                file_list.append(
                    os.path.join(self._ct_dir, self._ct_dummy + date + ".nc")
                )
            ct_data = xr.open_mfdataset(file_list, combine="by_coords")

            self.ct_data = ct_data[["co2", "pressure"]].compute()
            return self.ct_data

        def save_endpoints(self, name: str = "endpoints.pkl", dir: str = None):

            if dir is None:
                dir = self._dir
            save_path = os.path.join(dir, name)
            self.endpoints.to_pickle(save_path)
            print(f"Saved endpoints to {save_path}")

        def load_endpoints(self, name: str = None, dir: str = None):
            """Load endpoints from endpoints.pkl file or file with personalized name.

            Args:
                name (str, optional): Name of output. Defaults to None (results in endpoints.pkl).
                dir (dir, optional): Directory for file. Defaults to None.
            """
            if name is None:
                name = "endpoints.pkl"
            if dir is None:
                dir = self._dir
            read_path = os.path.join(dir, name)
            self.endpoints = pd.read_pickle(read_path).sort_values(self._id_key)
            self._outer._last_boundary = self.endpoints.attrs["boundary"]

        def co2_from_endpoints(
            self,
            exists_ok: bool = True,
            boundary: list[float, float, float, float] = None,
            ct_dir: str = None,
            ct_dummy: str = None,
        ) -> np.ndarray:
            """Returns CO2 at positions of the endpoints of the particles. Result is also saved to endpoints. Pressure weights are also calculated based on the pointspec value of the particles.

            Args:
                exists_ok (bool, optional): Flag for recalculation if co2 was allready calculated. Defaults to True.
                boundary (list, optional): Boundaries for optional endpoint calculateion if no endpoints exist so far. Defaults to None.
                ct_dir (str, optional): Directory for carbon tracker data. Defaults to None.
                ct_dummy (str, optional): Start of each Carbon Tracker file util the time stamp. Defaults to None.

            Returns:
                np.ndarray: predicted CO2 values
            """
            if self.endpoints is None:
                print(
                    "No endpoints found. To load use load_endpoints(). Calculation of endpoints..."
                )
                _ = self.ct_endpoints(boundary, ct_dir, ct_dummy)
                print("Done")
            if self.ct_data is None:
                _ = self.load_ct_data(ct_dir, ct_dummy)

            if "co2" in self.endpoints.columns:
                print(
                    "'co2' is allready in endpoints. To calculate again set exists_ok=Flase"
                ) or "n"
                if not exists_ok:
                    self.endpoints.drop("co2")
                    co2_values = self.ct_data.co2.isel(
                        time=xr.DataArray(self.endpoints.ct_time.values),
                        latitude=xr.DataArray(self.endpoints.ct_latitude.values),
                        longitude=xr.DataArray(self.endpoints.ct_longitude.values),
                        level=xr.DataArray(self.endpoints.ct_height.values),
                    )
                    self.endpoints.insert(loc=1, column="co2", value=co2_values)
            else:
                co2_values = self.ct_data.co2.isel(
                    time=xr.DataArray(self.endpoints.ct_time.values),
                    latitude=xr.DataArray(self.endpoints.ct_latitude.values),
                    longitude=xr.DataArray(self.endpoints.ct_longitude.values),
                    level=xr.DataArray(self.endpoints.ct_height.values),
                )
                self.endpoints.insert(loc=1, column="co2", value=co2_values)

            return self.endpoints.co2.values

        def plot(
            self,
            ax: plt.Axes = None,
            id: list[int] = None,
            hours: int = None,
            add_map: bool = True,
            add_endpoints: bool = False,
            endpoint_kwargs: dict = dict(color="black", s=100),
            plot_station: bool = True,
            station_kwargs: dict = dict(color="black"),
            **kwargs,
        ) -> tuple[plt.Figure, plt.Axes]:

            if ax is None:
                fig, ax = self._outer.subplots()
            else:
                fig = ax.get_figure()
            if id is None:
                id = np.random.randint(0, self.dataframe.id.max())
            if isinstance(id, int) or isinstance(id, float):
                id = [id]

            plot_kwargs = dict(color="grey")
            plot_kwargs.update(kwargs)

            dataframe = self.dataframe.copy()
            if hours is not None:
                dataframe = dataframe[
                    dataframe.time >= dataframe.time.max() - np.timedelta64(hours, "h")
                ]

            for i in id:
                data = dataframe[dataframe["id"].values == i]
                lon = data.longitude
                lat = data.latitude
                ax.plot(lon, lat, **plot_kwargs)
                if add_endpoints:
                    if hours is None:
                        end = self.endpoints[self.endpoints["id"].values == i]
                        ax.scatter(end.longitude, end.latitude, **endpoint_kwargs)
                    else:
                        ax.scatter(lon.values[-1], lat.values[-1], **endpoint_kwargs)
            if plot_station:
                ax.scatter(
                    self._outer.release["lon1"],
                    self._outer.release["lat1"],
                    **station_kwargs,
                )
            if add_map:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                self._outer.add_map(ax)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            return fig, ax
