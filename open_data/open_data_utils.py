import copy
import json
import os
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Union

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from shapely.geometry import Polygon
from typing import Iterable, Any

# import geoplot
# import contextily as ctx


def to_tuple(convert_arg, convert_ind):
    def to_tuple_inner(function):
        def new_function(*args, **kwargs):
            args = list(args)
            for i, arg in enumerate(args):
                if i in convert_ind:
                    if not (isinstance(arg, int) or arg is None):
                        args[i] = tuple(arg)
            for i, (key, value) in enumerate(kwargs.items()):
                if key in convert_arg:
                    if not (isinstance(value, int) or value is None):
                        kwargs[key] = tuple(value)
            args = tuple(args)
            return function(*args, **kwargs)

        return new_function

    return to_tuple_inner


def val_to_list(types, val, expand_to=None):
    if not isinstance(type(types), list):
        types = [types]
    if type(val) in types:
        val = [val]
    val = list(val)
    if expand_to is not None and len(val) == 1:
        val = val * expand_to
    return val


def datetime64_to_yyyymmdd_and_hhmmss(time: np.datetime64) -> tuple[str, str]:
    string = str(time)
    string = string.replace("-", "").replace(":", "")
    date, time = string.split("T")
    return date, time


def printval(inp):
    out = {}
    exec(f"val={inp}", globals(), out)
    out = out["val"]
    print(f"{inp}: {out}")


def load_era_to_gdf(file, data_keys=["u", "v", "w"], coarsen=None):
    """
    Returns a geodataframe with geomertry Point(long, lat) and columns: ["time", "level"] + data_keys.

        Parameters:
            file (str): Path to nc file
            data_keys (list): List of str naming wind field coordinates to extract
            coarsen (int/list): If int is given longitude and latitude are averaged over with neighbors according to int.
                                If list is given as [x, y] then longitude (x) and latitude (y) are averaged over differnent numbers of neighbors.

        Returns:
            geodataframe (gpd.GeoDataFrame): Frame of era data
    """
    #
    assert coarsen is None or type(coarsen) in (int, list)
    coarsen = [coarsen, coarsen] if type(coarsen) == int else coarsen

    # load file into xarray
    dataset = xr.open_mfdataset(file, combine="by_coords", concat_dim="None")
    # center coordinates
    dataset.longitude.values = (dataset.longitude.values + 180) % 360 - 180
    # extract u,v,w values
    if coarsen is not None:
        data_dict = {
            key: dataset[key]
            .coarsen(longitude=coarsen[0], latitude=coarsen[1], boundary="trim")
            .mean()
            for key in data_keys
        }
        # data_dict = {key: dataset[key][::100,...].coarsen(longitude=coarsen[0], latitude=coarsen[1], boundary='trim').mean() for key in data_keys}
    else:
        data_dict = {key: dataset[key] for key in data_keys}

    # convert to frames and remove unnecessary columns
    frame_list = [data_dict[key].to_dataframe(key).reset_index() for key in data_keys]
    for i in range(len(frame_list)):
        if i > 0:
            frame_list[i] = frame_list[i].drop(
                ["time", "level", "longitude", "latitude"], axis=1
            )
    # merge u,v,w into one pd.DataFrame
    dataframe = pd.concat(frame_list, axis=1)
    # use merge instead
    # build GeoDataFrame
    geodataframe = gpd.GeoDataFrame(
        dataframe[["time", "level"] + data_keys],
        geometry=gpd.points_from_xy(dataframe.longitude, dataframe.latitude),
        crs="EPSG:4326",
    )

    return geodataframe


def plot_hwind_field(
    fig, ax, gdf, time, level, normalize=False, plot_range=None, use_cmap=True, **kwargs
):
    """
    Returns figure and axis of plot of horizonal wind field.

        Parameters:
            fig (Figure): Figure to plot to
            ax (Axes): Axes to plot to
            gdf (gpd.GeoDataFrame): Frame of era data
            time (str): Date in format "YYYY-MM-DD"
            level (float): Height
            normalize (bool): Whether to normalize the vectors or not
            plot_range (list/nd.array): Specifiy rectangle to be plottet in format [[x1,x2],[y1,y2]]
            use_cmap (bool): Specify if length should be used for cmap
            **kwargs: for quiver

        Returns:
            fig (Figure): Figure with plot
            ax (Axes): Axes of quiver plot
    """
    ax.set_title("Horizontal wind field ({} at {} hPa)".format(time, level))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if plot_range is not None:
        gdf = gdf.cx[
            plot_range[0][0] : plot_range[0][1], plot_range[1][0] : plot_range[1][1]
        ]

    sel_gdf = gdf.loc[(gdf["time"] == time) & (gdf["level"] == level)]
    # sel_gdf = sel_gdf.loc[]
    field = np.array([sel_gdf["u"].values, sel_gdf["v"].values])
    long = sel_gdf.geometry.x
    lat = sel_gdf.geometry.y

    norm = np.linalg.norm(field, axis=0)
    if normalize:
        field = field / norm
    if use_cmap:
        c = ax.quiver(long, lat, *field, norm, **kwargs)
    else:
        c = ax.quiver(long, lat, *field, **kwargs)

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Wind velocity [m/s]")
    return fig, ax


def add_world_map(fig, ax, plot_range=None, country=None, **kwargs):
    """Add world map to ax of fig

    Args:
        fig (Figure): Figure to map on
        ax (Axes): Axes to map on
        plot_range (list/nd.array): Specifiy rectangle to be plottet in format [[x1,x2],[y1,y2]]
        **kwargs: for world plot
    Returns:
        fig (Figure): Figure with plot
        ax (Axes): Axes with world map
    """
    if plot_range is not None:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).cx[
            plot_range[0][0] : plot_range[0][1], plot_range[1][0] : plot_range[1][1]
        ]
    else:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    if country is not None:
        world = world[world.name == "Australia"]

    world.plot(ax=ax, **kwargs)
    return fig, ax


def plot_hwind_field_season(
    fig,
    ax,
    gdf,
    season,
    hemisphere,
    year,
    level,
    normalize=False,
    plot_range=None,
    use_cmap=True,
    **kwargs,
):
    """
    Returns figure and axis of plots of horizonal wind fields averaged over season months and years. (Winter in northern hemisphere for e.g 2009 starts 12.2009 and ends 02.2010)

        Parameters:
            fig (Figure): Figure to plot to
            ax (Axes or list of Axes): Axes(') to plot to
            gdf (gpd.GeoDataFrame): Frame of era data
            season (string or list of strings): seasons to be plotted or list of seasons
            heisphere (string): Either "north" or "south" for correct seasons for \
                respective hemisphere
            year (int or list of ints): year(s) to use for averaging 
            level (float): Height
            normalize (bool): Whether to normalize the vectors or not
            plot_range (list/nd.array): Specifiy rectangle to be plottet in format [[x1,x2],[y1,y2]] 
            use_cmap (bool): Specify if length should be used for cmap 
            **kwargs: for plt.quiver

        Returns:
            fig (Figure): Figure with plot
            ax (Axes): Axes of quiver plot
    """
    assert hemisphere in ["north", "south"]
    if hemisphere == "north":
        season_dict = {
            "summer": ["06", "07", "08"],
            "autumn": ["09", "10", "11"],
            "winter": ["12", "01", "02"],
            "spring": ["03", "04", "05"],
        }
    elif hemisphere == "south":
        season_dict = {
            "summer": ["12", "01", "02"],
            "autumn": ["03", "04", "05"],
            "winter": ["06", "07", "08"],
            "spring": ["09", "10", "11"],
        }

    ax = [ax] if type(ax) not in [list, np.ndarray] else ax
    season = [season] if type(season) is not list else season

    for i, a in enumerate(ax):
        a.set_title(
            "Horizontal wind field in {} ({}ern hemisphere) at {} hPa, {} ".format(
                season[i], hemisphere, level, year
            )
        )
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")

    if plot_range is not None:
        gdf = gdf.cx[
            plot_range[0][0] : plot_range[0][1], plot_range[1][0] : plot_range[1][1]
        ]

    for i, s in enumerate(season):
        s_months = season_dict[s]
        print(
            [
                str(y if m not in ["01", "02"] else y + 1) + "-" + m + "-01"
                for y in year
                for m in s_months
            ]
        )
        s_data = [
            gdf.loc[
                (
                    gdf["time"]
                    == str(y if m not in ["01", "02"] else y + 1) + "-" + m + "-01"
                )
                & (gdf["level"] == level)
            ]
            for y in year
            for m in s_months
        ]

        long = s_data[0].geometry.x
        lat = s_data[0].geometry.y

        s_data = pd.concat(s_data)
        s_data = s_data.groupby(["geometry"], sort=False).mean()

        field = np.array([s_data["u"].values, s_data["v"].values])
        norm = np.linalg.norm(field, axis=0)

        if normalize:
            field = field / norm
        if use_cmap:
            c = ax[i].quiver(long, lat, *field, norm, **kwargs)
            cbar = fig.colorbar(c, ax=ax[i])
            cbar.set_label("Wind velocity [m/s]")
        else:
            c = ax[i].quiver(long, lat, *field, **kwargs)

    return fig, ax


def plot_hwind_field_month(
    fig,
    ax,
    gdf,
    month,
    year,
    level,
    normalize=False,
    plot_range=None,
    use_cmap=True,
    **kwargs,
):
    """
    Returns figure and axis of plots of horizonal wind fields averaged over season months and years. 

        Parameters:
            fig (Figure): Figure to plot to
            ax (Axes or list of Axes): Axes(') to plot to
            gdf (gpd.GeoDataFrame): Frame of era data \n
            month (int/string or list or list of lists): Inner list: months to be averaged over,\
                outer lists, sets for different plots
            year (int or list of ints): year(s) to use for averaging 
            level (float): Height
            normalize (bool): Whether to normalize the vectors or not
            plot_range (list/nd.array): Specifiy rectangle to be plottet in format [[x1,x2],[y1,y2]] 
            use_cmap (bool): Specify if length should be used for cmap 
            **kwargs: for plt.quiver

        Returns:
            fig (Figure): Figure with plot
            ax (Axes): Axes of quiver plot
    """
    month = [month] if type(month) is not list else month
    month = [[m for m in month]] if type(month[0]) is not list else month
    for i, l in enumerate(month):
        for j, m in enumerate(l):
            month[i][j] = str(m)
            month[i][j] = "0" + month[i][j] if len(month[i][j]) == 1 else month[i][j]
    ax = [ax] if type(ax) not in [list, np.ndarray] else ax

    for i, a in enumerate(ax):
        a.set_title(
            "Horizontal wind field (months: {}, years: {}, level: {} hPa)".format(
                month[i], year, level
            )
        )
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")

    if plot_range is not None:
        gdf = gdf.cx[
            plot_range[0][0] : plot_range[0][1], plot_range[1][0] : plot_range[1][1]
        ]

    for i, months in enumerate(month):
        m_data = [
            gdf.loc[(gdf["time"] == str(y) + "-" + m + "-01") & (gdf["level"] == level)]
            for y in year
            for m in months
        ]

        long = m_data[0].geometry.x
        lat = m_data[0].geometry.y

        m_data = pd.concat(m_data)
        m_data = m_data.groupby(["geometry"], sort=False).mean()

        field = np.array([m_data["u"].values, m_data["v"].values])
        norm = np.linalg.norm(field, axis=0)

        if normalize:
            field = field / norm
        if use_cmap:
            c = ax[i].quiver(long, lat, *field, norm, **kwargs)
            cbar = fig.colorbar(c, ax=ax[i])
            cbar.set_label("Wind velocity [m/s]")
        else:
            c = ax[i].quiver(long, lat, *field, **kwargs)

    """cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Wind velocity [m/s]")"""
    return fig, ax


def avg_by_time(dataset, avg_index=2, err_mode="std_err"):
    """Extract data form xarray of e.g. obspack data and avereage over  year/month/days.. according to avg index 1/2/3... respectively

    Args:
        dataset (xarray.Dataset): Dataset to unpack and average.
        avg_index (int, optional): Specification over time to average over (years/months/days.. according to avg index 1/2/3... respectively). Defaults to 2.
        err_mode (str): Choose from: "std_err" (for calculation from data error), "std_dev" (for std of averaged data). Defaults to "std_err".

    Returns:
        numpy.ndarray: averaged CO2 values
        numpy.ndarray: standard error of values
        numpy.ndarray with numpy.datetime64: times of averages (month and year for avg_index=2)
    """
    keys = ["Y", "M", "D", "h", "m", "s"]
    v_list = []
    err_list = []
    t_list = []
    t, ind = np.unique(
        dataset.time_components.values[:, :avg_index], axis=0, return_index=True
    )
    err_flag = False
    for i in range(len(t) - 1):
        err = None

        if err_mode == "std_err":
            try:
                err = np.mean(
                    dataset.value_std_dev.values[ind[i] : ind[i + 1]]
                ) / np.sqrt(ind[i + 1] - ind[i])
            except AttributeError:
                err_flag = True
        elif err_mode == "std_dev":
            err = np.std(dataset.value.values[ind[i] : ind[i + 1]])

        err_list.append(err)
        t_list.append(
            dataset.time.values[ind[i]].astype(
                "datetime64[{}]".format(keys[avg_index - 1])
            )
        )
        v = np.mean(dataset.value.values[ind[i] : ind[i + 1]])
        v_list.append(v)
    print(f"No error avaliable for data.") if err_flag else None
    return np.array(v_list), np.array(err_list), np.array(t_list)


def detrend_hawaii(values, times):
    """Detrend CO2 measurement data according to Mauna Loa measurements https://gml.noaa.gov/ccgg/trends/gl_data.html

    Args:
        values (np.ndarray): CO2 measurements in ppm
        times (np.datetime64): times of measurements

    Returns:
        np.ndarray: detrended values
    """
    # get data of ann means
    data = pd.read_csv(
        "/home/clueken/master/open_data/data/co2_annmean_gl.csv", ",", header=0
    )
    # extract time of means as YYYY-07-01
    years = data.year.to_numpy(dtype="str").astype("datetime64") + np.array(
        "06", dtype="timedelta64[M]"
    )
    # extract means
    mean = data["mean"].to_numpy()
    # find indices thar are closest to times of data
    ind = np.sort(
        np.argsort(np.abs(times[:, None] - years[None, :]), axis=-1)[:, :2], axis=-1
    )
    # calculate point of interpolation
    deltas = times - years[ind[:, 0].T]
    frac = deltas / np.timedelta64(1, "Y").astype(deltas.dtype)
    # do the interpolation
    interpolation = mean[ind[:, 0].T] + (mean[ind[:, 1].T] - mean[ind[:, 0].T]) * frac
    # subtract background
    det = values - interpolation
    return det


def merge_arange(times, values, min_time, max_time, type):
    """Set equally distanced times and sort values to respective slots in dataframe.

    Args:
        times (numpy.ndarray): times of data
        values (numpy.ndarray): data values
        min_time (str): date/time from which arange starts
        max_time (str): data/time at which arange ends
        type (str): steps (Y, M, D, ...)

    Returns:
        pandas.DataFrame: values organized in times of np.arange(min_time, max_time)
    """
    full_times = np.arange(min_time, max_time, dtype=f"datetime64[{type}]")
    df = pd.DataFrame({"times": full_times})

    times = times.astype(f"datetime64[{type}]")

    df_val = pd.DataFrame({"times": times, "values": values})
    df = df.merge(df_val, on="times", how="outer")
    return df


def xr_to_gdf(xarr, *data_variables, crs="EPSG:4326"):
    """Convert xarray.DataArray to GeoDataFrame

    Args:
        xarr (DataArray): To be converted
        *data_variables (str): data variables to be transferred to GeoDataFrame
        crs (str, optional): Coordinate reference system for GeoDataFrame. Defaults to "EPSG:4326".

    Returns:
        GeoDataFrame: Output
    """
    df = xarr.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
        df[data_variables[0]],
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=crs,
    )
    for i, dv in enumerate(data_variables):
        if i == 0:
            continue
        gdf[dv] = df[dv]
    return gdf


def add_country_names(gdf):
    """Add country names to GeaDataFrame

    Args:
        gdf (GeoDataFrame): to add country names to

    Returns:
        GeoDataFrame: with added names
    """
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = gpd.GeoDataFrame(world[["name"]], geometry=world.geometry)
    gdf = gpd.sjoin(gdf, world, how="left").drop(columns=["index_right"])
    return gdf


def country_intersections(gdf, country, crs="EPSG:4326"):
    """Cut GeoDataFrame w.r.t. one country into country, , not country (rest), other countries and ocean

    Args:
        gdf (GeoDataFrame): GeoDataFrame to split up
        country (str): Country to single out
        crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".

    Returns:
        dict: Dict of GeoDataFrames
    """
    assert (
        country in gdf.name.values
    ), f"No country found with name {country} in GeoDataFrame"
    ov_count = gdf[gdf.name == country]
    ov_rest = gdf[gdf.name != country]
    ov_other = gdf[(gdf.name != country) & (gdf.name.notnull())]
    ov_sea = gdf[gdf.name.isnull()]
    ret = dict(country=ov_count, rest=ov_rest, other_countries=ov_other, sea=ov_sea)
    return ret


def select_extent(xarr, lon1, lon2, lat1, lat2):
    """Select extent of xarray.DataArray with geological data

    Args:
        xarr (xarray.DataArray): ...returns
        lon1 (float): left
        lon2 (float): right
        lat1 (float): lower
        lat2 (float): upper

    Returns:
        xarray.DataArray: cut xarray
    """
    xarr = xarr.isel(longitude=(lon1 <= xarr.longitude) * (xarr.longitude <= lon2))
    xarr = xarr.isel(latitude=(lat1 <= xarr.latitude) * (xarr.latitude <= lat2))
    return xarr


def load_ct_data(
    ct_file: str,
    startdate: Union[np.datetime64, datetime],
    enddate: Union[np.datetime64, datetime],
) -> xr.DataArray:
    first = True
    for date in pd.date_range(startdate, enddate):
        ct_single_file = (
            ct_file
            + str(date.year)
            + str(date.month).zfill(2)
            + str(date.day).zfill(2)
            + ".nc"
        )
        ct_flux_day = xr.open_mfdataset(
            ct_single_file,
            combine="by_coords",
            drop_variables="time_components",
            chunks="auto",
        )
        if first:
            first = False
            ct_flux = ct_flux_day
        else:
            ct_flux = xr.concat([ct_flux, ct_flux_day], dim="time")
    # sum flux components:
    ct_flux = (
        ct_flux.bio_flux_opt
        + ct_flux.ocn_flux_opt
        + ct_flux.fossil_flux_imp
        + ct_flux.fire_flux_imp
    )
    ct_flux.name = "total_flux"

    # can be deleted when footprint has -179.5 coordinate
    ct_flux = ct_flux[:, :, 1:]
    return ct_flux


def get_fp_array(fp_dataset: xr.Dataset) -> xr.DataArray:
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        fd_array = fp_dataset.spec001_mr[0, :, :, 0, :, :]
    dt = np.timedelta64(90, "m")
    fd_array = fd_array.assign_coords(time=(fd_array.time + dt))
    # flip time axis
    fd_array = fd_array.sortby("time")
    return fd_array


def combine_flux_and_footprint(
    flux: xr.DataArray, footprint: xr.DataArray, chunks: dict = None
) -> xr.Dataset:
    if chunks is not None:
        footprint = footprint.chunk(chunks=chunks)
        flux = flux.chunk(chunks=chunks)
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        fp_co2 = xr.merge([footprint, flux])
    fp_co2 = fp_co2.persist()
    # 1/layer height*flux [mol/m²*s]*fp[s] -> mol/m³ -> kg/m³
    fp_co2 = (
        1 / 100 * fp_co2.total_flux * fp_co2.spec001_mr * 0.044
    )  # dim : time, latitude, longitude, pointspec: 36
    # sum over time, latitude and longitude, remaining dim = layer
    fp_co2 = fp_co2.sum(dim=["time", "latitude", "longitude"])
    fp_co2 = fp_co2.compute()
    fp_co2.name = "CO2"
    fp_co2 = fp_co2.to_dataset()
    return fp_co2


def calc_enhancement(
    fp_data: xr.Dataset,
    ct_file: str,
    startdate: Union[np.datetime64, datetime],
    enddate: Union[np.datetime64, datetime],
    boundary: list[float, float, float, float] = None,
    chunks: dict = None,
) -> np.ndarray:

    # test the right order of dates
    assert enddate > startdate, "The startdate has to be before the enddate"

    # read CT fluxes (dayfiles) and merge into one file
    ct_flux = load_ct_data(ct_file, startdate, enddate).compute()

    # from .interp can be deleted as soon as FP dim fits CTFlux dim, layers repeated???? therfore only first 36
    footprint = get_fp_array(fp_data).compute()
    # selct times of Footprint in fluxes
    ct_flux = ct_flux.sel(time=slice(footprint.time.min(), footprint.time.max()))
    if boundary is not None:
        ct_flux = select_extent(ct_flux, *boundary)

    # OLDVERSION pressure_weights = 1/(p_surf-0.1)*(fp_pressure_layers[0:-1]-fp_pressure_layers[1:])
    # FP_pressure_layers = np.array(range(numLayer,0,-1))*1000/numLayer
    fp_co2 = combine_flux_and_footprint(ct_flux, footprint, chunks=chunks)

    molefractions = fp_co2.CO2.values
    return molefractions * 1e6


def load_nc_partposit(dir_path: str, chunks: dict = None) -> xr.Dataset:
    files = []
    for file in os.listdir(dir_path):
        if "partposit" in str(file) and ".nc" in str(file):
            files.append(os.path.join(dir_path, file))
    xarr = xr.open_mfdataset(files, chunks=chunks)
    return xarr


class FlexDataset2:
    def __init__(self, directory: str, **kwargs):
        nc_path = self.get_nc_path(directory)
        self._nc_path = nc_path
        self._dir = nc_path.rsplit("/", 1)[0]
        self.sounding = None
        self._kwargs = dict(
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

        self.footprint = self.Footprint(self)
        if os.path.exists(os.path.join(self._dir, "trajectories.pkl")):
            self.trajectories = self.Trajectories(self)
        else:
            print("No trajectory information found.")
            self.trajectories = None
        self.start, self.stop, self.release = self.get_metadata()
        self._background = None
        self._background_layer = None
        self._background_inter = None
        self._enhancement = None
        self._enhancement_layer = None
        self._enhancement_inter = None
        
        self._last_boundary = None

    def get_nc_path(self, directory: str) -> str:
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
            [cf.STATES, dict(alpha=0.1)],
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
        print(f"mode = {mode}")
        return mode


    def enhancement_layer(
        self,
        ct_file: str = None,
        boundary: list[float, float, float, float] = None,
        allow_read: bool = True,
        force_calculation: bool = False,
        chunks: dict = dict(time=10)
        ) -> list[float]:
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
    ) -> float:
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
                molefractions = xr.DataArray(
                    molefractions, 
                    coords = dict(pointspec = self.footprint.dataset.pointspec.values),
                    dims = dict(pointspec = self.footprint.dataset.pointspec),
                    name = "enhancement_layer"
                    )
                pressures = self.sounding.pressure_levels.values[0,-1] * self.pressure_factor(self.release["heights"], Tb=273.15 + 22)
                molefractions = self.interpolate_to_acos_levels(molefractions, "pointspec", pressures, self.sounding)
                molefractions = self.add_acos_variables(molefractions, self.sounding)
                self._enhancement_inter = self.pressure_weighted_sum(molefractions, "enhancement_layer", with_averaging_kernel = False).values
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
                co2_means = xr.DataArray(
                    co2_means, 
                    coords = dict(pointspec = self.footprint.dataset.pointspec.values),
                    dims = dict(pointspec = self.footprint.dataset.pointspec),
                    name = "background_layer"
                    )
                pressures = self.sounding.pressure_levels.values[0,-1] * self.pressure_factor(self.release["heights"], Tb=273.15 + 22)
                co2_means = self.interpolate_to_acos_levels(co2_means, "pointspec", pressures, self.sounding)
                co2_means = self.add_acos_variables(co2_means, self.sounding)
                self._background_inter = self.pressure_weighted_sum(co2_means, "background_layer", with_averaging_kernel=True).values
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

    @staticmethod
    def interpolate_to_acos_levels(dataarray: xr.DataArray, pressure_key: str, pressure_values: Iterable, acos_data: xr.Dataset) -> xr.DataArray:
        dataarray = dataarray.assign_coords({pressure_key: pressure_values}).rename({pressure_key: "pressure"})
        pressure_sounding = acos_data.pressure_levels.values[0]
        #find possible levels to interpolate to
        within_sounding_range = (pressure_sounding > dataarray.pressure.values.min()) & (pressure_sounding < dataarray.pressure.values.max())
        #get respective level information
        pressure_sounding = pressure_sounding[within_sounding_range]
        levels_sounding= acos_data.levels.values[within_sounding_range]
        #interpolation to acos levels
        dataarray = dataarray.interp(pressure=pressure_sounding)
        #assign levels to pressures
        dataarray = dataarray.assign_coords({"pressure": levels_sounding}).rename({"pressure": "levels"})
        dataarray = dataarray.compute()
        return dataarray.squeeze(drop=True)

    @staticmethod
    def add_acos_variables(
        dataarray: xr.DataArray, 
        acos_data: xr.Dataset, 
        variables: list[str]= ["pressure_weight", "xco2_averaging_kernel", "co2_profile_apriori"]
    ) -> xr.Dataset:

        acos_variable_data = [acos_data[var].squeeze(drop=True) for var in variables]
        dataset = xr.merge([dataarray.squeeze(drop=True), *acos_variable_data])
        return dataset
    
    @staticmethod
    def pressure_weighted_sum(dataset: xr.Dataset, data_var: str, with_averaging_kernel: bool = True):
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

    def delete_results(self, names=["enhancement", "enhancement_inter", "enhancement_layer", "background", "background_inter", "background_layer"]):
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

    # def get_pressures(self, heights, lon, lat):
    #     pressures = self.pressure_factor(heights)
    #     return pressures

    def load_sounding(self, path: Union[Path, str]) -> xr.Dataset:
        """Loads acos sounding of FLEXPART run

        Args:
            path (Union[Path, str]): Path of Acos file unitl timestamp. E.g.: '/path/to/acos/folder/acos_LtCO2_'

        Returns:
            xr.Dataset: part of loaded file containing the exact sounding
        """
        path = Path(path)
        sounding_datetime = np.datetime64(self.release["start"])
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
            sounding_id=acos_data.time.values.astype("datetime64[s]")
            == sounding_datetime.astype("datetime64[s]")
        )
        self.sounding = acos_data
        return self.sounding

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
            self._outer = outer
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

        def save_total(self):
            """Saves Footprints

            Args:
                include_sums (bool, optional): _description_. Defaults to True.
            """
            sum_path = os.path.join(self._dir, "Footprint_total.nc")
            self._total.to_netcdf(sum_path)

        def calc_total(self):
            """Calculates total Footprints"""
            self._total = None
            self._total = self.dataarray.sum(dim=["time", "pointspec"]).compute()

        def load_total(self):
            """Loads Footprints from directory of DataSet data"""
            self._total = None
            path = os.path.join(self._dir, "Footprint_total.nc")
            self._total = xr.load_dataarray(path)

        def total(self):
            """Get footprints from either loading of calculation"""
            if self._total is None:
                try:
                    self.load_total()
                except FileNotFoundError:
                    self.calc_total()
                    self.save_total()
            return self._total

        def plot(
            self,
            ax: plt.Axes = None,
            time: list[int] = None,
            pointspec: list[int] = None,
            plot_func: str = None,
            plot_station: bool = True,
            add_map: bool = True,
            station_kwargs: dict = dict(color="black"),
            **kwargs,
        ) -> tuple[plt.Figure, plt.Axes]:

            plot_kwargs = self._plot_kwargs.copy()
            plot_kwargs.update(kwargs)
            if ax is None:
                fig, ax = self._outer.subplots()
            else:
                fig = ax.get_figure()
            footprint = self.sum(time, pointspec)
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
            self, time: list[int] = None, pointspec: list[int] = None
        ) -> xr.DataArray:
            footprint = None
            if time is None and pointspec is None:
                footprint = self.total()
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


def in_dir(path: str, string: str) -> bool:
    for file in os.listdir(path):
        if string in file:
            return True
    return False