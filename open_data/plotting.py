from curses import noecho
from lib2to3.pgen2.token import OP
from tkinter import E
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from pathlib import Path
from typing import Any, Union, Literal, Optional, Callable, Iterable
import os
import warnings
from functools import cache, partial
from copy import deepcopy
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter, FormatStrFormatter
from matplotlib import cm

from open_data_utils import FlexDataset2, detrend_hawaii
import bayesinverse

settings = dict()

path = "/work/bb1170/RUN/b381737/data/FLEXPART/sensitivity/partnums_unpacked/predictions_fixed.pkl"


def select_boundary(
    data: Union[xr.DataArray, xr.Dataset], 
    boundary: Optional[tuple[float, float, float, float]]
    ) -> Union[xr.DataArray, xr.Dataset]:
    if not boundary is None:
        data = data.isel(
            longitude = (data.longitude >= boundary[0]) * (data.longitude <= boundary[1]),
            latitude = (data.latitude >= boundary[2]) * (data.latitude <= boundary[3]) 
        )
    return data

class Sensitivity():
    def __init__(self, predictions_file: str, mode=0, data_variables: list[str]=["enhancement", "background", "xco2"]):
        self.predictions_file = predictions_file
        self.predictions = pd.read_pickle(predictions_file)
        if mode == 0:
            self.data = self._get_prediction_data(data_variables) 
        elif mode == 1:
            self.data = self._get_prediction_data_legacy()
        elif mode == 2:
            self.data = self._get_prediction_data_legacy2()

    def _get_prediction_data(self, data_variables: list[str]=["enhancement", "background", "xco2"]):
        data = dict()
        data_setup = dict(darwin = dict(unit = [], pressure = []), wollongong = dict(unit = [], pressure = []))
        data["particles"] = deepcopy(data_setup)
        data["directories"] = deepcopy(data_setup)
        for variable in data_variables:
            data[variable] = deepcopy(data_setup)
        
        for i in range(len(self.predictions)):
            pred = self.predictions.iloc[i]
            part = int(pred.directory.rsplit("k",1)[0][-3:])
            city = 'wollongong' if pred.directory[-1] == "0" else 'darwin' 
            typ = 'unit' if 'unit' in pred.directory else 'pressure'
            for variable in data_variables:
                value = pred[variable]
                data[variable][city][typ].append(value)
            data["particles"][city][typ].append(part)
            data["directories"][city][typ].append(pred.directory)

        for city in ["darwin", "wollongong"]:         
            for typ in ["unit", "pressure"]:
                ids = np.argsort(data["particles"][city][typ])
                data["particles"][city][typ] = np.array(data["particles"][city][typ])[ids]
                data["directories"][city][typ] = np.array(data["directories"][city][typ])[ids]
                for variable in data_variables:
                    data[variable][city][typ] = np.array(data[variable][city][typ])[ids]
        return data
    
    def _get_prediction_data_legacy(self):
        data_variables = ["enhancement"]
        data = dict()
        data_setup = dict(darwin = dict(unit = [], pressure = []), wollongong = dict(unit = [], pressure = []))
        data["particles"] = deepcopy(data_setup)
        data["directories"] = deepcopy(data_setup)
        for variable in data_variables:
            data[variable] = deepcopy(data_setup)
        
        for i in range(len(self.predictions)):
            pred = self.predictions.iloc[i]
            part = int(pred.directory.rsplit("k",1)[0][-3:])
            city = 'wollongong' if "Wollongong" in pred.directory else 'darwin' 
            typ = 'unit' if 'wu' in pred.directory else 'pressure'
            for variable in data_variables:
                value = pred[variable]
                data[variable][city][typ].append(value)
            data["particles"][city][typ].append(part)
            data["directories"][city][typ].append(pred.directory)

        for city in ["darwin", "wollongong"]:         
            for typ in ["unit", "pressure"]:
                ids = np.argsort(data["particles"][city][typ])
                data["particles"][city][typ] = np.array(data["particles"][city][typ])[ids]
                data["directories"][city][typ] = np.array(data["directories"][city][typ])[ids]
                for variable in data_variables:
                    data[variable][city][typ] = np.array(data[variable][city][typ])[ids]
        return data

    def _get_prediction_data_legacy2(self):
        data_variables = ["enhancement"]
        data = dict()
        data_setup = dict(darwin = dict(unit = [], pressure = []), wollongong = dict(unit = [], pressure = []))
        data["particles"] = deepcopy(data_setup)
        data["directories"] = deepcopy(data_setup)
        for variable in data_variables:
            data[variable] = deepcopy(data_setup)
        
        for i in range(len(self.predictions)):
            pred = self.predictions.iloc[i]
            part = int(pred.directory.rsplit("k",1)[0][-3:])
            city = 'wollongong' if "wollongong" in pred.directory else 'darwin' 
            typ = 'unit' if 'wu' in pred.directory else 'pressure'
            for variable in data_variables:
                value = pred[variable]
                data[variable][city][typ].append(value)
            data["particles"][city][typ].append(part)
            data["directories"][city][typ].append(pred.directory)

        for city in ["darwin", "wollongong"]:         
            for typ in ["unit", "pressure"]:
                ids = np.argsort(data["particles"][city][typ])
                data["particles"][city][typ] = np.array(data["particles"][city][typ])[ids]
                data["directories"][city][typ] = np.array(data["directories"][city][typ])[ids]
                for variable in data_variables:
                    data[variable][city][typ] = np.array(data[variable][city][typ])[ids]
        return data

    def plot_partnum_vs_variable(
        self, 
        variable: str, 
        style: str, 
        band_style: str = "percent",
        band_vals: list[float] = None, 
        band_colors: list[str] = ["orange", "red"],
        ax: plt.Axes = None,
        cities: list[str] = ["darwin", "wollongong"], 
        markers: list[str] = ["*", "x"], 
        types: list[str] = ["unit", "pressure"], 
        coloring: list[str] = ["blue", "red"],
        figsize: tuple[int, int] = None
        ) -> tuple[plt.Figure, plt.Axes]:
        """Function to plot enhancement background or xco2 values in different ways.

        Args:
            variable (str): Variable to be plottet (enhancement, background or xco2)
            style (str): Choose from standard, difference, difference_norm, difference_end, difference_end_norm, for absolute values, differences to mean or last value(_end) or difference devided by mean value
            band_style (str, optional): Style of error band. Choose from percent and absolute. Defaults to "percent".
            band_vals (list[float], optional): Values for radius of bands. Defaults to None.
            band_colors (list[str], optional): Colors for each band. Defaults to ["orange", "red"].
            ax (plt.Axes, optional): Axes to plot on. An Axex will be initialized of none given. Defaults to None.
            cities (list[str], optional): List of cities to plot. Choose from darwin and wollongong Defaults to ["darwin", "wollongong"].
            markers (list[str], optional): List of markers for each citiy. Defaults to ["*", "x"].
            types (list[str], optional): List of types of releases to plot. Defaults to ["unit", "pressure"].
            coloring (list[str], optional): Colors for each type. Defaults to ["blue", "red"].
            figsize (tuple[int, int], optional): Siye of figure if newly initialized. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes with plot
        """        
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        for i, city in enumerate(cities):
            marker = markers[i]
            for j, typ in enumerate(types):
                color = coloring[j]
                v = self.data[variable][city][typ]
                p = self.data["particles"][city][typ]
                title = None

                v0 = v
                bar_center = v.mean()

                if style == "standard":
                    title = f"Comparison of absolute value of {variable}"
                    ylabel = f"{variable} [ppm]"
                    v = v
                elif style == "difference":
                    title = f"Comparison of difference to mean of {variable}"
                    ylabel = f"Difference to mean [ppm]"
                    v = v - np.mean(v)
                    bar_center = 0
                elif style == "difference_norm":
                    title = f"Comparison of normalized difference of mean \n of {variable}"
                    ylabel = r"$\frac{value - mean}{mean}$"
                    v = (v - np.mean(v))/np.mean(v)*100
                    bar_center = 0
                elif style == "difference_end":
                    title = f"Comparison of difference to endpoint of {variable}"
                    ylabel = f"Difference to mean [ppm]"
                    v = v - v[-1]
                    bar_center = 0
                elif style == "difference_end_norm":
                    title = f"Comparison of normalized difference of endpoint \n of {variable}"
                    ylabel = r"$\frac{value - value[-1]}{mean}$"
                    v = (v - v[-1])/np.mean(v)*100
                    bar_center = 0
                ax.scatter(p, v, marker=marker, c=color, label=f"{typ} {city}")



                if not band_vals is None:
                    if len(cities) > 1 or len(types) > 1:
                        if band_style == "percent":
                            assert "norm" in style, "Multiple cities with error band_style='percent' only possible for styles with 'norm'"
                        if band_style == "absolute":
                            assert  not "norm" in style, "Multiple cities with error band_style='absolute' only possible for styles without 'norm'"
                        if i != len(cities)-1 or j != len(types)-1:
                            continue 
                    xlim = ax.get_xlim()
                    ax.hlines(bar_center, *xlim, color="grey", linestyle="dashed")
                    if band_style == "percent":
                        last_err = 0
                        for i, band_val in enumerate(band_vals):
                            if "norm" in style:
                                err = band_val
                            else: 
                                err = v0.mean() * band_val/100
                            ax.fill_between([*xlim], bar_center + last_err, bar_center + err, color=band_colors[i], alpha=0.3, label=f"{band_val} % error band")
                            ax.fill_between([*xlim], bar_center - err, bar_center - last_err, color=band_colors[i],alpha=0.3)
                            ax.hlines(bar_center + err, *xlim, color=band_colors[i], linestyle="dashed")
                            ax.hlines(bar_center - err, *xlim, color=band_colors[i], linestyle="dashed")
                            last_err = err
                        ax.set_xlim(*xlim)
                    
                    if band_style == "absolute":
                        last_err = 0
                        for i, band_val in enumerate(band_vals):
                            if "norm" in style:
                                err = band_val/v0.mean()
                            else:
                                err = band_val
                            ax.fill_between([*xlim], bar_center + last_err, bar_center + err, color=band_colors[i], alpha=0.3, label=f"{band_val} ppm error band")
                            ax.fill_between([*xlim], bar_center - err, bar_center - last_err, color=band_colors[i],alpha=0.3)
                            ax.hlines(bar_center + err, *xlim, color=band_colors[i], linestyle="dashed")
                            ax.hlines(bar_center - err, *xlim, color=band_colors[i], linestyle="dashed")
                            last_err = err
                        ax.set_xlim(*xlim)
                        


        ax.set_xlabel(r"Particle Number [$10^3$]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return fig, ax

    @cache
    def calc_footprint_sum_vs_release_height(self, city: str, typ: str, file_ind: int=-1):
        """Calculation sum of footprints from 0 to each height value in the

        Args:
            city (str): 
            typ (str): _description_
            file_ind (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """        
        file = str(self.data["directories"][city][typ][file_ind])
        fd = FlexDataset2(file)
        heights = (fd.footprint.dataset.RELZZ2.values + fd.footprint.dataset.RELZZ1)/2
        heights = heights.compute()
        fp_sum = []
        for i in fd.footprint.dataset.pointspec.values:
            fp_sum.append(fd.footprint.dataarray.sel(pointspec=slice(0,i)).sum().compute())
        return heights, fp_sum

    def footprint_sum_vs_release_height(
        self, 
        city: str, 
        typ: str, 
        file_ind: int=-1, 
        ax: plt.Axes=None, 
        figsize: tuple[int, int] = None, 
        **kwargs
        ) -> tuple[plt.Figure, plt.Axes]:

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        heights, fp_sum = self.calc_footprint_sum_vs_release_height(city, typ, file_ind)
        fp_sum_arr = np.array(fp_sum)
        fp_sum_arr = fp_sum_arr/max(fp_sum_arr)*100
        ax.plot(heights, fp_sum_arr, **kwargs)
        ax.grid()
        ax.set_xlabel("height [m]")
        ax.set_ylabel("Percentage of included footprint [%]")
        ax.set_title(f"Effect of release height ({city}, {typ})")
        return fig, ax

    def particles_leaving_domain(
        self, 
        city: str, 
        typ: str, 
        bins:int = 10,
        range: tuple|list = None,
        line: bool = True,
        file_ind: int=-1, 
        vline: int = 10,
        vline_kwargs: dict = dict(),
        border_list: list[float] = [0.75], 
        color_list: list[str] = ["grey"],
        linestyle_list: list[str] = ["dashed"],
        border_kwargs: dict = dict(),
        ax: plt.Axes=None, 
        figsize: tuple[int, int] = None, 
        fix_xaxis: bool = True,
        **kwargs
        ) -> tuple[plt.Figure, plt.Axes]: 
        
        default_vline_kwargs = dict(color="red", label=f"{vline} days after simulation begin")
        default_vline_kwargs.update(vline_kwargs)
        vline_kwargs = default_vline_kwargs

        hist_kwargs = dict()
        line_kwargs = dict()
        if line:
            line_kwargs.update(kwargs)
        else:
            hist_kwargs.update(kwargs)


        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        file = str(self.data["directories"][city][typ][file_ind])
        fd = FlexDataset2(file)
        fd.trajectories.load_endpoints()
        end = fd.trajectories.endpoints
        times = end.time.values
        times = np.sort(times)[::-1]
        values, counts, hist = ax.hist(times, cumulative=-1, density=True, bins=bins, range=range, **hist_kwargs)
        if line:
            hist.remove()
            ax.plot(((counts[1:] + counts[:-1])/2).astype("datetime64[D]"), values, **line_kwargs)
        if fix_xaxis:
            ax.invert_xaxis()
            ax.get_yaxis().set_major_formatter(PercentFormatter(1, symbol=""))
            # dates = ax.get_xticks().astype("datetime64[D]")
            # dates = abs(dates-max(dates)).astype(int)
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore")
            #     ax.set_xticklabels(dates)
            ax.tick_params(axis='x', labelrotation=45)

        ax.set_xlabel("Simulated time [days]")
        ax.set_ylabel("Amount of particles outside the domain [%]")
        ax.set_title(f"Particles that left domain over time ({city}, {typ})")
        
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        
        if not vline is None:
            ax.vlines(np.datetime64(fd.release["start"]) - np.timedelta64(vline, "D"), *ylim, **vline_kwargs)
            ax.set_ylim(*ylim)


        for i, border in enumerate(border_list):
            color = color_list[i] if len(color_list) == len(border_list) else color_list[0]
            linestyle = linestyle_list[i] if len(linestyle_list) == len(border_list) else linestyle_list[0]
            ax.hlines(border, *xlim, color=color, linestyle=linestyle, label=f"{border*100}%", **border_kwargs)
        ax.set_xlim(*xlim)
        #ax.legend()
        return fig, ax
        
Pathlike = Union[Path,str]

class ResultsConcentrations():
    def __init__(self, dirs: list[Pathlike], filenames: list[Pathlike] = ["results.pkl"]):
        self.dirs = [Path(dir) for dir in dirs]
        self.filenames = filenames if len(filenames) != 1 else filenames*len(self.dirs)
        self.predictions = self.get_predictions()

    def get_predictions(self) -> list[pd.DataFrame]:
        """Loads all prediction files into one dataframe"""
        predictions = []
        for dir, filename in zip(self.dirs, self.filenames):
            file = dir / filename
            predictions.append(pd.read_pickle(file))
        predictions = pd.concat(predictions).reset_index(drop=True)
        return predictions.sort_values("time")

    def mean(self, data, frame: str):
        if frame == "month":
            dt_val = "M"
        elif frame == "day":
            dt_val = "D"
        else:
            raise ValueError(f"Only acceptable values for frame: 'month', 'day'. Not {frame}")
        data = deepcopy(data)
        data.insert(1, frame, data.time.values.astype(f"datetime64[{dt_val}]"))
        data = data.groupby(frame, as_index=False).mean()
        return data

    def monthly_average_over_time(
        self, 
        ax: plt.Axes = None,
        interpolated: bool = True,
        ignore_defaults: bool = False,
        plot_measurement: bool = True,
        measurement_kwargs: dict = dict(),
        model_kwargs: dict = dict(),
        plot_enh: bool = True,
        enh_kwargs: dict = dict(),
        enh_pos_kwargs: dict = dict(),
        enh_neg_kwargs: dict = dict(),
        plot_bgd: bool = True,
        bgd_kwargs: dict = dict(),
        figsize: tuple[int, int] = (7,5),
        detrend: bool = False,
        ) -> tuple[plt.Figure, plt.Axes]:
        """PLots monthly average of xco2 prediction in comparison to measurement data. 

        Args:
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            interpolated (bool, optional): Wheter or not to use interpolated values (with averaging kernel). Defaults to True.
            ignore_defaults (bool, optional): Whether only kwargs set by user are to be used. Defaults to False.
            plot_measurement (bool, optional): Whether to plot measurement dataframe. Defaults to True.
            measurement_kwargs (dict, optional): Kwargs of measurement errorbar plot. Defaults to dict().
            model_kwargs (dict, optional): Kwargs of modeled xco2 values. Defaults to dict().
            plot_enh (bool, optional): Wether to plot enhancement data (fill_between). Defaults to True.
            enh_kwargs (dict, optional): Kwargs to use in both fill_between plots. Defaults to dict().
            enh_pos_kwargs (dict, optional): Kwargs for positive enhancement. Defaults to dict().
            enh_neg_kwargs (dict, optional): Kwargs for negative enhancement. Defaults to dict().
            plot_bgd (bool, optional): Wether to plot background. Defaults to True.
            bgd_kwargs (dict, optional): Kwargs for background errorbar plot. Defaults to dict().
            figsize (tuple[int, int], optional): Figsize. Defaults to (7,5).

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes with plot
        """        
    
        
        measurement_kw = dict()
        model_kw = dict()
        enh_kw = dict()
        enh_pos_kw = dict()
        enh_neg_kw = dict()
        bgd_kw = dict()

        measurement_defaults = dict(fmt="o", linestyle="-", color="firebrick", label="measurement") 
        model_defaults = dict(fmt="o", linestyle="-", color="black", label="Model")
        enh_defaults = dict(alpha=0.2)
        enh_pos_defaults = dict(color="green", label="positive_enhancement")
        enh_neg_defaults = dict(color="red", label="negative_enhancement")
        bgd_defaults = dict(color="black", linestyle="dashed", linewidth=1, label="Model background")

        if not ignore_defaults:
            measurement_kw.update(measurement_defaults)
            model_kw.update(model_defaults)
            enh_kw.update(enh_defaults)
            enh_pos_kw.update(enh_pos_defaults)
            enh_neg_kw.update(enh_neg_defaults)
            bgd_kw.update(bgd_defaults)

        measurement_kw.update(measurement_kwargs)
        model_kw.update(model_kwargs)
        enh_kw.update(enh_kwargs)
        enh_pos_kw.update(enh_pos_kwargs)
        enh_neg_kw.update(enh_neg_kwargs)
        bgd_kw.update(bgd_kwargs)
    

        xco2_variable = "xco2_inter" if interpolated else "xco2"
        background_variable = "background_inter" if interpolated else "background"
        enhancements_variable = "enhancement_diff" if interpolated else "enhancement"
        data = self.predictions[["time", xco2_variable, background_variable, enhancements_variable, "xco2_measurement"]]
        data = data.rename(columns={xco2_variable:"xco2", background_variable:"background", enhancements_variable:"enhancement"})
        keys = ["background","xco2", "xco2_measurement"]

        data = self.mean(data, "month").sort_values("month")
        xvals =  np.arange(len(data))

        if detrend:
            for i, key in enumerate(keys):
                data = detrend_hawaii(data, key, "month")
                keys[i] = key + "_detrend"
        
        bgd_key, xco2_key, measurement_key = keys
        enh_key = "enhancement"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        if plot_measurement:
            ax.errorbar(xvals, data[measurement_key], **measurement_kw)
        #ax.plot(xvals, data.xco2_measurement, color=measurement_color)
        ax.errorbar(xvals, data[xco2_key], **model_kw)
        #ax.plot(xvals, data.xco2, color=model_color, )

        #ylim = ax.get_ylim()

        
        if plot_bgd:
            ax.plot(xvals, data[bgd_key], **bgd_kw)
        if plot_enh:
            xvals_interp = np.linspace(xvals.min(), xvals.max(), 100)
            bgd_interp=np.interp(xvals_interp, xvals, np.array(data[bgd_key]))
            xco2_interp=np.interp(xvals_interp, xvals, np.array(data[xco2_key]))
            enh_interp=np.interp(xvals_interp, xvals, np.array(data[enh_key]))
            ax.fill_between(xvals_interp, bgd_interp, xco2_interp, where=enh_interp > 0, **enh_kw, **enh_pos_kw)
            ax.fill_between(xvals_interp, bgd_interp, xco2_interp, where=enh_interp < 0, **enh_kw, **enh_neg_kw)

        #ax.set_ylim(*ylim)
        ax.grid(True)
        ax.set_xticks(xvals, data.month.values.astype("datetime64[M]"))
        ax.set_xlabel("Month")
        ax.set_ylabel("XCO2 [ppm]")
        ax.set_title("XCO2 prediction (monthly average) - Model vs. measurement ")
        ax.legend()
        
        return fig, ax


    def measurement_correlation(
        self, 
        ax: plt.Axes = None,
        mean_over: str = "month",
        interpolated: bool = True,
        cmap: str = "jet",
        color_range: tuple[float, float] = (0,1),
        plot_diag: bool = True,
        diag_kwargs = dict(linestyle="dashed", color="grey"),
        figsize: tuple[int, int] = None,
        square: bool = False,
        detrend: bool = False,
        **kwargs
        ) -> tuple[plt.Figure, plt.Axes]:

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        detrend_str = ""
        xco2_variable = "xco2_inter" if interpolated else "xco2"
        data = self.predictions[["time", xco2_variable, "xco2_measurement"]]
        data = data.rename(columns={xco2_variable:"xco2"})
        keys = ["xco2", "xco2_measurement"]
        if mean_over is None:
            mean_over = "time"
        else:
            data = self.mean(data, mean_over)

        if detrend:
            detrend_str = " (detrended)"
            for i, key in enumerate(keys):
                data = detrend_hawaii(data, key, mean_over)
                keys[i] = key + "_detrend"
        
        xco2_key, measurement_key = keys

        n_months = len(np.unique(data[[mean_over]].values.astype("datetime64[M]")))
        cmap = cm.get_cmap(cmap)
        colors = cmap(np.linspace(*color_range, n_months))
        
        if mean_over != "month":
            data.insert(1, "month", data[[mean_over]].values.astype("datetime64[M]"))
        
        for month, color in zip(np.unique(data.month), colors):
            selection = data[data.month == month]
            ax.scatter(selection[[xco2_key]], selection[measurement_key], label=month.astype("datetime64[M]"), color=color, **kwargs)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if plot_diag:
            ax.plot((-1e2,1e3), (-1e2,1e3), **diag_kwargs)
        ax.axis("scaled")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if square:
            min_val = min([*xlim, *ylim])
            max_val = max([*xlim, *ylim])
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

        ax.set_xlabel(f"Modeled XCO2{detrend_str}[ppm]")
        ax.set_ylabel(f"measurement XCO2{detrend_str}[ppm]")
        ax.set_title(f"Correlation of measured vs. modeled XCO2 \nmonthly mean{detrend_str}")
        ax.grid()
        ax.legend()
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        return fig, ax

Boundary = Optional[tuple[float, float, float, float]]
Timeunit = Literal["week", "day"]
Coarsefunc = Union[Callable, tuple[Callable, Callable], str, tuple[str, str]]
Concentrationkey = Literal["background", "background_inter", "xco2", "xco2_inter"]

class Inversion():
    def __init__(
        self, 
        spatial_valriables: list[str],
        result_path: Pathlike,
        month: str, 
        flux_path: Pathlike, 
        time_coarse: Optional[int] = None, 
        coarsen_boundary: str = "pad",
        time_unit: Timeunit = "week",
        boundary: Boundary = None,
        concentration_key: Concentrationkey = "background_inter",
        data_outside_month: bool = False,
        bio_only: bool = False
    ):
        """Calculates inversion of footprints and expected concentrations and offers plotting possibilities.

        Args:
            result_path (Pathlike): Path of file of collected results (from script calc_results.py)
            month (str): String to specify month to load measurement and flux data from in format YYYY-MM
            flux_path (Pathlike): Directory and filename of data until timestamp e.g. /path/to/dir/CT2019B.flux1x1.
            time_coarse (Optional[int], optional): By how much new time coordinate should be coarsened. Defaults to None.
            coarsen_boundary (str, optional): Parameter for xr.DataArray.coarsen. Defaults to "pad".
            time_unit (Timeunit, optional): Time unit to group to. Defaults to "week".
            boundary (Boundary, optional): region of footprint to use. Defaults to None.
            concentration_key (Concentrationkey, optional): Data with which difference to measurement xco2 is calculated. Defaults to "background_inter".
            data_outdside_month (bool, optional): Wether to also use footprint data outsinde of target month. Defaults to False.
            bio_only (bool, optional): Whether to only use emission data of biospheric fluxes (no fire and fossile fuel). Defaults to Flase
        """
        self.spatial_valriables = spatial_valriables
        self.result_path = Path(result_path)
        self.results = pd.read_pickle(self.result_path)
        self.start = np.datetime64(month).astype("datetime64[D]")
        self.stop = (np.datetime64(month) + np.timedelta64(1, "M")).astype("datetime64[D]")
        self.flux_path = Path(flux_path)
        self.time_coarse = time_coarse
        self.coarsen_boundary = coarsen_boundary
        self.time_unit = time_unit
        self.boundary = boundary
        self.concentration_key = concentration_key
        self.data_outside_month = data_outside_month
        self.bio_only = bio_only
        self.min_time = None
        self.fit_result: Optional[tuple] = None
        self.predictions: Optional[xr.DataArray] = None
        self.predictions_flat: Optional[xr.DataArray] = None
        self.l_curve_result: Optional[tuple] = None
        self.alpha: Optional[Iterable[float]] = None
        self.reg: Optional[bayesinverse.Regression] = None

        self.time_coord, self.isocalendar = self.get_time_coord(self.time_unit)
        self.footprints, self.concentrations, self.concentration_errs = self.get_footprint_and_measurement(self.concentration_key)
        self.flux, self.flux_errs = self.get_flux()

        self.coords = self.footprints.stack(new=[self.time_coord, *self.spatial_valriables]).new

        self.footprints_flat = self.footprints.stack(new=[self.time_coord, *self.spatial_valriables])
        self.flux_flat = self.flux.stack(new=[self.time_coord, *self.spatial_valriables])
        self.flux_errs_flat = self.flux_errs.stack(new=[self.time_coord, *self.spatial_valriables])

    @staticmethod
    def get_time_coord(time_unit: Timeunit):
        """Gives name of time unti to group data by

        Args:
            time_unit (Timeunit): Name of time_unit

        Raises:
            ValueError: If no valid unit is given

        Returns:
            tuple[str, bool]: time_coord: Name of time coordinate, isocalendar, whether time coord is to be applied on DataArray.dt of DataArray.dt.isocalendar()
        """        
        if time_unit == "week":
            time_coord = "week"
            isocalendar = True
            
        elif time_unit == "day":
            time_coord = "dayofyear"
            isocalendar = False
        else:
            raise ValueError(f"time_by value '{time_coord}' not acceptable. Choose from: 'week', 'day'")
        return time_coord, isocalendar

    @staticmethod
    def apply_coarse_func(
        xarr: Union[xr.core.rolling.DataArrayCoarsen, xr.core.groupby.DataArrayGroupBy],
        coarse_func: Union[Callable[[xr.DataArray], float], str]
        ) -> xr.DataArray:
        """Applies coarsening function to grouped/coarsened DataArray. If string is given it is used to call the function from getattr, else its used by .apply

        Args:
            xarr (Union[xr.core.rolling.DataArrayCoarsen, xr.core.groupby.DataArrayGroupBy]): Array to apply function on.
            coarse_func (Union[Callable, str]): Function to apply. String used as getattr(xarr, coarse_func), Callable will be used with xarr.apply(coarse_func)

        Returns:
            xr.DataArray: Output after application of function
        """        
        if isinstance(coarse_func, str):
            xarr = getattr(xarr, coarse_func)()
        elif isinstance(coarse_func, Callable):
            xarr = xarr.apply(coarse_func)
        return xarr

    def coarsen_data(
        self, 
        xarr: xr.DataArray, 
        coarse_func: Coarsefunc, 
        time_coarse: Optional[int] = None
        ) -> xr.DataArray:    
        """Coarsens DataArray and applyling coarse function to blocks. Takes care of spatial and temporal coarsening

        Args:
            xarr (xr.DataArray): DataArray to reduce
            coarse_func (Coarsefunc): Function to apply
            time_coarse (Optional[int], optional): Specifiing by how much to coarsen time coordinte. Defaults to None.

        Returns:
            xr.DataArray: Coarser DataArray
        """            
        raise NotImplementedError

    def get_footprint_and_measurement(self, 
        data_key: Concentrationkey
        ) -> tuple[xr.DataArray, xr.DataArray , xr.DataArray]:
        """Loads and coarsens footprints according to parameters in __init__. Reads out measurement xco2 values and errors. Calculates difference to given data_key e.g. background and puts result in xr.DataArray.coarsen

        Args:
            data_key (Concentrationkey): Data with which difference to measurement xco2 is calculated

        Returns:
            tuple[xr.DataArray, xr.DataArray , xr.DataArray]: footprints, concentration_differences, measurement_xco2 errors
        """     
        min_time = self.start
        concentrations = []
        concentration_errs = []
        measurement_id = []
        footprints = []
        for i, result_row in tqdm(self.results.iterrows(), desc="Loading footprints", total=self.results.shape[0]):
            concentrations.append(result_row["xco2_measurement"] - result_row[data_key])
            concentration_errs.append(result_row["measurement_uncertainty"])
            measurement_id.append(i)
            file = os.path.join(result_row["directory"], "Footprint_total_inter_time.nc")
            if not os.path.exists(path):
                continue        
            footprint: xr.DataArray = xr.load_dataarray(file)
            footprint = footprint.expand_dims("measurement").assign_coords(dict(measurement=[i]))
            if not self.data_outside_month:
                footprint = footprint.where((footprint.time >= self.start) * (footprint.time < self.stop), drop=True)
            else:
                min_time = footprint.time.min().values.astype("datetime64[D]") if footprint.time.min() < min_time else min_time

            footprint = select_boundary(footprint, self.boundary)
            footprint = self.coarsen_data(footprint, "sum", None)
            footprints.append(footprint)
        self.min_time = min_time
        concentrations = xr.DataArray(
            data = concentrations,
            dims = "measurement",
            coords = dict(measurement = measurement_id)
        )
        concentration_errs  = xr.DataArray(
            data = concentration_errs,
            dims = "measurement",
            coords = dict(measurement = measurement_id)
        )
        # merging and conversion from s m^3/kg to s m^2/mol
        footprints = xr.concat(footprints, dim = "measurement")/100 * 0.044
        # extra time coarsening for consistent coordinates 
        if not self.time_coarse is None:
            footprints = footprints.coarsen({self.time_coord:self.time_coarse}, boundary=self.coarsen_boundary).sum()
        footprints = footprints.where(~np.isnan(footprints), 0)
        return footprints, concentrations, concentration_errs

    def get_flux(self) -> tuple[xr.DataArray,xr.DataArray]:
        """Loads fluxes and its errors and coarsens to parameters given in __init__

        Returns:
            tuple[xr.DataArray,xr.DataArray]: fluxes and errors
        """        
        flux_files = []
        for date in np.arange(self.min_time.astype("datetime64[D]"), self.stop):
            date_str = str(date).replace("-", "")
            flux_files.append(self.flux_path.parent / (self.flux_path.name + f"{date_str}.nc"))
        flux = xr.open_mfdataset(flux_files, drop_variables="time_components").compute()
        if self.bio_only:
            flux = flux.bio_flux_opt + flux.ocn_flux_opt
        else:
            flux = flux.bio_flux_opt + flux.ocn_flux_opt + flux.fossil_flux_imp + flux.fire_flux_imp
        flux = select_boundary(flux, self.boundary)
        flux_mean = self.coarsen_data(flux, "mean", self.time_coarse)
        flux_err = self.get_flux_err(flux, flux_mean)
        # Error of mean calculation
        return flux_mean, flux_err

    def get_flux_err(self, flux: xr.DataArray, flux_mean: xr.DataArray) -> xr.DataArray:
        """Computes error of the mean of the fluxes

        Args:
            flux (xr.DataArray): original flux data
            flux_mean (xr.DataArray): fully coarsened flux data

        Returns:
            xr.DataArray: errors of means in flux_mean
        """        
        raise NotImplementedError

    def get_regression(
        self,
        x: Optional[np.ndarray] = None,
        yerr: Optional[Union[np.ndarray, xr.DataArray, float, list[float]]] = None, 
        xerr: Optional[Union[np.ndarray, list]] = None,
        with_prior: Optional[bool] = True,
        ) -> bayesinverse.Regression:
        """Constructs a Regression object from class attributes and inputs 

        Args:
            x (Optional[np.ndarray], optional): Alternative values for prior. Defaults to None.
            yerr (Optional[Union[np.ndarray, xr.DataArray, float, list[float]]], optional): Alternative values for y_covariance. Defaults to None.
            xerr (Optional[Union[np.ndarray, list]], optional): Alternative values for x_covariances. Defaults to None.
            with_prior (Optional[bool], optional): Switch to use prior. Defaults to True.

        Returns:
            bayesinverse.Regression: Regression model
        """        
        concentration_errs = self.concentration_errs.values
        if not yerr is None:
            if isinstance(yerr, float):
                yerr = np.ones_like(concentration_errs) * yerr
            elif isinstance(yerr, xr.DataArray):
                if yerr.size == 1:
                    yerr = np.ones_like(concentration_errs) * yerr.values
                else:
                    yerr = yerr.values
            concentration_errs = yerr
        
        flux_errs = self.flux_errs_flat
        if not xerr is None:
            if isinstance(xerr, float):
                xerr = np.ones_like(flux_errs) * xerr
            if isinstance(xerr, xr.DataArray):
                if xerr.size == 1:
                    xerr = np.ones_like(flux_errs) * xerr.values
                elif xerr.shape == self.flux_errs.shape:
                    xerr = xerr.stack(new=[self.time_coord, "longitude", "latitude"]).values        
                else:
                    xerr.values
            flux_errs = xerr

        if not x is None:
            x_prior = x
        else:
            x_prior = self.flux_flat.values

        if with_prior:
            self.reg = bayesinverse.Regression(
                y = self.concentrations.values*1e-6, 
                K = self.footprints_flat.values, 
                x_prior = x_prior, 
                x_covariance = flux_errs**2, 
                y_covariance = concentration_errs*1e-6**2
            )
        else:
            self.reg = bayesinverse.Regression(
                y = self.concentrations.values*1e-6, 
                K = self.footprints_flat.values
            )
        return self.reg

    def fit(
        self, 
        x: Optional[np.ndarray] = None,
        yerr: Optional[Union[np.ndarray, xr.DataArray, float, list[float]]] = None, 
        xerr: Optional[Union[np.ndarray, list]] = None,
        with_prior: Optional[bool] = True,
        alpha: float = 1,
        ) -> xr.DataArray:
        """Uses bayesian inversion to estiamte emissions.

        Args:
            yerr (Optional[list], optional): Can be used instead of loaded errorof measurement. Defaults to None.
            xerr (Optional[list], optional): Can be used instead of error of the mean of the fluxes. Defaults to None.
            with_prior (Optional[bool]): Wether to use a prior or not. True strongly recommended. Defaults to True.
            alpha (Optional[float]): Value to weigth xerr against yerr. Is used as in l curve calculation. Defaults to 1.

        Returns:
            xr.DataArray: estimated emissions
        """        
        _ = self.get_regression(x, yerr, xerr, with_prior)
        

        if alpha != 1:
            result = self.reg.compute_l_curve([alpha])
            self.fit_result = (
                result["x_est"][0],
                result["res"][0],
                result["rank"][0],
                result["s"][0]
            )
        else:
            self.fit_result = self.reg.fit()
        self.predictions_flat = xr.DataArray(
            data = self.fit_result[0],
            dims = ["new"],
            coords = dict(new=self.coords)
        )
        self.predictions = self.predictions_flat.unstack("new")
        return self.predictions

    def compute_l_curve(
        self, 
        alpha: Iterable[float],
        cond: Any=None,
        x: Optional[np.ndarray] = None,
        yerr: Optional[Union[np.ndarray, xr.DataArray, float, list[float]]] = None, 
        xerr: Optional[Union[np.ndarray, list]] = None,
        with_prior: Optional[bool] = True
        ) -> xr.DataArray:
        """Uses bayesian inversion to estiamte emissions.

        Args:
            yerr (Optional[list], optional): Can be used instead of loaded errorof measurement. Defaults to None.
            xerr (Optional[list], optional): Can be used instead of error of the mean of the fluxes. Defaults to None.
            with_prior (Optional[bool]): Wether to use a prior or not. True strongly recommended. Defaults to True.

        Returns:
            xr.DataArray: estimated emissions
        """        

        concentration_errs = self.concentration_errs.values
        if not yerr is None:
            if isinstance(yerr, float):
                yerr = np.ones_like(concentration_errs) * yerr
            elif isinstance(yerr, xr.DataArray):
                if yerr.size == 1:
                    yerr = np.ones_like(concentration_errs) * yerr.values
                else:
                    yerr = yerr.values
            concentration_errs = yerr
        
        flux_errs = self.flux_errs_flat
        if not xerr is None:
            if isinstance(xerr, float):
                xerr = np.ones_like(flux_errs) * xerr
            if isinstance(xerr, xr.DataArray):
                if xerr.size == 1:
                    xerr = np.ones_like(flux_errs) * xerr.values
                elif xerr.shape == self.flux_errs:
                    xerr = xerr.stack(new=[self.time_coord, self.spatial_valriables]).values        
                else:
                    xerr.values
            flux_errs = xerr

        if not x is None:
            x_prior = x
        else:
            x_prior = self.flux_flat.values

        if with_prior:
            reg = bayesinverse.Regression(
                y = self.concentrations.values*1e-6, 
                K = self.footprints_flat.values, 
                x_prior = x_prior, 
                x_covariance = flux_errs**2, 
                y_covariance = concentration_errs*1e-6**2
            )
        else:
            reg = bayesinverse.Regression(
                y = self.concentrations.values*1e-6, 
                K = self.footprints_flat.values
            )
        self.l_curve_result = reg.compute_l_curve(alpha, cond)
        self.alpha = alpha
        return self.l_curve_result

    def map_on_grid(self, xarr: xr.DataArray) -> xr.DataArray:
        """Takes DataArray and returns the valaues represented on a lat lon grid

        Args:
            xarr (xr.DataArray): Original Dataarray

        Returns:
            xr.DataArray: DataArray on latitude longitude grid
        """        
        raise NotImplementedError
    ################################## Plotting ########################################

    def plot_l_curve(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple[int, int]] = None,
        cbar: bool = False,
        cbar_kwargs: dict = dict(),
        mark_ind: Optional[int] = None,
        mark_kwargs: dict = dict(),
        
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        

        sc = ax.scatter(
            self.l_curve_result["loss_forward_model"],
            self.l_curve_result["loss_regularization"],
            **kwargs)
        if cbar:
            fig.colorbar(sc, ax=ax, **cbar_kwargs)
        if mark_ind:
            mark_alpha = self.alpha[mark_ind]

            mark_default_kwargs = dict(
            s=50,
            label=f"$lambda = {mark_alpha:.4}$",
            alpha = 0.5,
            color="grey"
            )
            mark_default_kwargs.update(mark_kwargs)
            x_val = self.l_curve_result["loss_forward_model"][mark_ind]
            y_val = self.l_curve_result["loss_regularization"][mark_ind]
            ax.scatter(
                x_val,
                y_val,
                **mark_default_kwargs
            )
        ax.set_xlabel("Loss of forward model")
        ax.set_ylabel("Loss of regularization")
        ax.set_title("L-curve")
        ax.set_xscale("log")
        ax.set_yscale("log")
        return fig, ax

    def plot_correlation_concentration(
        self, 
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple[int, int]] = None,
        add_line: bool = True,
        line_kwargs: dict = dict(),
        **kwargs   
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        line_default = dict(color="gray", linestyle="dashed", linewidth=1)
        line_default.update(line_kwargs)

        concentrations = self.concentrations.values
        concentration_prediciton = self.footprints_flat.values @ self.predictions_flat.values * 1e6

        ax.scatter(concentrations, concentration_prediciton, **kwargs)
        ax.axis("equal")
        if add_line:
            ax.autoscale(enable=False)
            ax.plot([-100, 100],[-100, 100], **line_default)
        ax.set_title("Correlation of concentrations")
        ax.set_xlabel("Concentration measurement [ppm]")
        ax.set_ylabel("Concentration from prediction [ppm]")
        ax.grid(True)
        return fig, ax

    def plot_correlation_emission(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple[int, int]] = None,
        add_line: bool = True,
        line_kwargs: dict = dict(),
        **kwargs   
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        line_default = dict(color="gray", linestyle="dashed", linewidth=1)
        line_default.update(line_kwargs)

        ax.scatter(self.flux_flat.values, self.predictions_flat.values, **kwargs)
        ax.axis("equal")
        if add_line:
            ax.autoscale(enable=False)
            ax.plot([-100, 100],[-100, 100], **line_default)
        ax.set_title("Correlation of emissions")
        ax.set_xlabel(r"Emmission prior [mol s$^{-1}$ m$^{-2}$]")
        ax.set_ylabel(r"Emission prediction [mol s$^{-1}$ m$^{-2}$]")
        ax.grid(True)
        return fig, ax

    def plot_correlation_prediction_footprint(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[tuple[int, int]] = None,
        **kwargs   
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.scatter(self.footprints_flat.sum("measurement").values, self.predictions_flat.values, **kwargs)
        ax.set_title("Correlation footprint and predicted emissions")
        ax.set_xlabel(r"Footprint values [s m$^{2}$ mol$^{-1}$]")
        ax.set_ylabel(r"Emission prediction [mol s$^{-1}$ m$^{-2}$]")
        ax.grid(True)
        return fig, ax

    @staticmethod
    def add_plot_on_map(ax: plt.Axes, xarr: xr.DataArray, **kwargs):
        xarr.plot(ax = ax, **kwargs)
        FlexDataset2.add_map(ax = ax)
        return ax

    def plot_total_emission(
        self,
        data_type: Literal["prior", "predicted", "difference"],
        ax: plt.Axes = None,
        figsize: tuple[int, int] = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:

        if data_type == "prior":
            data = self.flux
            title = "Prior emissions"
        elif data_type == "predicted":
            data = self.predictions
            title = "Predicted emissions"
        elif data_type == "difference":
            data: xr.DataArray = self.predictions - self.flux 
            title = "Predicted minus Prior emissions"
        else:
            raise ValueError(f'"data_type" can oly be "prior", "predicted", "difference" not {data_type}')
        data = self.map_on_grid(data.mean(self.time_coord))
        v = np.abs(data).max()
        default_kwargs = dict(cmap = "bwr", x = "longitude", y = "latitude", vmin = -v, vmax = v, cbar_kwargs=dict(label=r"mol s$^{-1}$ m$^{-2}$"))
        default_kwargs.update(kwargs)

        if ax is None:
            fig, ax = FlexDataset2.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax = self.add_plot_on_map(ax=ax, xarr=data, **default_kwargs)
        ax.set_title(title)
        return fig, ax

    def get_emission_v(self):
        v = max([np.abs(self.flux.mean(self.time_coord)).max(), np.abs(self.predictions.mean(self.time_coord)).max()])
        return v

    def plot_all_total_emissions(
        self,
        axes: tuple[plt.Axes, plt.Axes, plt.Axes] = None,
        figsize: tuple[int, int] = (16, 3),
        **kwargs
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
        if axes is None:
            fig, axes = FlexDataset2.subplots(1,3, figsize=figsize, **kwargs)
        else:
            fig = axes[0].get_figure()
        v = self.get_emission_v()
        _ = self.plot_total_emission(data_type="prior", ax = axes[0], vmin = -v, vmax = v)
        _ = self.plot_total_emission(data_type="predicted", ax = axes[1], vmin = -v, vmax = v)
        _ = self.plot_total_emission(data_type="difference", ax = axes[2])
        return fig, axes

    def plot_footprint(
        self,
        ax: plt.Axes = None,
        time_ind: int = None,
        figsize: tuple[int, int] = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = FlexDataset2.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(cmap="jet", cbar_kwargs=dict(label = r"s m$^2$ mol$^{-1}$"))
        default_kwargs.update(kwargs)
        footprint = self.map_on_grid(self.footprints.mean("measurement"))
        if time_ind is None:
            _ = self.add_plot_on_map(ax = ax, xarr = footprint.sum([self.time_coord]), **default_kwargs)

        else:
            _ = self.add_plot_on_map(ax = ax, xarr = footprint({self.time_coord: time_ind}), **default_kwargs)
        return fig, ax

class InversionGrid(Inversion):
    def __init__(
        self, 
        result_path: Pathlike,
        month: str, 
        flux_path: Pathlike, 
        lon_coarse: int, 
        lat_coarse: int, 
        time_coarse: Optional[int] = None, 
        coarsen_boundary: str = "pad",
        time_unit: Timeunit = "week",
        boundary: Boundary = None,
        concentration_key: Concentrationkey = "background_inter",
        data_outside_month: bool = False,
        bio_only: bool = False
        ):
        """Calculates inversion of footprints and expected concentrations and offers plotting possibilities.

        Args:
            result_path (Pathlike): Path of file of collected results (from script calc_results.py)
            month (str): String to specify month to load measurement and flux data from in format YYYY-MM
            flux_path (Pathlike): Directory and filename of data until timestamp e.g. /path/to/dir/CT2019B.flux1x1.
            lon_coarse (int): By how much longitude should be coarsened 
            lat_coarse (int): By how much latitude should be coarsened
            time_coarse (Optional[int], optional): By how much new time coordinate should be coarsened. Defaults to None.
            coarsen_boundary (str, optional): Parameter for xr.DataArray.coarsen. Defaults to "pad".
            time_unit (Timeunit, optional): Time unit to group to. Defaults to "week".
            boundary (Boundary, optional): region of footprint to use. Defaults to None.
            concentration_key (Concentrationkey, optional): Data with which difference to measurement xco2 is calculated. Defaults to "background_inter".
            data_outdside_month (bool, optional): Wether to also use footprint data outsinde of target month. Defaults to False.
            bio_only (bool, optional): Whether to only use emission data of biospheric fluxes (no fire and fossile fuel). Defaults to Flase
        """
        self.lon_coarse = lon_coarse
        self.lat_coarse = lat_coarse
        super().__init__(
            ["longitude", "latitude"],
            result_path,
            month, 
            flux_path, 
            time_coarse, 
            coarsen_boundary,
            time_unit,
            boundary,
            concentration_key,
            data_outside_month,
            bio_only
        )

    def coarsen_data(
        self, 
        xarr: xr.DataArray, 
        coarse_func: Coarsefunc, 
        time_coarse: Optional[int] = None
        ) -> xr.DataArray:
        """Coarsens DataArray in two steps. First groups time to units that are given e.g. weeks and applies coarse_func[0] to grouped object. Then groups along new time_coordinate, longitude and latitude. with coarse_func[1]. If coarse_func is a single value it is applied in both cases

        Args:
            xarr (xr.DataArray): Array to coarsen
            coarse_func (Coarsefunc): Function(s) to apply in first and second grouping
            time_coarse (Optional[int], optional): By how much the new time component should be coarsened. Defaults to None.

        Returns:
            xr.DataArray: Coarser DataArray
        """        

        if not isinstance(coarse_func, (list, tuple)):
            coarse_func = [coarse_func]*2
        time_class = xarr["time"].dt
        if self.isocalendar:
            time_class = time_class.isocalendar()
        xarr = xarr.groupby(getattr(time_class, self.time_coord))
        xarr = self.apply_coarse_func(xarr, coarse_func[0])
        coarse_dict = dict(longitude=self.lon_coarse, latitude=self.lat_coarse)
        if not time_coarse is None:
            coarse_dict[self.time_coord] = time_coarse
        xarr = xarr.coarsen(coarse_dict, boundary=self.coarsen_boundary)
        xarr = self.apply_coarse_func(xarr, coarse_func[1])
        return xarr
    
    def get_flux_err(self, flux: xr.DataArray, flux_mean: xr.DataArray) -> xr.DataArray:
        def get_mean(xarr: xr.DataArray, full_arr: xr.DataArray, groupbyarr: xr.core.groupby.DataArrayGroupBy, means: xr.DataArray):
            """Helper function for error calculation. Finds out mean of block in xarr based on ungrouped, grouped and fully grouped DataArray"""
            for key, values in groupbyarr.groups.items():
                if all(np.isin(xarr.time.values, full_arr.time[values])):
                    time = int(key)
            time_mean = means[self.time_coord][np.argmin(np.abs(time - means[self.time_coord].values))]
            lon_mean = means.longitude.values[np.argmin(np.abs(xarr.longitude.values[:, None] - means.longitude.values[None, :]), axis = 1)]
            lat_mean = means.latitude.values[np.argmin(np.abs(xarr.latitude.values[:, None] - means.latitude.values[None, :]), axis = 1)]
            mean = means.sel({self.time_coord : time_mean, "longitude" : lon_mean, "latitude":lat_mean})
            return mean
            
        def squared_diff_to_mean(xarr: xr.DataArray, full_arr: xr.DataArray, groupbyarr: xr.core.groupby.DataArrayGroupBy, means: xr.DataArray):
            """Helper function for error calculation. Calculates square of difference of element of grouped DataArray to the final mean after further coarsening"""
            mean = get_mean(xarr, full_arr, groupbyarr, means)
            squared_diff = (xarr - mean.values[None, :])**2
            squared_diff = squared_diff.sum("time")
            return squared_diff
        
        def get_squared_diff_to_mean(self, flux: xr.DataArray) -> Callable:
            """Helper function for error calculation. Processes squared_diff_to_mean to be used with xr.core.groupby.DataArrayGroupBy.apply"""
            time_class = flux["time"].dt
            if self.isocalendar:
                time_class = time_class.isocalendar()
            flux_group = flux.groupby(getattr(time_class, self.time_coord))
            squared_diff_to_mean_partial = partial(
                squared_diff_to_mean,
                full_arr = flux,
                groupbyarr = flux_group, 
                means = flux_mean
            )
            return squared_diff_to_mean_partial
        flux_counts = self.coarsen_data(flux, ["count", "sum"], self.time_coarse)
        err_func = get_squared_diff_to_mean(self, flux)
        flux_err = self.coarsen_data(flux, [err_func, "sum"], self.time_coarse)
        flux_err = np.sqrt(flux_err/flux_counts)/np.sqrt(flux_counts)
        return flux_err

    def get_land_ocean_error(self, factor: float) -> xr.DataArray:
        """Selects pixels in the ocean and assignes scaled error by factor 'factor' to those.

        Args:
            factor (float): Factor for scaling

        Returns:
            xr.DataArray: Error with scaled values
        """        
        errs = xr.DataArray(np.ones_like(self.flux_errs), coords = self.flux_errs.coords) * self.flux_errs.mean()
        errs_dataframe = errs.to_dataframe(name="flux_errs").reset_index()
        errs = gpd.GeoDataFrame(
            errs_dataframe[[self.time_coord, "flux_errs"]], 
            geometry = gpd.points_from_xy(
                errs_dataframe.longitude, 
                errs_dataframe.latitude
            ), 
            crs = "EPSG:4326"
        )
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[["name", "continent", "geometry"]]
        world.crs = 'epsg:4326'
        world = world[world.continent != "Seven seas (open ocean)"]
        ocean_errs = gpd.overlay(errs, world, "difference")
        ocean_errs = pd.DataFrame(dict(week = ocean_errs.week, flux_errs = ocean_errs.flux_errs, longitude = ocean_errs.geometry.x, latitude = ocean_errs.geometry.y))
        ocean_errs.flux_errs = ocean_errs.flux_errs*factor
        ocean_errs = ocean_errs.set_index([self.time_coord, "latitude", "longitude"])
        errs_dataframe = errs_dataframe.set_index([self.time_coord, "latitude", "longitude"])
        errs_dataframe.update(ocean_errs)
        errs = errs_dataframe.to_xarray()["flux_errs"]
        return errs

    def map_on_grid(self, xarr: xr.DataArray) -> xr.DataArray:
        return xarr

class InversionEcosystems(Inversion):
    def __init__(self, 
        result_path: Pathlike,
        ecosyst_path: Pathlike,
        month: str, 
        flux_path: Pathlike, 
        time_coarse: Optional[int] = None, 
        coarsen_boundary: str = "pad",
        time_unit: Timeunit = "week",
        boundary: Boundary = None,
        concentration_key: Concentrationkey = "background_inter",
        data_outside_month: bool = False,
        bio_only: bool = False
        ):
        self.result_path = Path(result_path)
        self.ecosyst_path = Path(ecosyst_path)
        self.results = pd.read_pickle(self.result_path)
        self.start = np.datetime64(month).astype("datetime64[D]")
        self.stop = (np.datetime64(month) + np.timedelta64(1, "M")).astype("datetime64[D]")
        self.flux_path = Path(flux_path)
        self.time_coarse = time_coarse
        self.coarsen_boundary = coarsen_boundary
        self.time_unit = time_unit
        self.boundary = boundary
        self.concentration_key = concentration_key
        self.data_outside_month = data_outside_month
        self.bio_only = bio_only
        self.min_time = None
        self.fit_result: Optional[tuple] = None
        self.predictions: Optional[xr.DataArray] = None
        self.predictions_flat: Optional[xr.DataArray] = None
        self.l_curve_result: Optional[tuple] = None
        self.alpha: Optional[Iterable[float]] = None
        self.reg: Optional[bayesinverse.Regression] = None

        self.ecosyst_mask = self.get_ecosyst_mask()
        self.time_coord, self.isocalendar = self.get_time_coord(self.time_unit)
        self.footprints, self.concentrations, self.concentration_errs = self.get_footprint_and_measurement(self.concentration_key)
        self.flux, self.flux_errs = self.get_flux()

    def get_ecosyst_mask(self) -> xr.DataArray:
        mask = xr.load_dataarray(self.ecosyst_path).rename(dict(Lat="latitude", Long="longitude"))
        mask = select_boundary(mask, self.boundary)
        return mask
    
    def coarsen_data(
        self,
        xarr: xr.DataArray,
        coarse_func: Union[Callable, tuple[Callable, Callable, Callable], str, tuple[str, str, str]],
        coarsen_boundary: str,
        time_coarse: Optional[int] = None
        ) -> xr.DataArray:
        if not isinstance(coarse_func, (list, tuple)):
            coarse_func = [coarse_func]*3
        name = xarr.name
        combination = xr.merge([xarr, self.ecosyst_mask]).groupby("bioclass")
        xarr = self.apply_coarse_func(combination, coarse_func[0])[name]
        time_class = xarr["time"].dt
        if self.isocalendar:
            time_class = time_class.isocalendar()
        xarr = xarr.groupby(getattr(time_class, self.time_coord))
        xarr = self.apply_coarse_func(xarr, coarse_func[1])
        if not time_coarse is None:
            xarr = xarr.coarsen({self.time_coord: time_coarse}, boundary=coarsen_boundary)
            xarr = self.apply_coarse_func(xarr, coarse_func[2])
        return xarr

    
def calc_point(reg: bayesinverse.Regression, alpha:float) -> tuple[float, float]:
    """Calculates points on L-curve given a regulatization parameter and the model

    Args:
        reg (bayesinverse.Regression): Regression model
        alpha (float): Regularization weight factor

    Returns:
        tuple[float, float]: Logarithm of losses of forward model and regulatization
    """    
    result = reg.compute_l_curve([alpha])
    return np.log10(result["loss_forward_model"][0]), np.log10(result["loss_regularization"][0])

def euclidean_dist(P1: tuple[float, float], P2: tuple[float, float]) -> float:
    """Calculates euclidean distanceof two points given in tuples

    Args:
        P1 (tuple[float, float]): Point 1
        P2 (tuple[float, float]): Point 2

    Returns:
        float: distance
    """    
    P1 = np.array(P1)
    P2 = np.array(P2)
    return np.linalg.norm(P1-P2)**2

def calc_curvature(Pj: tuple[float, float], Pk: tuple[float, float], Pl: tuple[float, float]) -> float:
    """Calculates menger curvature of circle by three points

    Args:
        Pj (tuple[float, float]): Point with smallest regularization weight
        Pk (tuple[float, float]): Point with regularization weight in between 
        Pl (tuple[float, float]): Point with largest regulatization weight

    Returns:
        float: curvature
    """    
    return 2*(Pj[0] * Pk[1] + Pk[0] * Pl[1] + Pl[0] * Pj[1] - Pj[0] * Pl[1] - Pk[0] * Pj[1] - Pl[0] * Pk[1])/np.sqrt(euclidean_dist(Pj, Pk) * euclidean_dist(Pk, Pl) * euclidean_dist(Pl, Pj))

def get_l2(l1: float, l4: float) -> float:
    """Calculates position of regularization weight 2 from full boundary

    Args:
        l1 (float): Regularization weight lower boundary
        l4 (float): Regularization weight upper boundary

    Returns:
        float: New regularization weight in between
    """      
    phi = (1 + np.sqrt(5))/2
    
    x1 = np.log10(l1)
    x4 = np.log10(l4)
    x2 = (x4 + phi * x1) / (1 + phi)
    return 10**x2

def get_l3(l1: float, l2: float, l4: float) -> float:
    """Calculates position of regularization weight 3 other values

    Args:
        l1 (float): Regularization weight lower boundary
        l2 (float): Regularization weight in between
        l4 (float): Regularization weight upper boundary

    Returns:
        float: New regularization weight
    """    
    return 10**(np.log10(l1) + (np.log10(l4) - np.log10(l2)))

def optimal_lambda(reg: bayesinverse.Regression, interval: tuple[float, float], threshold: float):    
    """Find optiomal value for weigthing factor in regression. Based on https://doi.org/10.1088/2633-1357/abad0d

    Args:
        reg (bayesinverse.Regression): regression Model
        interval (tuple[float, float]): Values of weightung factors. Values within to search 
        threshold (float): Search stops if normalized search interval (upper boundary - lower boundary)/ upper boundary is smaller then threshold
    """    
    
    l1 = interval[0]
    l4 = interval[1]
    l2 =  get_l2(l1, l4)
    l3 = get_l3(l1, l2, l4)
    l_list = [l1, l2, l3, l4]

    p_list = []
    for l in l_list:
        p_list.append(calc_point(reg, l))
    
    while (l_list[3] - l_list[0]) / l_list[3] >= threshold:
        c2 = calc_curvature(*p_list[:3])
        c3 = calc_curvature(*p_list[1:])
        # Find convex part of curve
        while c3 <= 0:
            l_list[3] = l_list[2]
            l_list[2] = l_list[1]
            p_list[3] = p_list[2]
            p_list[2] = p_list[1]
            l_list[1] = get_l2(l_list[0], l_list[3])
            p_list[1] = calc_point(reg, l_list[1])
            c3 = calc_curvature(*p_list[1:])
            print(l_list[0], l_list[3])
        # Approach higher curvature
        if c2 > c3:
            # Store current guess
            l = l_list[1]
            # Set new boundaries
            l_list[3] = l_list[2]
            l_list[2] = l_list[1]
            p_list[3] = p_list[2]
            p_list[2] = p_list[1]
            l_list[1] = get_l2(l_list[0], l_list[3])
            p_list[1] = calc_point(reg, l_list[1])
        else:
            # Store current guess
            l = l_list[2]
            # Set new boundaries
            l_list[0] = l_list[1]
            p_list[0] = p_list[1]
            l_list[1] = l_list[2]
            p_list[1] = p_list[2]
            l_list[2] = get_l3(l_list[0], l_list[1], l_list[3])
            p_list[2] = calc_point(reg, l_list[2])
        print(l)
    return l

if __name__ == "__main__":

    Inversion = ResultsInversion(
        result_path="/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/11_unpacked/results_new.pkl",
        month="2009-11", 
        flux_path="/work/bb1170/static/CT2019/Flux3hour_1x1/CT2019B.flux1x1.",
        lon_coarse=5, 
        lat_coarse=5, 
        time_coarse = None,
        boundary=[110.0, 155.0, -45.0, -10.0],
        data_outside_month=True
    )
    

    # ds2, coord = collect_coarse_footprints_and_differences(
    #     result_path="/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/09_unpacked/results.pkl",
    #     start="2009-09-01", 
    #     stop="2009-10-01",
    #     time_by="week",
    #     lon=2,
    #     lat=5,
    #     time=2,
    #     boundary=None,
    #     coarsen_boundary="trim",  
    # )


    #sens = Sensitivity("/work/bb1170/RUN/b381737/data/FLEXPART/sensitivity/partnums_unpacked/predictions_fixed.pkl")
    # acos_file = "/work/bb1170/RUN/b381737/data/ACOS/acos_LtCO2_"
    # dirs = ["/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/09_unpacked",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/10_unpacked",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/11_unpacked",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/12_unpacked_mp",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2010/01_unpacked_mp",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2010/02_unpacked_mp",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2010/03_unpacked_mp",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2010/04_unpacked",
    #     "/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2010/05_unpacked"]

    # res = Results(dirs)
    # res.acos_correlation()