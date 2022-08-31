from lib2to3.pgen2.token import OP
from tkinter import E
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from typing import Union, Literal, Optional, Callable
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
        plot_acos: bool = True,
        acos_kwargs: dict = dict(),
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
        """PLots monthly average of xco2 prediction in comparison to acos data. 

        Args:
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            interpolated (bool, optional): Wheter or not to use interpolated values (with averaging kernel). Defaults to True.
            ignore_defaults (bool, optional): Whether only kwargs set by user are to be used. Defaults to False.
            plot_acos (bool, optional): Whether to plot acos dataframe. Defaults to True.
            acos_kwargs (dict, optional): Kwargs of acos errorbar plot. Defaults to dict().
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
    
        
        acos_kw = dict()
        model_kw = dict()
        enh_kw = dict()
        enh_pos_kw = dict()
        enh_neg_kw = dict()
        bgd_kw = dict()

        acos_defaults = dict(fmt="o", linestyle="-", color="firebrick", label="ACOS") 
        model_defaults = dict(fmt="o", linestyle="-", color="black", label="Model")
        enh_defaults = dict(alpha=0.2)
        enh_pos_defaults = dict(color="green", label="positive_enhancement")
        enh_neg_defaults = dict(color="red", label="negative_enhancement")
        bgd_defaults = dict(color="black", linestyle="dashed", linewidth=1, label="Model background")

        if not ignore_defaults:
            acos_kw.update(acos_defaults)
            model_kw.update(model_defaults)
            enh_kw.update(enh_defaults)
            enh_pos_kw.update(enh_pos_defaults)
            enh_neg_kw.update(enh_neg_defaults)
            bgd_kw.update(bgd_defaults)

        acos_kw.update(acos_kwargs)
        model_kw.update(model_kwargs)
        enh_kw.update(enh_kwargs)
        enh_pos_kw.update(enh_pos_kwargs)
        enh_neg_kw.update(enh_neg_kwargs)
        bgd_kw.update(bgd_kwargs)
    

        xco2_variable = "xco2_inter" if interpolated else "xco2"
        background_variable = "background_inter" if interpolated else "background"
        enhancements_variable = "enhancement_diff" if interpolated else "enhancement"
        data = self.predictions[["time", xco2_variable, background_variable, enhancements_variable, "xco2_acos"]]
        data = data.rename(columns={xco2_variable:"xco2", background_variable:"background", enhancements_variable:"enhancement"})
        keys = ["background","xco2", "xco2_acos"]

        data = self.mean(data, "month").sort_values("month")
        xvals =  np.arange(len(data))

        if detrend:
            for i, key in enumerate(keys):
                data = detrend_hawaii(data, key, "month")
                keys[i] = key + "_detrend"
        
        bgd_key, xco2_key, acos_key = keys
        enh_key = "enhancement"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        if plot_acos:
            ax.errorbar(xvals, data[acos_key], **acos_kw)
        #ax.plot(xvals, data.xco2_acos, color=acos_color)
        ax.errorbar(xvals, data[xco2_key], **model_kw)
        #ax.plot(xvals, data.xco2, color=model_color, )

        ylim = ax.get_ylim()

        
        if plot_bgd:
            ax.plot(xvals, data[bgd_key], **bgd_kw)
        if plot_enh:
            xvals_interp = np.linspace(xvals.min(), xvals.max(), 100)
            bgd_interp=np.interp(xvals_interp, xvals, np.array(data[bgd_key]))
            xco2_interp=np.interp(xvals_interp, xvals, np.array(data[xco2_key]))
            enh_interp=np.interp(xvals_interp, xvals, np.array(data[enh_key]))
            ax.fill_between(xvals_interp, bgd_interp, xco2_interp, where=enh_interp > 0, **enh_kw, **enh_pos_kw)
            ax.fill_between(xvals_interp, bgd_interp, xco2_interp, where=enh_interp < 0, **enh_kw, **enh_neg_kw)

        ax.set_ylim(*ylim)
        ax.grid(True)
        ax.set_xticks(xvals, data.month.values.astype("datetime64[M]"))
        ax.set_xlabel("Month")
        ax.set_ylabel("XCO2 [ppm]")
        ax.set_title("XCO2 prediction (monthly average) - Model vs. ACOS ")
        ax.legend()
        
        return fig, ax


    def acos_correlation(
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
        data = self.predictions[["time", xco2_variable, "xco2_acos"]]
        data = data.rename(columns={xco2_variable:"xco2"})
        keys = ["xco2", "xco2_acos"]
        if mean_over is None:
            mean_over = "time"
        else:
            data = self.mean(data, mean_over)

        if detrend:
            detrend_str = " (detrended)"
            for i, key in enumerate(keys):
                data = detrend_hawaii(data, key, mean_over)
                keys[i] = key + "_detrend"
        
        xco2_key, acos_key = keys

        n_months = len(np.unique(data[[mean_over]].values.astype("datetime64[M]")))
        cmap = cm.get_cmap(cmap)
        colors = cmap(np.linspace(*color_range, n_months))
        
        if mean_over != "month":
            data.insert(1, "month", data[[mean_over]].values.astype("datetime64[M]"))
        
        for month, color in zip(np.unique(data.month), colors):
            selection = data[data.month == month]
            ax.scatter(selection[[xco2_key]], selection[acos_key], label=month.astype("datetime64[M]"), color=color, **kwargs)
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
        ax.set_ylabel(f"ACOS XCO2{detrend_str}[ppm]")
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

class ResultsInversion():
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
        concentration_key: Concentrationkey = "background_inter"
        ):
        """Calculates inversion of footprints and expected concentrations and offers plotting possibilities.

        Args:
            result_path (Pathlike): Path of file of collected results (from script calc_results.py)
            month (str): String to specify month to load sounding and flux data from in format YYYY-MM
            flux_path (Pathlike): Directory and filename of data until timestamp e.g. /path/to/dir/CT2019B.flux1x1.
            lon_coarse (int): By how much longitude should be coarsened 
            lat_coarse (int): By how much latitude should be coarsened
            time_coarse (Optional[int], optional): By how much new time coordinate should be coarsened. Defaults to None.
            coarsen_boundary (str, optional): Parameter for xr.DataArray.coarsen. Defaults to "pad".
            time_unit (Timeunit, optional): Time unit to group to. Defaults to "week".
            boundary (Boundary, optional): region of footprint to use. Defaults to None.
            concentration_key (Concentrationkey, optional): Data with which difference to sounding xco2 is calculated. Defaults to "background_inter".
        """        
        self.result_path = Path(result_path)
        self.results = pd.read_pickle(self.result_path)
        self.start = np.datetime64(month).astype("datetime64[D]")
        self.stop = (np.datetime64(month) + np.timedelta64(1, "M")).astype("datetime64[D]")
        self.flux_path = Path(flux_path)
        self.lon_coarse = lon_coarse
        self.lat_coarse = lat_coarse
        self.time_coarse = time_coarse
        self.coarsen_boundary = coarsen_boundary
        self.time_unit = time_unit
        self.boundary = boundary
        self.concentration_key = concentration_key

        self.time_coord, self.isocalendar = self.get_time_coord(self.time_unit)
        self.flux, self.flux_errs = self.get_flux()
        self.footprints, self.concentrations, self.concentration_errs = self.get_footprint_and_sounding(self.concentration_key)

        self.coords = self.footprints.stack(new=[self.time_coord, "longitude", "latitude"]).new

        self.footprints_flat = self.footprints.stack(new=[self.time_coord, "longitude", "latitude"])
        self.flux_flat = self.flux.stack(new=[self.time_coord, "longitude", "latitude"])
        self.flux_errs_flat = self.flux_errs.stack(new=[self.time_coord, "longitude", "latitude"])

        self.fit_result: Optional[tuple] = None
        self.predictions: Optional[xr.DataArray] = None
        self.predictions_flat: Optional[xr.DataArray] = None
    
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
        coarse_func: Union[Callable, str]
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
        time_unit: str, 
        lon_coarse: int, 
        lat_coarse: int, 
        coarse_func: Coarsefunc, 
        coarsen_boundary: str, 
        time_coarse: Optional[int] = None
        ) -> xr.DataArray:
        """Coarsens DataArray in two steps. First groups time to units that are given e.g. weeks and applies coarse_func[0] to grouped object. Then groups along new time_coordinate, longitude and latitude. with coarse_func[1]. If coarse_func is a single value it is applied in both cases

        Args:
            xarr (xr.DataArray): Array to coarsen
            time_unit (str): Time unit to convert time to
            lon_coarse (int): By how much longitude should be coarsened
            lat_coarse (int): By how much latitude should be coarsened
            coarse_func (Coarsefunc): Function(s) to apply in first and second grouping
            coarsen_boundary (str): Value for 'boundary' argument of xr.DataArray.coarsen
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
        coarse_dict = dict(longitude=lon_coarse, latitude=lat_coarse)
        if not time_coarse is None:
            coarse_dict[self.time_coord] = time_coarse
        xarr = xarr.coarsen(coarse_dict, boundary=coarsen_boundary)
        xarr = self.apply_coarse_func(xarr, coarse_func[1])
        return xarr
        
    def get_flux(self) -> tuple[xr.DataArray,xr.DataArray]:
        """Loads fluxes and its errors and coarsens to parameters given in __init__

        Returns:
            tuple[xr.DataArray,xr.DataArray]: fluxes and errors
        """        
        def get_mean(xarr: xr.DataArray, full_arr: xr.DataArray, groupbyarr: xr.core.groupby.DataArrayGroupBy, means: xr.DataArray):
            """Helper function for error calculation. Finds out mean of block in xarr based on ungrouped, grouped and fully grouped DataArray"""
            for key, values in groupbyarr.groups.items():
                if all(np.isin(xarr.time.values, full_arr.time[values])):
                    week = int(key)
            week_mean = means.week.values[np.argmin(week- means.week.values)]
            lon_mean = means.longitude.values[np.argmin(week- means.longitude.values)]
            lat_mean = means.latitude.values[np.argmin(week- means.latitude.values)]
            mean = means.sel(week = week_mean, longitude = lon_mean, latitude=lat_mean)
            return mean
            
        def squared_diff_to_mean(xarr: xr.DataArray, full_arr: xr.DataArray, groupbyarr: xr.core.groupby.DataArrayGroupBy, means: xr.DataArray):
            """Helper function for error calculation. Calculates square of difference of element of grouped DataArray to the final mean after further coarsening"""
            mean = get_mean(xarr, full_arr, groupbyarr, means)
            squared_diff = (xarr - mean)**2
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

        flux_files = []
        for date in np.arange(self.start, self.stop):
            date_str = str(date).replace("-", "")
            flux_files.append(self.flux_path.parent / (self.flux_path.name + f"{date_str}.nc"))
        flux = xr.open_mfdataset(flux_files, drop_variables="time_components").compute()
        flux = flux.bio_flux_opt + flux.ocn_flux_opt + flux.fossil_flux_imp + flux.fire_flux_imp
        flux = select_boundary(flux, self.boundary)
        
        flux_mean = self.coarsen_data(flux, self.time_unit, self.lon_coarse, self.lat_coarse, "mean", self.coarsen_boundary, self.time_coarse)
        
        # Error of mean calculation
        flux_counts = self.coarsen_data(flux, self.time_unit, self.lon_coarse, self.lat_coarse, ["count", "sum"], self.coarsen_boundary, self.time_coarse)
        err_func = get_squared_diff_to_mean(self, flux)
        flux_err = self.coarsen_data(flux, self.time_unit, self.lon_coarse, self.lat_coarse, [err_func, "sum"], self.coarsen_boundary, self.time_coarse)
        flux_err = np.sqrt(flux_err/flux_counts)/np.sqrt(flux_counts)
        return flux_mean, flux_err

    def get_footprint_and_sounding(self, 
        data_key: Concentrationkey
        ) -> tuple[xr.DataArray, xr.DataArray , xr.DataArray]:
        """Loads and coarsens footprints according to parameters in __init__. Reads out sounding xco2 values and errors. Calculates difference to given data_key e.g. background and puts result in xr.DataArray.coarsen

        Args:
            data_key (Concentrationkey): Data with which difference to sounding xco2 is calculated

        Returns:
            tuple[xr.DataArray, xr.DataArray , xr.DataArray]: footprints, concentration_differences, sounding_xco2 errors
        """        
        concentrations = []
        concentration_errs = []
        sounding_id = []
        footprints = []
        for i, result_row in tqdm(self.results.iterrows(), desc="Loading footprints", total=self.results.shape[0]):
            concentrations.append(result_row["xco2_acos"] - result_row[data_key])
            concentration_errs.append(result_row["acos_uncertainty"])
            sounding_id.append(i)
            file = os.path.join(result_row["directory"], "Footprint_total_inter_time.nc")
            if not os.path.exists(path):
                continue        
            footprint: xr.DataArray = xr.load_dataarray(file)
            footprint = footprint.expand_dims("sounding").assign_coords(dict(sounding=[i]))
            footprint = footprint.where((footprint.time >= self.start) * (footprint.time < self.stop), drop=True)
            footprint = select_boundary(footprint, self.boundary)
            footprint = self.coarsen_data(footprint, self.time_unit, self.lon_coarse, self.lat_coarse, "sum", self.coarsen_boundary, None)
            footprints.append(footprint)
        concentrations = xr.DataArray(
            data = concentrations,
            dims = "sounding",
            coords = dict(sounding = sounding_id)
        )
        concentration_errs  = xr.DataArray(
            data = concentration_errs,
            dims = "sounding",
            coords = dict(sounding = sounding_id)
        )
        # merging and conversion from s m^3/kg to s m^2/mol
        footprints = xr.concat(footprints, dim = "sounding")/100 * 0.044
        # extra time coarsening for consistent coordinates 
        if not self.time_coarse is None:
            footprints = footprints.coarsen({self.time_coord:self.time_coarse}, boundary=self.coarsen_boundary).sum()
        footprints = footprints.where(~np.isnan(footprints), 0)
        return footprints, concentrations, concentration_errs

    def fit(
        self, 
        costum_yerr: Optional[Union[xr.DataArray, float, list[float]]] = None, 
        costum_xerr: Optional[list] = None
        ) -> xr.DataArray:
        """Uses bayesian inversion to estiamte emissions.

        Args:
            costum_yerr (Optional[list], optional): Can be used instead of loaded errorof soundings. Defaults to None.
            costum_xerr (Optional[list], optional): Can be used instead of error of the mean of the fluxes. Defaults to None.

        Returns:
            xr.DataArray: estimated emissions
        """        

        concentration_errs = self.concentration_errs.values
        if not costum_yerr is None:
            if isinstance(costum_yerr, float):
                costum_yerr = np.ones_like(concentration_errs) * costum_yerr
            elif isinstance(costum_yerr, xr.DataArray):
                if costum_yerr.size == 1:
                    costum_yerr = np.ones_like(flux_errs) * costum_yerr.values
                else:
                    costum_yerr = costum_yerr.values
            concentration_errs = costum_yerr
        
        flux_errs = self.flux_errs_flat
        if not costum_xerr is None:
            if isinstance(costum_xerr, float):
                costum_xerr = np.ones_like(flux_errs) * costum_xerr
            if isinstance(costum_xerr, xr.DataArray):
                if costum_xerr.size == 1:
                    costum_xerr = np.ones_like(flux_errs) * costum_xerr.values
                elif costum_xerr.shape == self.flux_errs:
                    costum_xerr = costum_xerr.stack(new=[self.time_coord, "longitude", "latitude"]).values        
                else:
                    costum_xerr.values
            flux_errs = costum_xerr


        reg = bayesinverse.Regression(
            y = self.concentrations.values*1e-6, 
            K = self.footprints_flat.values, 
            x_prior = self.flux_flat.values, 
            x_covariance = flux_errs**2, 
            y_covariance = concentration_errs*1e-6**2
        )
        self.fit_result = reg.fit()
        self.predictions_flat = xr.DataArray(
            data = self.fit_result[0],
            dims = ["new"],
            coords = dict(new=self.coords)
        )
        self.predictions = self.predictions_flat.unstack("new")
        return self.predictions

    ################################## Plotting ########################################

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
        ax.scatter(self.footprints_flat.sum("sounding").values, self.predictions_flat.values, **kwargs)
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
        elif data_type == "predicited":
            data = self.predictions
            title = "Predicted emissions"
        elif data_type == "difference":
            data = self.flux - self.predictions
            title = "Predicted minus Prior emissions"
        else:
            raise ValueError(f'"data_type" can oly be "prior", "predicted", "difference" not {data_type}')
        data = data.sum(self.time_coord)
        v = max(np.abs(data))
        default_kwargs = dict(cmap = "bwr", x = "longitude", y = "latitude", vmin = -v, vmax = v, cbar_kwargs=dict(label=r"mol s$^{-1}$ m$^{-2}$"))
        default_kwargs.update(kwargs)

        if ax is None:
            fig, ax = FlexDataset2.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax = self.add_plot_on_map(ax=ax, xarr=data, **default_kwargs)
        ax.set_title(title)
        return fig, ax

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
        data = data.sum(self.time_coord)
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



    def plot_all_total_emissions(
        self,
        figsize: tuple[int, int] = (12, 3),
        **kwargs
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
        fig, axes = FlexDataset2.subplots(1,3, figsize=figsize, **kwargs)
        v = max([np.abs(self.flux.sum("week")).max(), np.abs(self.predictions.sum("week")).max()])
        _ = self.plot_total_emission(data_type="prior", ax = axes[0], vmin = -v, vmax = v)
        _ = self.plot_total_emission(data_type="predicted", ax = axes[1], vmin = -v, vmax = v)
        _ = self.plot_total_emission(data_type="difference", ax = axes[2])
        return fig, axes

    def plot_footprint(
        self,
        ax: plt.Axes = None,
        figsize: tuple[int, int] = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = FlexDataset2.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(cmap="jet", cbar_kwargs=dict(label = r"s m$^2$ mol$^{-1}$]"))
        default_kwargs.update(kwargs)
        _ = self.add_plot_on_map(ax = ax, xarr = self.footprints.sum([self.time_coord, "sounding"]), **default_kwargs)
        return fig, ax
        


def collect_coarse_footprints_and_differences(
    result_path: Pathlike, 
    start: str, 
    stop: str, 
    lon: int, 
    lat: int, 
    time: int=1, 
    time_by: Literal["week", "day"] = "week", 
    boundary: Optional[tuple[float, float, float, float]] = None, 
    coarsen_boundary: str = "pad", 
    interpolated: bool = True
    ) -> tuple[xr.Dataset, str]:
    if time_by == "week":
        time_by = "week"
        time_coord = "week"
        isocalendar = True
        
    elif time_by == "day":
        time_by = "dayofyear"
        time_coord = "dayofyear"
        isocalendar = False
    else:
        raise ValueError(f"time_by value '{time_by}' not acceptable. Choose from: 'week', 'day'")
    
    background_key = "background" if not interpolated else "background_inter"

    results = pd.read_pickle(result_path)
    
    differences = []
    acos_uncertainties = []
    sounding_id = []
    xarrs = []
    for i, result_row in results.iterrows():
        differences.append(result_row["xco2_acos"] - result_row[background_key])
        acos_uncertainties.append(result_row["acos_uncertainty"])
        sounding_id.append(i)
        file = os.path.join(result_row["directory"], "Footprint_total_inter_time.nc")
        if not os.path.exists(path):
            continue        
        xarr = xr.load_dataarray(file)
        xarr = xarr.expand_dims("sounding").assign_coords(dict(sounding=[i]))
        xarr = xarr.where((xarr.time >= np.datetime64(start)) * (xarr.time < np.datetime64(stop)), drop=True)
        if not boundary is None:
            xarr = select_boundary(xarr, boundary)
        if isocalendar:
            time_values = xarr["time"].dt.isocalendar()
        else:
            time_values = xarr["time"].dt
        xarr = xarr.groupby(getattr(time_values, time_by)).sum()
        xarr_coarse = xarr.coarsen({"longitude":lon, "latitude":lat}, boundary=coarsen_boundary).sum()
        xarrs.append(xarr_coarse)
    

    differences  = xr.Dataset(
        data_vars = dict(
            differences = (["sounding"], differences)
        ),
        coords = dict(
            sounding = (["sounding"], sounding_id)
        )
    )
    acos_uncertainties  = xr.Dataset(
        data_vars = dict(
            acos_uncertainties = (["sounding"], acos_uncertainties)
        ),
        coords = dict(
            sounding = (["sounding"], sounding_id)
        )
    )
    # /100 * 0.044 to account for height of Flexpart box and molar weight of CO2
    xarr_merged = xr.concat(xarrs, dim="sounding")/100 * 0.044
    xarr_merged = xarr_merged.coarsen({time_coord:time}, boundary=coarsen_boundary).sum()
    xarr_merged = xarr_merged.where(~np.isnan(xarr_merged), 0).to_dataset(name="footprints")
    dataset = xarr_merged.merge(differences).merge(acos_uncertainties)
    return dataset, time_coord

















if __name__ == "__main__":

    Inversion = ResultsInversion(
        result_path="/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/11_unpacked/results.pkl",
        month="2009-11", 
        flux_path="/work/bb1170/static/CT2019/Flux3hour_1x1/CT2019B.flux1x1.",
        lon_coarse=5, 
        lat_coarse=5, 
        boundary=[110.0, 155.0, -45.0, -10.0]
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