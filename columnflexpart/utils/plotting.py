import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Union, Literal, Optional, Callable, Iterable

from functools import cache
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import cm

import bayesinverse

from columnflexpart.classes import FlexDataset
from columnflexpart.utils import detrend_hawaii

settings = dict()

#path = "/work/bb1170/RUN/b381737/data/FLEXPART/sensitivity/partnums_unpacked/predictions_fixed.pkl"

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
        band_kwargs: dict = {},
        line_kwargs: dict = {},
        scatter_kwargs: dict = {},
        ax: plt.Axes = None,
        cities: list[str] = ["darwin", "wollongong"], 
        markers: list[str] = ["*", "x"], 
        types: list[str] = ["unit", "pressure"], 
        coloring: list[str] = ["blue", "red"],
        figsize: tuple[int, int] = None,
        show_lines: bool = True,
        show_bands: bool = True,
        show_marker_labels: bool = True,
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

        default_band_kwargs = dict(alpha=0.3)
        default_band_kwargs.update(band_kwargs)
        default_line_kwargs = dict(linestyle="dashed")
        default_line_kwargs.update(line_kwargs)
        default_scatter_kwargs = dict()
        default_scatter_kwargs.update(scatter_kwargs)


        city_names = dict(wollongong="Wollongong", darwin="Darwin")

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
                label = f"{typ} {city_names[city]}" if show_marker_labels else None
                ax.scatter(p, v, marker=marker, c=color, label=label, **scatter_kwargs)



                if not band_vals is None:
                    if len(cities) > 1 or len(types) > 1:
                        if band_style == "percent":
                            assert "norm" in style, "Multiple cities with error band_style='percent' only possible for styles with 'norm'"
                        if band_style == "absolute":
                            assert  not "norm" in style, "Multiple cities with error band_style='absolute' only possible for styles without 'norm'"
                        if i != len(cities)-1 or j != len(types)-1:
                            continue 
                    xlim = ax.get_xlim()
                    ax.hlines(bar_center, *xlim, color="grey", **default_line_kwargs)
                    if band_style == "percent":
                        last_err = 0
                        for i, band_val in enumerate(band_vals):
                            if "norm" in style:
                                err = band_val
                            else: 
                                err = v0.mean() * band_val/100
                            if show_bands:
                                ax.fill_between([*xlim], bar_center + last_err, bar_center + err, color=band_colors[i], **default_band_kwargs, label=f"{band_val} % error band")
                                ax.fill_between([*xlim], bar_center - err, bar_center - last_err, color=band_colors[i], **default_band_kwargs)
                            if show_lines:
                                ax.hlines(bar_center + err, *xlim, color=band_colors[i], **default_line_kwargs)
                                ax.hlines(bar_center - err, *xlim, color=band_colors[i], **default_line_kwargs)
                            last_err = err
                        ax.set_xlim(*xlim)
                    
                    if band_style == "absolute":
                        last_err = 0
                        for i, band_val in enumerate(band_vals):
                            if "norm" in style:
                                err = band_val/v0.mean()
                            else:
                                err = band_val
                            if show_bands:
                                ax.fill_between([*xlim], bar_center + last_err, bar_center + err, color=band_colors[i], **default_band_kwargs, label=f"{band_val} ppm error band")
                                ax.fill_between([*xlim], bar_center - err, bar_center - last_err, color=band_colors[i], **default_band_kwargs)
                            if show_lines:
                                ax.hlines(bar_center + err, *xlim, color=band_colors[i], **default_line_kwargs)
                                ax.hlines(bar_center - err, *xlim, color=band_colors[i], **default_line_kwargs)
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
        fd = FlexDataset(file)
        heights = (fd.footprint.dataset.RELZZ2.values + fd.footprint.dataset.RELZZ1)/2
        heights = heights.compute()
        fp_sum = []
        for i in fd.footprint.dataset.pointspec.values:
            fp_sum.append(fd.footprint.dataarray.sel(pointspec=slice(0,i)).sum().compute())
        return heights, fp_sum

    @cache
    def calc_footprint_layer_sum_vs_release_height(self, city: str, typ: str, file_ind: int=-1):
        """Calculation sum of footprints from 0 to each height value in the

        Args:
            city (str): 
            typ (str): _description_
            file_ind (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """        
        file = str(self.data["directories"][city][typ][file_ind])
        fd = FlexDataset(file)
        heights = (fd.footprint.dataset.RELZZ2.values + fd.footprint.dataset.RELZZ1)/2
        heights = heights.compute()
        fp_sum = []
        for i in fd.footprint.dataset.pointspec.values:
            fp_sum.append(fd.footprint.dataarray.sel(pointspec=i).sum().compute())
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
        ax.set_xlabel("Level height [m]")
        ax.set_ylabel("Percentage of included footprint [%]")
        ax.set_title(f"Effect of release height ({city}, {typ})")
        return fig, ax

    def footprint_layer_sum_vs_release_height(
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
        heights, fp_sum = self.calc_footprint_layer_sum_vs_release_height(city, typ, file_ind)
        fp_sum_arr = np.array(fp_sum)
        fp_sum_arr = fp_sum_arr/np.sum(fp_sum_arr)*100
        ax.plot(heights, fp_sum_arr, **kwargs)
        ax.grid()
        ax.set_xlabel("Level height [m]")
        ax.set_ylabel("Contribution to total footprint [%]")
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
        fd = FlexDataset(file)
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

    def particles_in_domain(
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
        fd = FlexDataset(file)
        fd.trajectories.load_endpoints()
        end = fd.trajectories.endpoints
        times = end.time
        days = times.values.astype("datetime64[D]")
        runtime = days.max() - days
        runtime = runtime.astype(int)
        y, x, hist = ax.hist(runtime, cumulative=-1, density=True, bins=14)
        y = y*100
        if line:
            hist.remove()
            ax.plot(x[:-1], y, **line_kwargs)

        ax.set_xlabel("Simulated time [days]")
        ax.set_ylabel("Particels in the domain [%]")
        ax.set_title(f"Particles that left domain over time ({city}, {typ})")
        
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        
        if not vline is None:
            ax.vlines(vline, *ylim, **vline_kwargs)
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
        plot_model: bool = True,
        figsize: tuple[int, int] = (7,5),
        detrend: bool = False,
        ct_file: Union[Path, str] = None,
        ct_kwargs: dict() = dict(),
        old_ct_data: bool = False

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
            plot_model (bool): Wether to plot model. Defaults to True.
            figsize (tuple[int, int], optional): Figsize. Defaults to (7,5).
            ct_file (Union[Path, str]): Path to Carbon tracker file of monthly average values.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes with plot
        """        
    
        
        measurement_kw = dict()
        model_kw = dict()
        enh_kw = dict()
        enh_pos_kw = dict()
        enh_neg_kw = dict()
        bgd_kw = dict()
        ct_kw = dict()

        measurement_defaults = dict(fmt="o", linestyle="-", color="firebrick", label="Measurement") 
        model_defaults = dict(fmt="o", linestyle="-", color="black", label="Model")
        enh_defaults = dict(alpha=0.2)
        enh_pos_defaults = dict(color="green", label="positive_enhancement")
        enh_neg_defaults = dict(color="red", label="negative_enhancement")
        bgd_defaults = dict(color="black", linestyle="dashed", linewidth=1, label="Model background")
        ct_defaults = dict(fmt="o", color="blue", linestyle="-", label="CT2019B")

        if not ignore_defaults:
            measurement_kw.update(measurement_defaults)
            model_kw.update(model_defaults)
            enh_kw.update(enh_defaults)
            enh_pos_kw.update(enh_pos_defaults)
            enh_neg_kw.update(enh_neg_defaults)
            bgd_kw.update(bgd_defaults)
            ct_kw.update(ct_defaults)

        measurement_kw.update(measurement_kwargs)
        model_kw.update(model_kwargs)
        enh_kw.update(enh_kwargs)
        enh_pos_kw.update(enh_pos_kwargs)
        enh_neg_kw.update(enh_neg_kwargs)
        bgd_kw.update(bgd_kwargs)
        ct_kw.update(ct_kwargs)
    

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
                data.month = data.month + pd.to_timedelta([15] * len(data), "d")
                data = detrend_hawaii(data, key, "month")
                keys[i] = key + "_detrend"
                data.month = data.month - pd.to_timedelta([15] * len(data), "d")
        
        bgd_key, xco2_key, measurement_key = keys
        enh_key = "enhancement"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        if plot_measurement:
            ax.errorbar(xvals, data[measurement_key], **measurement_kw)
        #ax.plot(xvals, data.xco2_measurement, color=measurement_color)
        if plot_model:
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
        if not ct_file is None:
            ct_data = pd.read_pickle(ct_file)
            if old_ct_data:
                co2_key = "CO2"
                time_key = "MonthDate"
            else:
                co2_key = "xco2"
                time_key = "time"
            ct_data = ct_data[np.isin(ct_data[time_key].to_numpy().astype("datetime64[M]"), data.month.values)]
            if detrend:
                ct_data = detrend_hawaii(ct_data, co2_key, time_key)
                co2_key = co2_key + "_detrend"
            ax.errorbar(xvals, ct_data[co2_key], **ct_kw)

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