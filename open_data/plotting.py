import numpy as np
import pandas as pd

import os
import warnings
from functools import cache
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter

from open_data_utils import FlexDataset2

settings = dict()

path = "/work/bb1170/RUN/b381737/data/FLEXPART/sensitivity/partnums_unpacked/predictions_fixed.pkl"

class Sensitivity():
    def __init__(self, predictions_file: str, mode=0):
        self.predictions_file = predictions_file
        self.predictions = pd.read_pickle(predictions_file)
        if mode == 0:
            self.data = self._get_prediction_data() 
        elif mode == 1:
            self.data = self._get_prediction_data_legacy()
        elif mode == 2:
            self.data = self._get_prediction_data_legacy2()

    def _get_prediction_data(self):
        data_variables = ["enhancement", "background", "xco2"]
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



                if band_vals is not None:
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
        
        if vline is not None:
            ax.vlines(np.datetime64(fd.release["start"]) - np.timedelta64(vline, "D"), *ylim, **vline_kwargs)
            ax.set_ylim(*ylim)


        for i, border in enumerate(border_list):
            color = color_list[i] if len(color_list) == len(border_list) else color_list[0]
            linestyle = linestyle_list[i] if len(linestyle_list) == len(border_list) else linestyle_list[0]
            ax.hlines(border, *xlim, color=color, linestyle=linestyle, label=f"{border*100}%", **border_kwargs)
        ax.set_xlim(*xlim)
        #ax.legend()
        return fig, ax

if __name__ == "__main__":
    sens = Sensitivity("/work/bb1170/RUN/b381737/data/FLEXPART/sensitivity/partnums_unpacked/predictions_fixed.pkl")