import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from master.utils.utils import select_boundary, optimal_lambda
from master.classes.flexdataset import FlexDataset
from functools import partial
import geopandas as gpd

from typing import Optional, Literal, Union, Callable, Iterable, Any
from pathlib import Path
import bayesinverse



Pathlike = Union[Path, str]
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
        self.min_time = None
        self.fit_result: Optional[tuple] = None
        self.predictions: Optional[xr.DataArray] = None
        self.predictions_flat: Optional[xr.DataArray] = None
        self.prediction_errs: Optional[xr.DataArray] = None
        self.prediction_errs_flat: Optional[xr.DataArray] = None
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
            if not os.path.exists(file):
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

    def get_flux(self, bio_only=False, no_bio=False) -> tuple[xr.DataArray,xr.DataArray]:
        """Loads fluxes and its errors and coarsens to parameters given in __init__

        Returns:
            tuple[xr.DataArray,xr.DataArray]: fluxes and errors
        """        
        flux_files = []
        for date in np.arange(self.min_time.astype("datetime64[D]"), self.stop):
            date_str = str(date).replace("-", "")
            flux_files.append(self.flux_path.parent / (self.flux_path.name + f"{date_str}.nc"))
        flux = xr.open_mfdataset(flux_files, drop_variables="time_components").compute()
        assert not (bio_only and no_bio), "Choose either 'bio_only' or 'no_bio' not both"
        if bio_only:
            flux = flux.bio_flux_opt + flux.ocn_flux_opt
        elif no_bio:
            flux = flux.fossil_flux_imp
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
        alpha: Optional[float] = None,
        ) -> bayesinverse.Regression:
        """Constructs a Regression object from class attributes and inputs 

        Args:
            x (Optional[np.ndarray], optional): Alternative values for prior. Defaults to None.
            yerr (Optional[Union[np.ndarray, xr.DataArray, float, list[float]]], optional): Alternative values for y_covariance. Defaults to None.
            xerr (Optional[Union[np.ndarray, list]], optional): Alternative values for x_covariances. Defaults to None.
            with_prior (Optional[bool], optional): Switch to use prior. Defaults to True.
            alpha (Optional[float]): Regulatization value

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
                    xerr = xerr.stack(new=[self.time_coord, *self.spatial_valriables]).values        
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
                y_covariance = concentration_errs*1e-6**2, 
                alpha = alpha
            )
        else:
            self.reg = bayesinverse.Regression(
                y = self.concentrations.values*1e-6, 
                K = self.footprints_flat.values,
                alpha = alpha
            )
        return self.reg

    def fit(
        self, 
        x: Optional[np.ndarray] = None,
        yerr: Optional[Union[np.ndarray, xr.DataArray, float, list[float]]] = None, 
        xerr: Optional[Union[np.ndarray, list]] = None,
        with_prior: Optional[bool] = True,
        alpha: float = None,
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
        _ = self.get_regression(x, yerr, xerr, with_prior, alpha)
        
        self.fit_result = self.reg.fit()
        self.predictions_flat = xr.DataArray(
            data = self.fit_result[0],
            dims = ["new"],
            coords = dict(new=self.coords)
        )
        self.predictions = self.predictions_flat.unstack("new")
        self.prediction_errs_flat = xr.DataArray(
            data = np.sqrt(np.diag(self.get_posterior_covariance())),
            dims = ["new"],
            coords = dict(new=self.coords)
        )
        self.prediction_errs = self.prediction_errs_flat.unstack("new")
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

        reg = self.get_regression(x, yerr, xerr, with_prior)
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

    def get_filtered_weeks(self, xarr):
        assert self.time_coord == "week" and self.time_coarse in [None, 1], "filtering of weeks only psooible with settings time_coord=week, time_coarse=1 or time_coarse=None"
        start_day = self.start.astype(datetime).isoweekday()
        end_day = self.stop.astype(datetime).isoweekday()    
        drop_start = start_day > 4
        drop_end = end_day < 5 
        offset_flag = False
        if 1 in xarr.week:
            offset_flag = True
            xarr = xarr.assign_coords(week = (xarr.week -1 + 20) % 53)
        
        if start_day == 1:
            drop_start = True

        if end_day != 1 and drop_end:
            xarr = xarr.isel(week = xarr.week != xarr.week.max())

        if (len(xarr.week) == 5 and drop_start) or (len(xarr.week) == 6 and not drop_start) :
            xarr = xarr.isel(week = xarr.week != xarr.week.min())
        elif len(xarr.week) == 6 and drop_start:
            xarr = xarr.isel(week = ~np.isin(xarr.week, np.sort(xarr.week)[:2]))
        elif len(xarr.week) == 7:
            xarr = xarr.isel(week = ~np.isin(xarr.week, np.sort(xarr.week)[:2]))
        
        if offset_flag:
            xarr = xarr.assign_coords(week = (xarr.week - 20) % 53 + 1)

        return xarr.week
    
    def get_averaging_kernel(self, reduce: bool=False, only_cols: bool=False):
        ak = self.reg.get_averaging_kernel()
        if reduce:
            weeks = self.flux.week.values
            filtered_weeks = self.get_filtered_weeks(self.flux).values
            mask = np.ones((len(self.flux.week), self.n_eco))
            mask[~np.isin(weeks, filtered_weeks)] = 0
            mask = mask.flatten().astype(bool)
            ak = ak[mask]
            if not only_cols:
                ak = ak[:, mask]
        return ak

    def get_correlation(self, reduce: bool=False, only_cols: bool=False):
        corr = self.reg.get_correlation()
        if reduce:
            weeks = self.flux.week.values
            filtered_weeks = self.get_filtered_weeks(self.flux).values
            mask = np.ones((len(self.flux.week), self.n_eco))
            mask[~np.isin(weeks, filtered_weeks)] = 0
            mask = mask.flatten().astype(bool)
            corr = corr[mask]
            if not only_cols:
                corr = corr[:, mask]
        return corr

    def get_posterior_covariance(self, reduce: bool=False, only_cols: bool=False):
        cov = self.reg.get_posterior_covariance()
        if reduce:
            weeks = self.flux.week.values
            filtered_weeks = self.get_filtered_weeks(self.flux).values
            mask = np.ones((len(self.flux.week), self.n_eco))
            mask[~np.isin(weeks, filtered_weeks)] = 0
            mask = mask.flatten().astype(bool)
            cov = cov[mask]
            if not only_cols:
                cov = cov[:, mask]
        return cov

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
    def add_plot_on_map(ax: plt.Axes, xarr: xr.DataArray, map_kwargs: dict = dict(), **kwargs):
        xarr.plot(ax = ax, x="longitude", y="latitude", **kwargs)
        FlexDataset.add_map(ax = ax, **map_kwargs)
        return ax

    def plot_total_emission(
        self,
        data_type: Literal["prior", "predicted", "difference"],
        ax: plt.Axes = None,
        week: Optional[int] = None,
        figsize: tuple[int, int] = None,
        filter_weeks: bool = True,
        map_kwargs: dict = dict(),
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
        if filter_weeks:
            data = data.sel(week = self.get_filtered_weeks(data))
        if not week is None:
            data = data.sel(week = week).expand_dims("week")
        data = self.map_on_grid(data)
        data = data.mean(self.time_coord) if self.time_coord in data.coords else data
        v = np.abs(data).max()
        default_kwargs = dict(cmap = "bwr", vmin = -v, vmax = v, cbar_kwargs=dict(label=r"mol s$^{-1}$ m$^{-2}$"))
        default_kwargs.update(kwargs)
        if ax is None:
            fig, ax = FlexDataset.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax = self.add_plot_on_map(ax=ax, xarr=data, map_kwargs=map_kwargs, **default_kwargs)
        ax.set_title(title)
        return fig, ax

    def get_emission_v(self):
        difference = self.predictions - self.flux
        v = max([np.abs(self.flux.mean(self.time_coord)).max(), np.abs(self.predictions.mean(self.time_coord)).max(), np.abs(difference.mean(self.time_coord)).max()])
        return v

    def plot_all_total_emissions(
        self,
        axes: tuple[plt.Axes, plt.Axes, plt.Axes] = None,
        figsize: tuple[int, int] = (16, 3),
        week: Optional[int] = None,
        emission_kwargs: dict = dict(),
        filter_weeks: bool = True,
        map_kwargs: dict = dict(),
        **kwargs
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
        if axes is None:
            fig, axes = FlexDataset.subplots(1,3, figsize=figsize, **kwargs)
        else:
            fig = axes[0].get_figure()
        v = self.get_emission_v()
        default_emission_kwargs = dict(vmin = -v, vmax = v)
        default_emission_kwargs.update(emission_kwargs)
        _ = self.plot_total_emission(data_type="prior", ax = axes[0], week=week, filter_weeks=filter_weeks, map_kwargs=map_kwargs, **default_emission_kwargs)
        _ = self.plot_total_emission(data_type="predicted", ax = axes[1], week=week, filter_weeks=filter_weeks, map_kwargs=map_kwargs, **default_emission_kwargs)
        _ = self.plot_total_emission(data_type="difference", ax = axes[2], week=week, filter_weeks=filter_weeks, map_kwargs=map_kwargs, **default_emission_kwargs)
        return fig, axes

    def plot_footprint(
        self,
        ax: plt.Axes = None,
        time_ind: int = None,
        figsize: tuple[int, int] = None,
        filter_weeks: bool = True,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = FlexDataset.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        default_kwargs = dict(cmap="jet", cbar_kwargs=dict(label = r"s m$^2$ mol$^{-1}$"))
        default_kwargs.update(kwargs)
        footprint = self.footprints
        if filter_weeks:
            footprint = footprint.sel(week = self.get_filtered_weeks(footprint))
        footprint = self.map_on_grid(footprint.mean("measurement"))
        if time_ind is None:
            _ = self.add_plot_on_map(ax = ax, xarr = footprint.sum([self.time_coord]), **default_kwargs)

        else:
            _ = self.add_plot_on_map(ax = ax, xarr = footprint({self.time_coord: time_ind}), **default_kwargs)
        return fig, ax

###########################################################
# Inherited classes
###########################################################

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
            data_outside_month
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

class InversionBioclass(Inversion):
    def __init__(self, 
        result_path: Pathlike,
        bioclass_path: Pathlike,
        month: str, 
        flux_path: Pathlike, 
        time_coarse: Optional[int] = None, 
        coarsen_boundary: str = "pad",
        time_unit: Timeunit = "week",
        boundary: Boundary = None,
        concentration_key: Concentrationkey = "background_inter",
        data_outside_month: bool = False
        ):

        self.n_eco: Optional[int] = None
        self.bioclass_path = bioclass_path
        self.bioclass_mask = select_boundary(self.get_bioclass_mask(), boundary)
        super().__init__(
            ["bioclass"],
            result_path,
            month, 
            flux_path, 
            time_coarse, 
            coarsen_boundary,
            time_unit,
            boundary,
            concentration_key,
            data_outside_month,
        )

    def get_bioclass_mask(self) -> xr.DataArray:
        mask = xr.load_dataset(self.bioclass_path)["bioclass"]
        if "Lat" in mask.coords:
            mask = mask.rename(dict(Lat="latitude", Long="longitude"))
        self.n_eco = len(np.unique(mask.values))
        return mask

    def coarsen_data(
        self,
        xarr: xr.DataArray,
        coarse_func: Union[Callable, tuple[Callable, Callable, Callable], str, tuple[str, str, str]],
        time_coarse: Optional[int] = None
        ) -> xr.DataArray:
        if not isinstance(coarse_func, (list, tuple)):
            coarse_func = [coarse_func]*3
        time_class = xarr["time"].dt
        if self.isocalendar:
            time_class = time_class.isocalendar()
        xarr = xarr.groupby(getattr(time_class, self.time_coord))
        xarr = self.apply_coarse_func(xarr, coarse_func[0])
        xarr.name = "data"
        combination = xr.merge([xarr, self.bioclass_mask]).groupby("bioclass")
        xarr = self.apply_coarse_func(combination, coarse_func[1])["data"]
        
        if not time_coarse is None:
            xarr = xarr.coarsen({self.time_coord: time_coarse}, boundary=self.coarsen_boundary)
            xarr = self.apply_coarse_func(xarr, coarse_func[2])
        return xarr
    
    def get_flux_err(self, flux: xr.DataArray, flux_mean: xr.DataArray) -> xr.DataArray: 
        mean_times = flux.time.dt.isocalendar() if self.isocalendar else flux.time.dt
        mean_times = getattr(mean_times, self.time_coord)
        mean_times = flux_mean[self.time_coord][np.argmin(np.abs(mean_times.values[:, None] - flux_mean[self.time_coord].values[None, :]), axis=1)]
        mean_bioclass = self.bioclass_mask.sel(longitude = flux.longitude, latitude = flux.latitude)
        means = flux_mean.sel({"bioclass": mean_bioclass, self.time_coord: mean_times})
        flux_err = (flux - means.transpose(self.time_coord, ...).values)**2
        flux_err = self.coarsen_data(flux_err, "sum", self.time_coarse)
        flux_counts = self.coarsen_data(flux, ["count", "sum", "sum"], self.time_coarse)
        flux_err = np.sqrt(flux_err/flux_counts)/np.sqrt(flux_counts)
        return flux_err

    def map_on_grid(self, xarr: xr.DataArray) -> xr.DataArray:
        mapped_xarr = xr.DataArray(
            data = np.zeros(
                (
                    len(xarr[self.time_coord]),
                    len(self.bioclass_mask.longitude),
                    len(self.bioclass_mask.latitude)
                )
            ),
            coords = {
                self.time_coord: xarr[self.time_coord], 
                "longitude": self.bioclass_mask.longitude, 
                "latitude": self.bioclass_mask.latitude
            }
        )
        for bioclass in xarr.bioclass:
            mapped_xarr = mapped_xarr + (self.bioclass_mask == bioclass) * xarr.where(xarr.bioclass == bioclass, drop=True)
            mapped_xarr = mapped_xarr.squeeze(drop=True)
        return mapped_xarr

    def get_land_ocean_error(self, factor):
        err = xr.ones_like(self.flux_errs) * self.flux_errs.mean()
        err = err.where(err.bioclass != 0, self.flux_errs.mean() * factor)
        return err

    #MOVE TO INVERISION BIOCLASS
def get_total(inv):
    predictions = inv.predictions
    if inv.time_coord == "week":
        predictions = inv.prediction.sel(week = inv.get_filtered_weeks(inv.predictions))
    predictions = predictions.mean(inv.time_coord)
    mapped_predictions = inv.map_on_grid(predictions)
    mapped_predictions = mapped_predictions.where(mapped_predictions.bioclass != 0)
    flux_sum = mapped_predictions.mean().values
    flux_sum = flux_sum * (inv.stop_day.astype("datetime64[s]") - inv.start_day.astype("datetime64[s]")).astype(int)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))[["name", "geometry"]].to_crs({'init': 'epsg:3857'})
    area = world[world.name=="Australia"].area
    flux_sum = area*flux_sum
    

if __name__ == "__main__":

    # Inversion = InversionGrid(
    #     result_path="/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/11_unpacked/results_new.pkl",
    #     month="2009-11", 
    #     flux_path="/work/bb1170/static/CT2019/Flux3hour_1x1/CT2019B.flux1x1.",
    #     lon_coarse=5, 
    #     lat_coarse=5, 
    #     time_coarse = None,
    #     boundary=[110.0, 155.0, -45.0, -10.0],
    #     data_outside_month=True
    # )

    Inversion = InversionBioclass(
        result_path="/work/bb1170/RUN/b381737/data/FLEXPART/ACOS_australia/2009/11_unpacked/results_new.pkl",
        month="2009-11", 
        flux_path="/work/bb1170/static/CT2019/Flux3hour_1x1/CT2019B.flux1x1.",
        bioclass_path= "/home/b/b381737/python_scripts/master/open_data/data/OekomaskAU_Flexpart_2",
        time_coarse = None,
        boundary=[110.0, 155.0, -45.0, -10.0],
        data_outside_month=True
    )