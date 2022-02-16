import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
#import geoplot
#import contextily as ctx

def load_era_to_gdf(file, data_keys=["u","v","w"], coarsen=None):
    '''
    Returns a geodataframe with geomertry Point(long, lat) and columns: ["time", "level"] + data_keys.
    
        Parameters:
            file (str): Path to nc file 
            data_keys (list): List of str naming wind field coordinates to extract
            coarsen (int/list): If int is given longitude and latitude are averaged over with neighbors according to int.
                                If list is given as [x, y] then longitude (x) and latitude (y) are averaged over differnent numbers of neighbors.

        Returns:
            geodataframe (gpd.GeoDataFrame): Frame of era data
    '''
    #
    assert (coarsen is None or type(coarsen) in (int, list))
    coarsen = [coarsen, coarsen] if type(coarsen) == int else coarsen

    #load file into xarray
    dataset = xr.open_mfdataset(file, combine='by_coords',concat_dim='None')
    #center coordinates
    dataset.longitude.values = (dataset.longitude.values + 180) % 360 - 180
    #extract u,v,w values
    if coarsen is not None:
        data_dict = {key: dataset[key].coarsen(longitude=coarsen[0], latitude=coarsen[1], boundary='trim').mean() for key in data_keys}
        #data_dict = {key: dataset[key][::100,...].coarsen(longitude=coarsen[0], latitude=coarsen[1], boundary='trim').mean() for key in data_keys}
    else:
        data_dict = {key: dataset[key] for key in data_keys}
    
    #convert to frames and remove unnecessary columns
    frame_list = [data_dict[key].to_dataframe(key).reset_index() for key in data_keys]
    for i in range(len(frame_list)):
        if i > 0:
            frame_list[i] = frame_list[i].drop(["time", "level", "longitude", "latitude"], axis=1)
    #merge u,v,w into one pd.DataFrame
    dataframe = pd.concat(frame_list, axis=1)
    #use merge instead
    #build GeoDataFrame
    geodataframe = gpd.GeoDataFrame(dataframe[["time", "level"] + data_keys],
            geometry=gpd.points_from_xy(dataframe.longitude, dataframe.latitude), 
            crs="EPSG:4326")
    
    return geodataframe

def plot_hwind_field(fig, ax, gdf, time, level, normalize=False, plot_range=None, use_cmap=True, **kwargs):
    '''
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
    '''
    ax.set_title("Horizontal wind field ({} at {} hPa)".format(time, level))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if plot_range is not None:
        gdf = gdf.cx[plot_range[0][0]:plot_range[0][1], 
                    plot_range[1][0]:plot_range[1][1]]

    sel_gdf = gdf.loc[(gdf['time'] == time) & (gdf['level'] == level)]
    #sel_gdf = sel_gdf.loc[]
    field = np.array([sel_gdf["u"].values, sel_gdf["v"].values])
    long = sel_gdf.geometry.x
    lat = sel_gdf.geometry.y
    
    norm = np.linalg.norm(field, axis=0)
    if normalize:
        field = field / norm
    if use_cmap:
        c = ax.quiver(long, lat, *field, 
            norm, **kwargs)
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
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).cx[plot_range[0][0]:plot_range[0][1], 
                plot_range[1][0]:plot_range[1][1]]
    else:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    if country is not None:
        world = world[world.name == "Australia"]
        
    world.plot(ax=ax, **kwargs)
    return fig, ax

def plot_hwind_field_season(fig, ax, gdf, season, hemisphere, year, level, normalize=False, plot_range=None, use_cmap=True, **kwargs):
    '''
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
    '''
    assert hemisphere in ["north", "south"]
    if hemisphere == "north": 
        season_dict = { "summer":["06", "07", "08"], 
                        "autumn":["09", "10", "11"], 
                        "winter":["12", "01", "02"],
                        "spring":["03", "04", "05"]}
    elif hemisphere == "south":
        season_dict = { "summer":["12", "01", "02"],
                        "autumn":["03", "04", "05"],
                        "winter":["06", "07", "08"],
                        "spring":["09", "10", "11"]}

    ax = [ax] if type(ax) not in [list, np.ndarray] else ax
    season = [season] if type(season) is not list else season

    for i, a in enumerate(ax):
        a.set_title("Horizontal wind field in {} ({}ern hemisphere) at {} hPa, {} ".format(season[i], hemisphere, 
                    level, year))
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")

    if plot_range is not None:
        gdf = gdf.cx[plot_range[0][0]:plot_range[0][1], 
                    plot_range[1][0]:plot_range[1][1]]
    
    for i, s in enumerate(season):
        s_months = season_dict[s]
        print([str(y if m not in ["01", "02"] else y+1) + "-" + m +"-01" for y in year for m in  s_months])
        s_data = [gdf.loc[(gdf['time'] == str(y if m not in ["01", "02"] else y+1) + "-" + m +"-01") & (gdf['level'] == level)] for y in year for m in  s_months]
        
        long = s_data[0].geometry.x
        lat = s_data[0].geometry.y
        
        s_data = pd.concat(s_data)
        s_data = s_data.groupby(['geometry'], sort=False).mean()
        
        field = np.array([s_data["u"].values, s_data["v"].values])
        norm = np.linalg.norm(field, axis=0)
        
        if normalize:
            field = field / norm
        if use_cmap:
            c = ax[i].quiver(long, lat, *field, 
                norm, **kwargs)
            cbar = fig.colorbar(c, ax=ax[i])
            cbar.set_label("Wind velocity [m/s]")
        else:
            c = ax[i].quiver(long, lat, *field, **kwargs)
    
        
    return fig, ax

def plot_hwind_field_month(fig, ax, gdf, month, year, level, normalize=False, plot_range=None, use_cmap=True, **kwargs):
    '''
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
    '''
    month = [month] if type(month) is not list else month
    month = [[m for m in month]] if type(month[0]) is not list else month
    for i, l in enumerate(month):
        for j, m in enumerate(l):
            month[i][j] = str(m)
            month[i][j] = "0"+ month[i][j] if len(month[i][j]) == 1 else month[i][j]
    ax = [ax] if type(ax) not in [list, np.ndarray] else ax

    for i, a in enumerate(ax):
        a.set_title("Horizontal wind field (months: {}, years: {}, level: {} hPa)".format(month[i], year, level))
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")

    if plot_range is not None:
        gdf = gdf.cx[plot_range[0][0]:plot_range[0][1], 
                    plot_range[1][0]:plot_range[1][1]]
    
    for i, months in enumerate(month): 
        m_data = [gdf.loc[(gdf['time'] == str(y) + "-" + m +"-01") & (gdf['level'] == level)] for y in year for m in  months]
        
        long = m_data[0].geometry.x
        lat = m_data[0].geometry.y
        
        m_data = pd.concat(m_data)
        m_data = m_data.groupby(['geometry'], sort=False).mean()
        
        field = np.array([m_data["u"].values, m_data["v"].values])
        norm = np.linalg.norm(field, axis=0)
        
        if normalize:
            field = field / norm
        if use_cmap:
            c = ax[i].quiver(long, lat, *field, 
                norm, **kwargs)
            cbar = fig.colorbar(c, ax=ax[i])
            cbar.set_label("Wind velocity [m/s]")
        else:
            c = ax[i].quiver(long, lat, *field, **kwargs)
    
    '''cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Wind velocity [m/s]")'''
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
    t, ind = np.unique(dataset.time_components.values[:,:avg_index], axis=0, return_index=True)
    err_flag = False
    for i in range(len(t)-1):
        err = None

        if err_mode == "std_err":
            try:
                err = np.mean(dataset.value_std_dev.values[ind[i]: ind[i+1]])/np.sqrt(ind[i+1]-ind[i])
            except AttributeError:
                err_flag = True
        elif err_mode == "std_dev":
            err = np.std(dataset.value.values[ind[i]: ind[i+1]])

        err_list.append(err)
        t_list.append(dataset.time.values[ind[i]].astype("datetime64[{}]".format(keys[avg_index-1])))
        v = np.mean(dataset.value.values[ind[i]: ind[i+1]])
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
    #get data of ann means
    data = pd.read_csv("/home/clueken/master/open_data/data/co2_annmean_gl.csv", ",", header=0)
    #extract time of means as YYYY-07-01
    years = data.year.to_numpy(dtype="str").astype("datetime64") + np.array("06", dtype="timedelta64[M]")
    #extract means
    mean = data["mean"].to_numpy()
    #find indices thar are closest to times of data
    ind = np.sort(np.argsort(np.abs(times[:, None] - years[None, :]), axis=-1)[:,:2], axis=-1)
    #calculate point of interpolation
    deltas = (times - years[ind[:,0].T])
    frac = deltas / np.timedelta64(1, "Y").astype(deltas.dtype)
    #do the interpolation
    interpolation = mean[ind[:,0].T] + (mean[ind[:,1].T] - mean[ind[:,0].T])*frac
    #subtract background
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
    full_times = np.arange(min_time, max_time, dtype=f'datetime64[{type}]')
    df = pd.DataFrame({"times":full_times})
    
    times = times.astype(f'datetime64[{type}]')
    
    df_val = pd.DataFrame({"times":times, "values":values})
    df = df.merge(df_val, on="times", how="outer")
    return df

def add_map(ax, 
            feature_list = [cf.COASTLINE, cf.BORDERS, [cf.STATES, dict(alpha=0.1)]],
            extent = None,
            **grid_kwargs,
           ):
    """Add map to axes using cartopy.

    Args:
        ax (Axes): Axes to add map to
        feature_list (list, optional): Features of cartopy to be added. Defaults to [cf.COASTLINE, cf.BORDERS, [cf.STATES, dict(alpha=0.1)]].
        extent (list, optional): list to define region ([lon1, lon2, lat1, lat2]). Defaults to None.

    Returns:
        Axes: Axes with added map
        Gridliner: cartopy.mpl.gridliner.Gridliner for further settings
    """    
    ax.set_extent(extent, crs=ccrs.PlateCarree()) if extent is not None else None
    for feature in feature_list:
        feature, kwargs = feature if type(feature) == list else [feature, dict()]
        ax.add_feature(feature, **kwargs)
    grid = True
    gl = None
    try:
        grid = grid_kwargs["grid"]
        grid_kwargs.pop("grid", None)
    except KeyError:
        pass
    if grid:
        grid_kwargs =  dict(draw_labels=True, dms=True, x_inline=False, y_inline=False) if not bool(grid_kwargs) else grid_kwargs
        gl = ax.gridlines(**grid_kwargs)
        gl.top_labels = False
        gl.right_labels = False
    return ax, gl

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
        crs=crs
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
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
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
    assert country in gdf.name.values, f"No country found with name {country} in GeoDataFrame"
    ov_count = gdf[gdf.name == country]
    ov_rest = gdf[gdf.name != country]
    ov_other = gdf[(gdf.name != country) & (gdf.name.notnull())]
    ov_sea = gdf[gdf.name.isnull()]
    ret = dict(
        country = ov_count, 
        rest = ov_rest, 
        other_countries = ov_other, 
        sea = ov_sea)
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
    xarr = xarr.where((xarr.longitude >= lon1) & (xarr.longitude <= lon2), drop=True)
    xarr = xarr.where((xarr.latitude >= lat1) & (xarr.latitude <= lat2), drop=True)
    return xarr