import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cf
from functools import cache
import copy
from datetime import datetime, date
#import geoplot
#import contextily as ctx

def to_tuple(convert_arg, convert_ind):
    def to_tuple_inner(function):
        def new_function(*args, **kwargs):
            args=list(args)
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
    if expand_to is not None and len(val)==1:
        val = val*expand_to#
    return val

def printval(inp):
    out={}
    exec(f"val={inp}", globals(), out)
    out = out["val"]
    print(f"{inp}: {out}")

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




def calc_enhancement(fp_data, ct_file, p_surf, startdate, enddate):

    def barometric(a, b):
        func = lambda x, y: x*np.exp(-(y/7800)) #Skalenhöhe von 7.8 km passt?
        return xr.apply_ufunc(func, a, b)

    def heights(p_surf, p):
        func = lambda x, y: np.log(y/x)*(-7800) #Skalenhöhe von 7.8 km passt?
        return xr.apply_ufunc(func, p_surf, p)
    #-----------------------------

    #test the right order of dates
    assert enddate > startdate, 'The startdate has to be before the enddate'

    #get Footprint
    DSFP = fp_data
    
    #DSFP.RELZZ1.values

    #read CT fluxes (dayfiles) and merge into one file
    first = True
    for date in pd.date_range(startdate, enddate):
        fileCT = ct_file+str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)+'.nc'
        DSCTfluxday = xr.open_mfdataset(fileCT, combine='by_coords',drop_variables = 'time_components', chunks="auto")
        if first:
            first = False
            DSCTflux = DSCTfluxday
        else:
            DSCTflux = xr.concat([DSCTflux,DSCTfluxday],dim = 'time')
    #calculate Satelite CO2 enhancement
            
    #sum flux components: 
    DSCT_totalflux = DSCTflux.bio_flux_opt+DSCTflux.ocn_flux_opt+DSCTflux.fossil_flux_imp+DSCTflux.fire_flux_imp
    DSCT_totalflux.name = 'total_flux'

    #can be deleted when footprint has -179.5 coordinate
    DSCT_totalflux = DSCT_totalflux[:,:,1:]#.compute()
    
    #from .interp can be deleted as soon as FP dim fits CTFlux dim, layers repeated???? therfore only first 36
    FP = DSFP.spec001_mr[0,:,:,0,:,:]
    dt = np.timedelta64(90,'m')
    FP = FP.assign_coords(time=(FP.time + dt))
    #flip time axis
    FP = FP.sortby('time')#.compute()

    #selct times of Footprint in fluxes
    DSCT_totalflux = DSCT_totalflux.sel(time=slice(FP.time.min(),FP.time.max()))
    
    #get pressure levels (boundaries of layer) of FP data, needed for interpolation on GOSAT levels
    numLayer = len(FP.pointspec)
    FP_pressure_layers = barometric(p_surf,np.append(np.array([0]),DSFP.RELZZ2[:].values))
    PW_FP = 1/(p_surf-0.1)*(FP_pressure_layers[0:-1]-FP_pressure_layers[1:]) 
    #FP_pressure_layers = np.array(range(numLayer,0,-1))*1000/numLayer

    #combine FP and emissions
    FPCO2_1 = xr.merge([FP, DSCT_totalflux])
    # 1/layer height*flux [mol/m²*s]*fp[s] -> mol/m³ -> kg/m³
    FPCO2_2 = 1/100*FPCO2_1.total_flux*FPCO2_1.spec001_mr*0.044 #dim : time, latitude, longitude, pointspec: 36
    #sum over time, latitude and longitude, remaining dim = layer
    FPCO2_tt = FPCO2_2.sum(dim=['time','latitude','longitude'])
    FPCO2_tt.name = 'CO2'
    FPCO2_tt = FPCO2_tt.to_dataset()

    CO2_enh_molfrac = FPCO2_tt.CO2.values
    CO2_enh_col = (PW_FP*CO2_enh_molfrac).sum()
    CO2_enh_col = CO2_enh_col*1e6
    return CO2_enh_col

def load_nc_partposit(dir_path, chunks=None):
    files = []
    for file in os.listdir(dir_path):
        if "partposit" in file and ".nc" in file:
            files.append(os.path.join(dir_path, file))
    xarr = xr.open_mfdataset(files, chunks=chunks)
    return xarr

class FlexDataset2():
    def __init__(self, nc_path, **kwargs):
        self._nc_path = nc_path
        self._dir = nc_path.rsplit("/", 1)[0]
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
            persist=False
        )
        self._kwargs.update(kwargs)
        self._plot_kwargs = self._kwargs.copy()
        [self._plot_kwargs.pop(key) for key in ['extent', 'ct_dir', 'ct_name_dummy', 'id_key', 'chunks', 'name', 'datakey', 'persist']]

        self.footprint = self.Footprint(self)
        if os.path.exists(os.path.join(self._dir, "trajectories.nc")):
            self.trajectories = self.Trajectories(self)
        else:
            print("No trajectory information found.")
            self.trajectories = None
        self.start, self.stop, self.release = self.get_metadata()
        self._background = None
        self._enhancement = None

    def get_metadata(self):
        #read things from the header and RELEASES.namelist
        with open(os.path.join(self._dir, 'header_txt'), 'r') as f:
            lines = f.readlines()
        ibdate, ibtime, iedate, ietime = lines[1].strip().split()[:4]

        start = datetime.strptime(iedate+ietime, "%Y%m%d%H%M%S")
        stop = datetime.strptime(ibdate+ibtime, "%Y%m%d%H%M%S")
        with open(os.path.join(self._dir, 'RELEASES.namelist'), 'r') as f:
            lines = f.readlines()[5: 13]
        
        release = dict()
        for i, line in enumerate(lines):
            lines[i] = line.split('=')[1].strip()[:-1].strip()
        
        release['start'] = datetime.strptime(lines[2] + lines[3], "%Y%m%d%H%M%S")
        release['stop'] = datetime.strptime(lines[0] + lines[1], "%Y%m%d%H%M%S")
        release['lon1'] = float(lines[4])
        release['lon2'] = float(lines[5])
        release['lat1'] = float(lines[6])
        release['lat2'] = float(lines[7])
        release['heights'] = np.mean([self.footprint.dataset.RELZZ1.values, self.footprint.dataset.RELZZ2.values], axis=0)

        return start, stop, release

    class Footprint():
        def __init__(self, outer):
            self._outer = outer
            self._nc_path = outer._nc_path
            self._dir = outer._dir
            self.extent = outer._kwargs['extent']
            self.chunks = outer._kwargs['chunks']
            self.name = outer._kwargs['name']
            self.datakey = outer._kwargs['datakey']

            self._plot_kwargs = outer._plot_kwargs
            
            self.dataset = xr.open_dataset(self._nc_path, chunks=self.chunks)
            self.dataarray = self.dataset[self.datakey]
            if self._outer._kwargs['persist']: self.dataarray = self.dataarray.persist()

            self._total = None
        
        def save_total(self):
            """Saves Footprints

            Args:
                include_sums (bool, optional): _description_. Defaults to True.
            """        
            sum_path = os.path.join(self._dir, "Footprint_total.nc")
            self._total.to_netcdf(sum_path)
                
        
        def calc_total(self):
            """Calculates total Footprints
            """
            self._total = None
            self._total = self.dataarray.sum(dim=["time", "pointspec"]).compute()
        
        def load_total(self):
            """Loads Footprints from directory of DataSet data
            """
            self._total = None
            path = os.path.join(self._dir, "Footprint_total.nc")
            self._total = xr.load_dataarray(path)
            
        def total(self):
            """Get footprints from either loading of calculation
            """ 
            if self._total is None:
                try:
                    self.load_total()
                except FileNotFoundError:
                    self.calc_total()
                    self.save_total()
            return self._total

        def plot(self, ax=None, time=None, pointspec=None, plot_func=None, plot_station=True, station_kwargs=dict(color="black"), **kwargs):
            return_fig = False
            fig = None
            
            plot_kwargs = self._plot_kwargs
            plot_kwargs.update(kwargs)
            if ax is None:
                fig, ax = self._outer.subplots()
                return_fig = True
            footprint = self.sum(time, pointspec)
            footprint = footprint.where(footprint != 0)[:,:,...]
            if plot_func is None:
                footprint.plot(ax=ax, **plot_kwargs)
            else:
                getattr(footprint.plot, plot_func)(ax=ax, **plot_kwargs)
            if plot_station:
                ax.scatter(self._outer.release['lon1'], self._outer.release['lat1'], **station_kwargs)
            if return_fig:
                return fig, ax
            else:
                return ax

        
        @to_tuple(["time", "pointspec"],[1, 2])
        @cache
        def sum(self, time=None, pointspec=None):
            footprint = None
            if time is None and pointspec is None:
                footprint = self.total()
            elif time is None:
                footprint = self.dataarray.sum(dim=["time"])
                footprint = footprint.isel(dict(pointspec = list(pointspec))).sum(dim=["pointspec"]).compute()
            elif pointspec is None:
                footprint = self.dataarray.sum(dim=["pointspec"])
                footprint = footprint.isel(dict(time = list(time))).sum(dim=["time"]).compute()
            return footprint
 
    class Trajectories:
        def __init__(self, outer):
            self._outer = outer
            self._dir = outer._dir
            self._ct_dir = outer._kwargs['ct_dir']
            self._ct_dummy = outer._kwargs['ct_name_dummy']
            self._id_key = outer._kwargs['id_key']
            self.dataset = xr.open_dataset(os.path.join(self._dir, "trajectories.nc"))
            self.dataset = self.dataset
            self.dataframe = self.dataset.to_dataframe().reset_index()
            self._min_time = self.dataframe.time.min()
            self._max_time = self.dataframe.time.max()
            self.endpoints = None
            self.ct_data = None

        def ct_endpoints(self, border=None, ct_dir=None, ct_dummy=None):
            if border is not None:
                df_outside = self.dataframe[
                    ((self.dataframe.longitude < border[0]) | 
                    (self.dataframe.longitude > border[1]) | 
                    (self.dataframe.latitude < border[2]) | 
                    (self.dataframe.latitude > border[3]))]
                df_outside = df_outside.loc[df_outside.groupby(self._id_key)["time"].idxmax()].reset_index().drop(columns="index")
                df_inside = self.dataframe[~self.dataframe[self._id_key].isin(df_outside[self._id_key])]
                df_inside = df_inside.loc[df_inside.groupby(self._id_key)["time"].idxmin()]
                df_total = pd.concat([df_outside, df_inside])
            
            else:
                df_total = self.dataframe.loc[self.dataframe.groupby(self._id_key)["time"].idxmin()]
            df_total.attrs["border"] = border

            _ = self.load_ct_data(ct_dir, ct_dummy)

            variables = ["time", "longitude", "latitude"]
            for i, var in enumerate(variables):
                ct_vals = self.ct_data[var].values
                df_vals = df_total[var].values
                diff = np.abs(df_vals[:,None] - ct_vals[None, :])
                inds = np.argmin(diff, axis=-1)
                df_total.insert(loc=1, column=f"ct_{var}", value=inds)
            df_total.insert(loc=1, column="pressure_height", value=10130 * self.pressure_factor(df_total.height))
            df_total.insert(loc=1, column = "ct_height", value=df_total.apply(lambda x: np.where(np.sort(np.append(np.array(self.ct_data.pressure[x.ct_time,:,x.ct_latitude,x.ct_longitude]), x.pressure_height))[::-1] == x.pressure_height)[0] - 1, axis=1))
            
            self.endpoints = df_total.sort_values(self._id_key)
            self.endpoints = self.endpoints[np.sign(self.endpoints.pointspec) == 1]
            return self.endpoints

        def load_ct_data(self, ct_dir=None, ct_dummy=None):
            if ct_dir is not None:
                self._ct_dir = ct_dir
            if ct_dummy is not None:
                self._ct_dummy = ct_dummy
            file_list = []
            for date in np.arange(self._min_time, self._max_time + np.timedelta64(1, "D"), dtype='datetime64[D]'):
                date = str(date)
                file_list.append(os.path.join(self._ct_dir, self._ct_dummy + date + ".nc"))
            ct_data = xr.open_mfdataset(file_list, combine="by_coords")
            
            self.ct_data = ct_data[["co2", "pressure"]].compute()
            return self.ct_data

        def save_endpoints(self, name="endpoints.pkl", dir=None):
            if dir is None:
                dir = self._dir
            save_path = os.path.join(dir, name)
            self.endpoints.to_pickle(save_path)
            print(f"Saved endpoints to {save_path}")
        
        def load_endpoints(self, name=None, dir=None):
            """Load endpoints from endpoints.pkl file or file with personalized name.

            Args:
                name (str, optional): Name of output. Defaults to None (results in endpoints.pkl).
                dir (dir, optional): Directory for file. Defaults to None.
            """            
            if name is None:
                name="endpoints.pkl"
            if dir is None:
                dir = self._dir
            read_path = os.path.join(dir, name)
            self.endpoints = pd.read_pickle(read_path).sort_values(self._id_key)

        def co2_from_endpoints(self, exists_ok=True, extent=None, ct_dir=None, ct_dummy=None, pressure_weight=True):
            """Returns CO2 at positions of the endpoints of the particles. Result is also saved to endpoints. Pressure weights are also calculated based on the pointspec value of the particles.

            Args:
                exists_ok (bool, optional): Flag for recalculation if co2 was allready calculated. Defaults to True.
                extent (list, optional): Borders for optional endpoint calculateion if no endpoints exist so far. Defaults to None.
                ct_dir (str, optional): Directory for carbon tracker data. Defaults to None.
                ct_dummy (str, optional): Start of each Carbon Tracker file util the time stamp. Defaults to None.
                pressure_weight (bool, optional): Flag to also add pressure weights in the enpoints dataframe. Defaults to True.

            Returns:
                _type_: _description_
            """                       
            if self.endpoints is None:
                print("No endpoints found. To load use load_endpoints(). Calculation of endpoints...")
                _ = self.ct_endpoints(extent, ct_dir, ct_dummy)
                print("Done")
            if self.ct_data is None:
                _ = self.load_ct_data(ct_dir, ct_dummy)

            if "co2" in self.endpoints.columns:
                print("'co2' is allready in endpoints. To calculate again set exists_ok=Flase") or "n"
                if not exists_ok:
                    self.endpoints.drop("co2")
                    self.endpoints.insert(loc=1, column = "co2", value=self.endpoints.apply(lambda x: self.ct_data.co2[x.ct_time, x.ct_height, x.ct_latitude, x.ct_longitude].values, axis=1))
            else:
                self.endpoints.insert(loc=1, column = "co2", value=self.endpoints.apply(lambda x: self.ct_data.co2[x.ct_time, x.ct_height, x.ct_latitude, x.ct_longitude].values, axis=1))
            
            if pressure_weight:
                pointspec = self.endpoints.pointspec.values.astype(int)
                heights = self._outer.release["heights"][pointspec-1]
                pressures = self.pressure_factor(heights)
                pressures = pressures/pressures.sum()
                self.endpoints.insert(loc=1, column="pressure_weight", value=pressures)

            return self.endpoints.co2.values
            

        def pressure_factor(self,
            h,
            Tb = 288.15,
            hb = 0,
            R = 8.3144598,
            g = 9.80665,
            M = 0.0289644,
            ):
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
            factor = np.exp(-g * M * (h - hb)/(R * Tb))
            return factor

    def add_map(self, ax, feature_list = [cf.COASTLINE, cf.BORDERS, [cf.STATES, dict(alpha=0.1)]],
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
        ax.set_extent(self._kwargs["extent"], crs=ccrs.PlateCarree()) if self._kwargs["extent"] is not None else None
        for feature in feature_list:
            feature, kwargs = feature if isinstance(feature, list) else [feature, dict()]
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

    def subplots(self, *args, **kwargs):
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
    
    def enhancement(self, ct_file=None, p_surf=1013, allow_read=True):
        """Returns enhancement based on Carbon Tracker emmission data and the footprint. Is either calculated, loaded, or read from class.

        Args:
            ct_file (str, optional): Name of Carbon Tracker (only shared part not the time stamp). Defaults to None.
            p_surf (float, optional): Surface pressure in millibars. Defaults to 1013.
            allow_read (bool, optional): Flag to allow reading of result from file . Defaults to True.

        Returns:
            float: Enhancement of CO2 in ppm
        """        
        if self._enhancement is None:
            try:
                assert allow_read
                with open(os.path.join(self._dir,"enhancement.txt"), "r") as f:
                    self._enhancement = float(f.read())
            except (FileNotFoundError, AssertionError):    
                self._enhancement = calc_enhancement(self.footprint.dataset, ct_file, p_surf, self.stop.date(), self.start.date())
                with open(os.path.join(self._dir,"enhancement.txt"), "w") as f:
                    f.write(str(self._enhancement))
        
        return self._enhancement
    
    def background(self, allow_read=True):
        """Returns background calculation in ppm based on trajectories in trajectories.pkl file. Is either loaded from file, calculated or read from class. 

        Args:
            allow_read (bool, optional): Flag to allow reading of result from file. Defaults to True.

        Returns:
            float: Background CO2 in ppm
        """        
        if self._background is None:
            try:
                assert allow_read
                with open(os.path.join(self._dir, "background.txt"), "r") as f:
                    self._background = float(f.read())
            except (FileNotFoundError, AssertionError):    
                self._background = np.sum(self.trajectories.endpoints.co2 * self.trajectories.endpoints.pressure_weight)[0]

                with open(os.path.join(self._dir, "background.txt"), "w") as f:
                    f.write(str(self._background))
        return self._background

    