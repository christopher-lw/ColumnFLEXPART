import os
import xarray as xr
import numpy as np
import argparse
import shutil
import pandas as pd
from datetime import datetime as dt
from columnflexpart.utils import in_dir
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
from typing import Union

def get_output_dirs(path: str, r: bool, dirs: list[str] = []) -> list[str]:
    """Collects all dir with flexpart output. Optionally searches recursively beneath path

    Args:
        path (str): Starting directory for search
        r (bool): Flag for recursive search
        dirs (list[str], optional): Directories to collect output in. Defaults to [].

    Returns:
        list[str]: List of paths of directories with FLEXPART output
    """    
    if r:
        if in_dir(path, "grid_time"):
            dirs.append(path)
        else: 
            for dir in os.listdir(path):
                dir = os.path.join(path, dir)
                if os.path.isdir(dir):
                    if in_dir(dir, "grid_time"):
                        dirs.append(dir)
                    else:
                        dirs.extend(get_output_dirs(dir, r, []))
    else:
        dirs = [path]
    return dirs

def check_names(dirs: list[str]) -> str:
    """Checks names of Flexpart output directories to set a sensible naming scheme

    Args:
        dirs (list[str]): List of paths of directories with FLEXPART output (from get_output_dirs)

    Returns:
        str: Naming scheme
    """    
    name_scheme = "by_dir"
    names = []
    for dir in dirs:
        if dir[-1] == "/": dir = dir[:-1]
        name = os.path.basename(dir)
        if name in names:
            name_scheme = "by_index"
            break
        names.append(name)
    return name_scheme
    

def group_releases(dir: str) -> tuple[str, list[list[str]], np.ndarray, np.ndarray, list[list[int]]]:  
    """Goes through the RELEASES.namelist file in dir and groups all the releases of one stations

    Args:
        dir (str): Flexpart output directory

    Returns:
        tuple[str, list[list[str]], np.ndarray, np.ndarray, list[list[int]]]: header of releases file, list with dims: (num_stations, num_release_per_station), longitude values of stations, latitude values of stations, partposit values for each station
    """    
    with open(os.path.join(dir, "RELEASES.namelist"), "r") as f:
        releases_file = f.read().split("&")[1:]    

    header = releases_file[0]
    releases = releases_file[1:]
    release_collections = []
    lon_list = []
    lat_list = []
    index_collections = []
    # date_list =  []
    # time_list = []
    for i, release in enumerate(releases):
        lines = release.splitlines()
        for line in lines:
            if "LON1" in line:
                lon = float(line.split("=")[1][:-1].strip())
            if "LAT1" in line:
                lat = float(line.split("=")[1][:-1].strip())
            # if "IDATE1" in line:
            #     date = float(line.split("=")[1][:-1].strip())
            # if "ITIME1" in line:
            #     time = float(line.split("=")[1][:-1].strip())
        # if new coordinate appears open new collection for new release
        if not lon in lon_list or not lat in lat_list: # or not date in date_list or not time in time_list:
            lon_list.append(lon)
            lat_list.append(lat)
            # date_list.append(date)
            # time_list.append(time)
            release_collections.append([])
            index_collections.append([])
        # add line to current release (last in release_collections) 
        index_collections[-1].append(i)  
        release_collections[-1].append(release)
    
    index_collections = np.array(index_collections, dtype=int)
    lon_list = np.array(lon_list, dtype=np.float32)
    lat_list = np.array(lat_list, dtype=np.float32)
    return header, release_collections, lon_list, lat_list, index_collections

def convert_partposit(dir: str) -> tuple[list[xr.Dataset], list[str]]:
    """Converts partpoist files into xarray.Datasets. Returns Data and respective origin binary filename

    Args:
        dir (str): Flexpart output directory

    Returns:
        tuple[list[xr.Dataset], list[str]]: Data, origin binary file names
    """    
    data = []
    files = []
    for file in os.listdir(dir):
        if "partposit" in file and not ".nc" in file:
            data.append(partposit_from_bin(os.path.join(dir, file)))
            files.append(file)
    return data, files
        

def get_name(dir: str, file_counter: int, i: int, name_scheme: str) -> str:
    """Constructs name of output directory depending on name scheme

    Args:
        dir (str): parent directory
        file_counter (int): Index countig total produced output directories
        i (int): Index counting produced dierctoies for current Flexpart output file
        name_scheme (str): States scheme to use for naming.

    Returns:
        str: Name for output directory
    """    
    if dir[-1] == "/": dir = dir[:-1]
    if name_scheme == "by_dir":
        name = f"{os.path.basename(dir)}_{i}"
    elif name_scheme == "by_index":
        name = f"FLEX_OUTPUT_{file_counter}"
    return name

def save_release(header: str, release_collection: list[str], save_dir: str):
    """Saves releases data into RELEASES.namelist file

    Args:
        header (str): header of RELEASES file
        release_collection (list[str]): list of single releases to save to file 
        save_dir (str): directory to save RELEASES.namelist to
    """    
    content = f"&{header}"
    for release in release_collection:
        content += f"&{release}"
    with open(os.path.join(save_dir, "RELEASES.namelist"), 'w') as f:
        f.write(content)

def copy_namelists(dir: str, save_dir: str):
    """Copies all useful textfiles of Flexpart output to the given directory

    Args:
        dir (str): Source
        save_dir (str): Destination
    """    
    for file in os.listdir(dir):
        if "namelist" in file and not "RELEASES" in file:
            shutil.copy(os.path.join(dir, file), os.path.join(save_dir, file))
        if "header_txt" in file:
            shutil.copy(os.path.join(dir, file), os.path.join(save_dir, file))

def get_footprint(dir: str) -> tuple[xr.Dataset, str]:
    """Finds and loads footprint file in Flexpart ouput directory

    Args:
        dir (str): Flexpart ouput directory

    Returns:
        tuple[xr.Dataset, str]: Loaded footprint, name of the file
    """    
    for file in os.listdir(dir):
        if "grid" in file:
            footprint_file = file 
            break
    footprint = xr.load_dataset(os.path.join(dir, footprint_file))
    return footprint, footprint_file

def partposit_from_bin(filename: str, nspec: int=1) -> xr.Dataset:
    """Converts binary partposit output file of FLEXPART to xr.Dataset.

    Args:
        filename (str): file to convert
        nspec (int, optional): Number of species. Defaults to 1.

    Returns:
        xr.Dataset: Converted binary file (dropped unuseful information)
    """    
    xmass_dtype = [('xmass_%d' % (i + 1), 'f4') for i in range(nspec)]
    # note age is calculated from itramem by adding itimein
    out_fields = [
                     ('pointspec', 'i4'), ('longitude', 'f4'), ('latitude', 'f4'), ('height', 'f4'),
                     ('itramem', 'i4'), ('topo', 'f4'), ('pvi', 'f4'), ('qvi', 'f4'),
                     ('rhoi', 'f4'), ('hmixi', 'f4'), ('tri', 'f4'), ('tti', 'f4')] + xmass_dtype
    raw_fields = [('begin_recsize', 'i4')] + out_fields + [('end_recsize', 'i4')]
    raw_rectype = np.dtype(raw_fields)
    recsize = raw_rectype.itemsize
    data_list = []
    with open(filename, "rb") as f:
        # The timein value is at the beginning of the file (skip fortran unformatted header)
        _ = f.read(12)
        # read the complete file
        data = f.read()
        read_records = int(len(data) / recsize)  # the actual number of records read
        chunk = np.ndarray(shape=(read_records,), buffer=data, dtype=raw_rectype)
        # Add the chunk to the out array
        data_list.append(chunk[:read_records])
    #convert to dataframe and drop
    df = pd.DataFrame(data = data_list[0])
    df = df.drop(df[df.longitude < -200].index)
    df = df.drop(["begin_recsize", "itramem", "topo", "pvi", "qvi", "rhoi", "hmixi", "tri", "tti", "xmass_1", "end_recsize"], axis=1)
    date = filename.rsplit("_")[-1]
    times = [np.datetime64(dt.strptime(str(date), "%Y%m%d%H%M%S"))]*len(df["pointspec"])
    df["time"] = times
    ids = np.arange(len(df["pointspec"]))
    df["id"] = ids
    df = df.set_index(["id", "time"])
    xarr = df.to_xarray()
    return xarr

def fix_pointspec(xarr: xr.Dataset) -> xr.Dataset:
    """Adjusts pointspec to start from 0

    Args:
        xarr (xr.Dataset): parposit data

    Returns:
        xr.Dataset: fixed partposit data
    """    
    pointspec = xarr.pointspec.values
    pointspec = pointspec - min(pointspec) + 1
    xarr = xarr.drop("pointspec")
    xarr["pointspec"] = (["id", "time"], pointspec) 
    return xarr

def fix_ids(xarr: xr.Dataset) -> xr.Dataset:
    """Adjusts id of particles to start from 0

    Args:
        xarr (xr.Dataset): partposit data

    Returns:
        xr.Dataset: fixed partposit data
    """    
    ids = xarr.id.values
    ids = ids - ids.min()
    xarr = xarr.assign_coords(dict(id = ids))
    return xarr

def combine_to_trajectories(dir: str):
    """Collects all partposit NetCDF files and Saves them to one trajectories.pkl file

    Args:
        dir (str): Directory with partposit files to collect
    """    
    partposit_files = []
    for file in os.listdir(dir):
        if "partposit" in file and ".nc" in file:
            partposit_files.append(os.path.join(dir, file))
    xr_trajectories = xr.open_mfdataset(partposit_files)
    pd_trajectories = xr_trajectories.to_dataframe().reset_index()
    pd_trajectories = pd_trajectories[~np.isnan(pd_trajectories.longitude)]
    pd_trajectories = pd_trajectories.set_index(["time", "id", "pointspec"])
    pd_trajectories.to_pickle(os.path.join(dir, "trajectories.pkl"))

def main(enumerated_dir, name_scheme, args, disable_tqdms):
    (i, dir) = enumerated_dir 
    # Filter RELEASES.namelist and copy other namelists
    header, release_collections, lon_list, lat_list, index_collections = group_releases(dir)
    footprint, footprint_file = get_footprint(dir)
    partposit_data, partposit_files = convert_partposit(dir)
    for j, release_collection in enumerate(tqdm(release_collections, desc = "Releases of directory", position=1, leave=False, disable=disable_tqdms)):
        # preparation
        file_counter = i*len(release_collections) + j
        name = get_name(dir, file_counter, j, name_scheme)
        save_dir = os.path.join(args.outdir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        # namelists and header
        save_release(header, release_collection, save_dir)
        copy_namelists(dir, save_dir)
        
        # footprint data
        release_indices = index_collections[j]
    
        release_footprint = footprint[dict(numpoint = release_indices, pointspec = release_indices)]
        release_footprint.to_netcdf(os.path.join(save_dir, footprint_file))
        # partposit data
        for partposit_single, file in tqdm(zip(partposit_data, partposit_files), desc = "Partposit files", total=len(partposit_files), position=2, leave=False, disable=disable_tqdms):
            # if the particles of this release are not inititialized yet continue
            # pointspec is the index of the release and characterizes its sounding and height (in the partposit files it starts with 1)
            if partposit_single.pointspec.isin(release_indices+1).sum() == 0: continue
            partposit_release = partposit_single.where(partposit_single.pointspec.isin(release_indices + 1), drop=True)
            partposit_release = fix_pointspec(partposit_release)
            partposit_release = fix_ids(partposit_release)
            partposit_release.to_netcdf(os.path.join(save_dir, f"{file}.nc"))
        combine_to_trajectories(save_dir)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script to spilt Flexpart output of one run with multiple stations.")
    parser.add_argument("dir", type=str, help="Flexpart output directory to split")
    parser.add_argument("outdir", type=str, help="Directory for split up data")
    parser.add_argument("--processes", type=int, default=1, help="Number of processes")
    parser.add_argument("-r", action="store_true", help="Flag to recursively search for flexpart output directories to split. All results will be saved to outdir")
    
    args = parser.parse_args()
    disable_main_tqdms = args.processes > 1

    dirs = get_output_dirs(args.dir, args.r)
    name_scheme = check_names(dirs)
    os.makedirs(args.outdir, exist_ok=True)

    # in case of no multi core processing use for loop
    if args.processes == 1:
        print("No multiprocessing")
        for enumerated_dir in enumerate(tqdm(dirs, desc = "Directories to split", position = 0)):
            main(enumerated_dir, name_scheme, args, disable_main_tqdms)

    # in case of multi core processing
    else:
        print("Using multiprocessing")
        partial_main = partial(main, name_scheme=name_scheme, args=args, disable_tqdms=disable_main_tqdms)
        enumerated_dirs = enumerate(dirs)
        with Pool(processes=args.processes)as p:
            _ = list(tqdm(p.imap(partial_main, enumerated_dirs), total=len(dirs)))