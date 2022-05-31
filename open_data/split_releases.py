from email import parser
from fileinput import FileInput
import os
from sre_parse import parse_template
from xxlimited import foo
import xarray as xr
import numpy as np
import argparse
import shutil
import pandas as pd
from datetime import datetime as dt


def partposit_from_bin(filename, nspec=1):
    """Converts binary partposit output file of FLEXPART to nc file. Optionally removes binary file.

    Args:
        filename (str): file to convert
        nspec (int, optional): number of species. Defaults to 1.
        remove_bin (bool, optional): Wether bin file should be removed. Defaults to False.
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script to spilt Flexpart output of one run with multiple stations.")
    parser.add_argument("dir", type=str, help="Flexpart output directory to split")
    
    args = parser.parse_args()

    # Extract RELEASES files for each release sight
    with open(os.path.join(args.dir, "RELEASES.namelist"), "r") as f:
        releases_file = f.read().split("&")[1:]    

    header = releases_file[0]
    releases = releases_file[1:]
    release_collections = []
    lon_list = []
    lat_list = []

    for release in releases:
        lines = release.splitlines()
        for line in lines:
            if "LON1" in line:
                lon = float(line.split("=")[1][:-1].strip())
            if "LAT1" in line:
                lat = float(line.split("=")[1][:-1].strip())
        if not lon in lon_list or not lat in lat_list:
            lon_list.append(lon)
            lat_list.append(lat)
            release_collections.append([])
        release_collections[-1].append(release)

    for i, release_collection in enumerate(release_collections):
        release_dir = os.path.join(args.dir, f"RELEASES_{i}")
        os.makedirs(release_dir)
        RELEASES = "&" + header
        for release in release_collection:
            RELEASES += "&" + release
        with open(os.path.join(release_dir, "RELEASES.namelist"), "w") as f:
            f.write(RELEASES)

        for file in os.listdir(args.dir):
            if "namelist" in file and not "RELEASES" in file:
                shutil.copy(os.path.join(args.dir, file), os.path.join(release_dir, file))

    for file in os.listdir(args.dir):
        if "grid" in file:
            footprint_file = file 
            break

    release_ind_collection = []
    footprint = xr.load_dataset(os.path.join(args.dir, footprint_file))    
    for i, lat in enumerate(np.unique(footprint.RELLAT1.values)):
        release_ind = footprint.numpoint[footprint.RELLAT1 == lat].values
        release_ind_collection.append(release_ind)
        release_set = footprint[dict(numpoint = release_ind, pointspec = release_ind)]
        release_set.to_netcdf(os.path.join(args.dir, f"RELEASES_{i}", footprint_file))

    for file in os.listdir(args.dir):
        if not "partposit" in file or ".nc" in file: continue
        partposit_data = partposit_from_bin(os.path.join(args.dir, file))
        for i, release_ind in enumerate(release_ind_collection):
            partposit_set = partposit_data.where(partposit_data.pointspec.isin(release_ind + 1))
            partposit_set.to_netcdf(os.path.join(args.dir, f"RELEASES_{i}", file+".nc"))
    
    