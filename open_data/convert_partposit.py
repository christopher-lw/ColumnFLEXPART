import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as dt

def partposit_bin_to_nc(filename, nspec=1, remove_bin=False, use_pointspec=False):
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
    if not use_pointspec:
        ids = np.arange(len(df["pointspec"]))
        df["id"] = ids
        df = df.set_index(["id", "time"])
    else:
        df = df.set_index(["pointspec", "time"])
    xarr = df.to_xarray()
    if remove_bin:
        os.remove(filename)
        print(f"Deleted binary file {filename}")
    xarr.to_netcdf(filename + ".nc")
    print(f"Saved netcdf file to {filename}.nc")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to convert partposit files in directorz to nc files.")
    parser.add_argument("dir_path", type=str, help="Directory with binary files to convert.")
    parser.add_argument('--rm', action='store_true', help="Flag to remove binary files.")
    parser.add_argument('--use_pointspec', action="store_true", help="Flag to use pointspec as id")
    args = parser.parse_args()
    dir_path = args.dir_path
    for file in os.listdir(dir_path):
        if "partposit" in file and not "nc" in file:
            path = os.path.join(dir_path, file)
            if os.path.exists(path + ".nc"):
                print(f"{path} allready converted.")
                continue
            print(file)
            partposit_bin_to_nc(path, remove_bin=args.rm, use_pointspec=args.use_pointspec)