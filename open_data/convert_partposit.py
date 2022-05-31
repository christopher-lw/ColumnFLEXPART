import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as dt
from tqdm.auto import tqdm

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
    else:
        df["id"] = df.pointspec.values
    df = df.set_index(["id", "time"])
    xarr = df.to_xarray()
    if remove_bin:
        os.remove(filename)
        print(f"Deleted binary file {filename}")
    xarr.to_netcdf(filename + ".nc")
    print(f"Saved netcdf file to {filename}.nc")
    
def repair_trajectories(dataframe, offset=1e2):
    dataframe = dataframe[~ np.isnan(dataframe.longitude)]
    # get information tor trajectory extrapolation
    df_ref1_full = dataframe[dataframe.time == np.unique(dataframe.time)[::-1][1]]
    df_ref2_full = dataframe[dataframe.time == np.unique(dataframe.time)[::-1][0]]
    dfs = []
    for time in tqdm(np.unique(dataframe.time)[::-1][2:]):
        
        # get dataframe of timestep
        df_id = dataframe[dataframe.time == np.datetime64(time)]
        df_full = df_id.drop(columns="id")
        df_full.insert(loc=1, column="id", value=np.nan)
        
        for pointspec in tqdm(np.unique(dataframe.pointspec), leave=False):
            if np.isnan(pointspec):
                continue
            # choose data of release
            df = df_full[df_full.pointspec == pointspec]
            df_ref1 = df_ref1_full[df_ref1_full.pointspec == pointspec]
            df_ref2 = df_ref2_full[df_ref2_full.pointspec == pointspec]
            df_ref2 = df_ref2.loc[df_ref2.id.isin(df_ref1.id)]
            # if no particle is missing, dont do anything
            if len(df) == len(df_ref1):
                i = df_full[df_full.pointspec==pointspec].index
                df_full.loc[i, "id"] = df_ref1.id.values
                continue
            # extract positions of particles, scale height to degrees for consistent distance calculation
            pos = np.array([df.longitude, df.latitude, df.height/1e3]).T
            pos_ref1 = np.array([df_ref1.longitude, df_ref1.latitude, df_ref1.height/1e3]).T
            pos_ref2 = np.array([df_ref2.longitude, df_ref2.latitude, df_ref2.height/1e3]).T
            # extrapolation
            pos_ref = pos_ref1 + (pos_ref1-pos_ref2)
            # calculate diffenernces between extrapolations and new values
            diff = np.linalg.norm(pos[:, None, :] - pos_ref[None, :, :], axis=-1)
            shape = diff.shape
            # penalty for impossible values
            diff = diff + offset * (np.tri(*shape, k=-1) + np.ones(shape) - np.tri(*shape, k=shape[1]-shape[0]+1))
            
            # sort to find best match
            match_ind = np.argsort(diff, axis=-1)
            match_diff = np.sort(diff, axis=-1)
            
            # get scores to compare by
            scores = calculate_score(match_diff)
            # get unique index prediction for each new particle
            match_ind = remove_duplicates(scores, match_ind)
            
            # get and assign ids to particles
            ids = df_ref1.id.values[match_ind]
            i = df_full[df_full.pointspec==pointspec].index
            df_full.loc[i, "id"] = ids

        df_ref1_full = df_full.copy()
        df_ref2_full = df_ref1_full
        dfs.append(df_full)
    df = pd.concat(dfs, axis=0)
    
    return df

def remove_duplicates(scores, match_ind):
    inds = match_ind[:, 0].copy()
    candidate_position = np.ones(len(inds)).astype(int)
    while True:
        #find new positions of reoccuring indices
        vals, counts = np.unique(inds, return_counts=True)
        if (counts != np.ones_like(counts)).any():
            non_unique_vals = vals[counts != 1]
            flag = inds == non_unique_vals[0]
        else:
            #stop if no idices reoccur
            break
        #choose scores of recurring indices
        score = scores[np.arange(len(scores)), candidate_position-1]*flag.astype(int)
        #find best score in candidates
        flag[np.argmax(score)] = 0
        #assign canditades to rest
        inds[flag] = match_ind[flag, candidate_position[flag]].copy()
        #assign new candidates
        candidate_position[flag] = candidate_position[flag] + 1
    return inds

def calculate_score(diff):
    d = np.abs(diff)    
    return 1e6-d

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def in_dir(path, string):
    for file in os.listdir(path):
        if string in file:
            return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to convert partposit files in directorz to nc files.")
    parser.add_argument("dir_path", type=str, help="Directory with binary files to convert.")
    parser.add_argument('--rm', action='store_true', help="Flag to remove binary files.")
    parser.add_argument('--use_pointspec', action="store_true", help="Flag to use pointspec as id")
    #parser.add_argument('--multiple', action="store_true", help="Flag to look for multiple directories to convert in dir_path")
    args = parser.parse_args()
    dir_paths = [args.dir_path]

    found_file = False
    for file in os.listdir(dir_paths[0]):
        if "grid_time" in file and ".nc" in file:
            found_file = True
    if not found_file: 
        dir_paths = listdir_fullpath(args.dir_path)
        print(f"Found following directories to convert:\n {dir_paths}")
        inp = input("Continue converting files? ([y]/n) ") or "y"
        assert inp == "y", "Insert y to continue."

    for dir_path in dir_paths:
        if not in_dir(dir_path, "partposit"):
            print(f"No partpoist files for {dir_path}")
            continue

        for file in os.listdir(dir_path):
            if "partposit" in file and not "nc" in file:
                path = os.path.join(dir_path, file)
                if os.path.exists(path + ".nc"):
                    print(f"{path} allready converted.")
                    if args.rm:
                        os.remove(path)
                        print(f"Deleted binary file {path}")
                    continue
                print(file)
                partposit_bin_to_nc(path, remove_bin=args.rm, use_pointspec=args.use_pointspec)
        
        files = []
        for file in os.listdir(dir_path):
            if "partposit" in file and ".nc" in file:
                files.append(os.path.join(dir_path, file))
        
        xarr = xr.open_mfdataset(files)
        df = xarr.to_dataframe().reset_index()
        if not args.use_pointspec:
            if os.path.exists(os.path.join(dir_path, "trajectories.nc")):
                print("trajectories.nc file allready exists.")
                continue
            else:
                print("Repairing trajectories")
                df = repair_trajectories(df)
        df = df[~np.isnan(df.longitude)]
        df = df.set_index(["time", "id"])
        xarr = df.to_xarray()
        xarr.to_netcdf(os.path.join(dir_path, "trajectories.nc"))
        print(f"Full trajectories saved to: {dir_path}/trajectories.nc")