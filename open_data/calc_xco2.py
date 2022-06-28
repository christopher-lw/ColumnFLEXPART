import argparse
import os
import pandas as pd
import numpy as np

from open_data_utils import FlexDataset2
from tqdm.auto import tqdm

def find_nc_files(dir, file_list=[]):
    found_file = False
    for file in os.listdir(dir):
        if "grid_time" in file:
            file_list.append(os.path.join(dir, file))
            return
    if not found_file:
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            if os.path.isdir(path):
                find_nc_files(path, file_list)
    return file_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to calculate the enahancements and backgrounds of multiple flexpart runs")
    
    parser.add_argument("dir", type=str, help="Directory beneath all flexpart runs are calculated")
    parser.add_argument("flux_file", type=str, help="File start for multiplication with footprints (until timestap)")
    parser.add_argument("conc_dir", type=str, help="Directory for concatenations for background calculation")
    parser.add_argument("conc_name", type=str, help="Name of files in conc_dir (until timestamp)")
    parser.add_argument("--boundary", type=float, nargs="+", default=None, help="Boundary to cut out for investigation [left, right, bottom, top]")

    args = parser.parse_args()
    files = find_nc_files(args.dir)

    # dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})
    # cluster = SLURMCluster(name='dask-cluster',
    #                         cores=8,
    #                         processes=8,
    #                         n_workers=8,
    #                         memory='10GB',
    #                         interface='ib0',
    #                         queue='interactive',
    #                         project='bb1170',
    #                         walltime='12:00:00',
    #                         asynchronous=0)
    # client = Client(cluster)
    # print(client)

    times = []
    enhancements = []
    backgrounds = []
    directories = []



    for file in tqdm(files):
        if not "trajectories.pkl" in os.listdir(os.path.dirname(file)):
            continue
        dir = os.path.dirname(file)        
        try: 
            fd = FlexDataset2(file, ct_dir=args.conc_dir, ct_name_dummy=args.conc_name, chunks=dict(time=20, pointspec=4))

            enhancement = fd.enhancement(allow_read=False, ct_file=args.flux_file, boundary=args.boundary)

            tr = fd.trajectories
            tr.ct_endpoints(boundary=args.boundary)
            tr.co2_from_endpoints(boundary=args.boundary)
            tr.save_endpoints()
            background = fd.background(allow_read=False, boundary=args.boundary)

            times.append(fd.release["start"])
            enhancements.append(enhancement)
            backgrounds.append(background)
            directories.append(fd._dir)
        except Exception as e:
            print(e)
            continue
    xco2 = np.array(enhancements) + np.array(backgrounds)
    dataframe = pd.DataFrame(data=dict(directory=directories, time=times, enhancement=enhancements, background=backgrounds, xco2=xco2))
    dataframe.to_pickle(os.path.join(args.dir, "predictions.pkl"))