import argparse
import os
from xml.sax import parse
import pandas as pd
import numpy as np

from open_data_utils import FlexDataset2
from tqdm.auto import tqdm

def find_nc_files(dir, file_list=[]):
    found_file = False
    for file in os.listdir(dir):
        if "grid_time" in file:
            file_list.append(dir)
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
    parser.add_argument("sounding_file", type=str, help="File start for sounding file")
    parser.add_argument("--boundary", type=float, nargs="+", default=None, help="Boundary to cut out for investigation [left, right, bottom, top]")
    parser.add_argument("--read_only", action="store_true", default=False, help="Flag to only try to read saved values")
    parser.add_argument("--out_name", type=str, default="predictions.pkl", help="Name for output pickle file (defaults to 'predictions.pkl')")

    args = parser.parse_args()
    files = find_nc_files(args.dir)

    print(args.dir)

    times = []
    enhancements = []
    enhancements_inter = []
    enhancements_diff = []
    backgrounds = []
    backgrounds_inter = []
    xco2s = []
    xco2s_inter = []
    xco2s_acos = []
    xco2s_acos = []
    directories = []
    acos_files  = []
    acos_sounding_ids = []
    acos_uncertainties = []

    for dir in tqdm(files):
        if not "trajectories.pkl" in os.listdir(dir):
            print(f"Missing trajectories.pkl in {dir}")
            continue     
        try: 
            fd = FlexDataset2(dir, ct_dir=args.conc_dir, ct_name_dummy=args.conc_name, chunks=dict(time=20, pointspec=4))
            fd.load_sounding(args.sounding_file)
            enhancement_inter = fd.enhancement(ct_file=args.flux_file, boundary=args.boundary, allow_read=args.read_only, interpolate=True)
            enhancement = fd.enhancement(ct_file=args.flux_file, boundary=args.boundary, allow_read=args.read_only, interpolate=False)

            tr = fd.trajectories
            
            try: 
                tr.load_endpoints()
                assert tr.endpoints.attrs["boundary"] == args.boundary
            except Exception as e:
                print(e)
                tr.endpoints = None
                tr.ct_endpoints(boundary=args.boundary)
                tr.co2_from_endpoints(boundary=args.boundary)
                tr.save_endpoints()
            

            background_inter = fd.background(allow_read=args.read_only, boundary=args.boundary, interpolate=True)
            background = fd.background(allow_read=args.read_only, boundary=args.boundary, interpolate=False)

            xco2_inter = fd.total(ct_file=args.flux_file, allow_read=args.read_only, boundary=args.boundary, chunks=dict(time=20, pointspec=4), interpolate=True)
            xco2 = fd.total(ct_file=args.flux_file, allow_read=args.read_only, boundary=args.boundary, chunks=dict(time=20, pointspec=4), interpolate=False)
            xco2_acos = float(fd.sounding.xco2)

            _ = fd.footprint.total()
            _ = fd.footprint.total(interpolate=False)

            

            times.append(fd.release["start"])
            enhancements.append(enhancement)
            enhancements_inter.append(enhancement_inter)
            backgrounds.append(background)
            backgrounds_inter.append(background_inter)
            xco2s.append(xco2)
            xco2s_inter.append(xco2_inter)
            xco2s_acos.append(xco2_acos)
            enhancements_diff.append(xco2_inter-background_inter)
            directories.append(fd._dir)
            acos_files.append(fd.sounding_path)
            acos_sounding_ids.append(fd.sounding_id)
            acos_uncertainties.append(float(fd.sounding.xco2_uncertainty))
        except Exception as e:
            print(e)
            continue
    dataframe = pd.DataFrame(
        data=dict(
            directory=directories, 
            time=times, 
            enhancement=enhancements, 
            enhancement_inter=enhancements_inter, 
            enhancement_diff =enhancements_diff,
            background=backgrounds, 
            background_inter = backgrounds_inter, 
            xco2=xco2s,
            xco2_inter=xco2s_inter,
            xco2_acos=xco2s_acos,
            acos_sounding_id=acos_sounding_ids,
            acos_file=acos_files,
            acos_uncertainty=acos_uncertainties
            ))
    dataframe.to_pickle(os.path.join(args.dir, args.out_name))