### file to prepare RELEASE file of a column receptor ###
import argparse
import yaml
import os
import shutil
import numpy as np
from utils import *

def save_release(dir_name, file_name, release_data, config=None):
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, file_name), 'w') as f:
        for item in release_data:
            f.write("%s\n" % item)
        f.write("%s\n" % "")
    if config is not None:
        shutil.copyfile(config, os.path.join(dir_name, "config.yaml"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction tool for FLEXPART RELAESE file of column receptor")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--out_dir", type=str, default="output", help="path of directory for output files (default is 'output')")
    parser.add_argument("--out_name", type=str, default="RELEASES", help="name for output file(default is 'RELEASES')")
    parser.add_argument("--split", type=str, help="If and how to split the output release files. 'station' to split according the station, else int n to split into n parts. (Number of releases should be devisable by n)")
    args = parser.parse_args()
    
    if os.path.isdir(args.config):
        dir_path = args.config
        file_names = os.listdir(dir_path)
    else:
        dir_path, file_names = args.config.rsplit("/", 1)
        file_names = [file_names]
    
    for file in file_names:
        file_path = os.path.join(dir_path, file)
        if len(file_names) > 1:
            args.out_name = file.split(".")[0]
        
        with open(file_path) as f:
            config = dict(yaml.full_load(f))

        zkind = config["zkind"]
        mass = config["mass"]

        height_levels, part_nums = setup_column(config)
        coords = setup_coords(config)
        times = setup_times(config)

        assert len(coords) == len(times), f'Coords and times have the same number of elements. Your input: len(coords)={len(coords)}, len(times)={len(times)}'

        RELEASES = load_header(config["species"])

        if config["discrete"]:
            height_levels = height_levels[1:]

        release_counter = 0
        save_counter = 0
        release_number = len(coords)*len(part_nums)

        split = False
        split_by = None
        if args.split is not None:
            if args.split == "station":
                split = True
                split_by = "station"
            else:
                split = True
                split_by = int(args.split)
                assert release_number % split_by == 0, f"If int is given for split argument it has to be denominator of number of releases ({release_number})"

        for i, ((lon1, lon2, lat1, lat2), (date1, time1, date2, time2)) in enumerate(zip(coords, times)):
            for j, parts in enumerate(part_nums):
                comment = f'"coords:{(lon1, lon2, lat1, lat2)}, height:{height_levels[j]}, time:{(date1, time1, date2, time2)}"'
                z1 = height_levels[j]
                z2 = height_levels[j if config["discrete"] else j+1]
                RELEASES.extend(write_to_dummy(date1, time1, date2, time2, lon1, lon2,
                    lat1, lat2, z1, z2, zkind, mass, parts, comment))
                release_counter += 1
                if split and isinstance(split_by, int) and release_number and release_counter%(release_number//split_by)==0:
                    save_release(args.out_dir, f"{args.out_name}_{save_counter}", RELEASES, args.config)
                    save_counter +=1
                    RELEASES = load_header(config["species"])
            if split and split_by == "station":
                save_release(args.out_dir, f"{args.out_name}_{save_counter}", RELEASES)
                save_counter +=1
                RELEASES = load_header(config["species"])
        print(f"Save destination: {os.path.join(args.out_dir, args.out_name)}")
        print(f"Total number of particles: {np.sum(part_nums)*len(coords)}")

        save_release(args.out_dir, args.out_name, RELEASES, file_path) if not split else None
