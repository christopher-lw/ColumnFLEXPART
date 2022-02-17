### file to prepare RELEASE file of a column receptor ###

if __name__ == "__main__":
    import argparse
    import yaml
    import os
    import shutil
    import numpy as np
    from utils import *

    parser = argparse.ArgumentParser(description="Construction tool for FLEXPART RELAESE file of column receptor")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--out_path", type=str, default="output", help="path of directory for output files")
    parser.add_argument("--out_name", type=str, default="RELEASES", help="name for output file")
    args = parser.parse_args()
    
    with open(args.config) as f:
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

    for i, ((lon1, lon2, lat1, lat2), (date1, time1, date2, time2)) in enumerate(zip(coords, times)):
        for j, parts in enumerate(part_nums):
            comment = f'"coords:{(lon1, lon2, lat1, lat2)}, height:{height_levels[j]}, time:{(date1, time1, date2, time2)}"'
            z1 = height_levels[j]
            z2 = height_levels[j if config["discrete"] else j+1]
            RELEASES.extend(write_to_dummy(date1, time1, date2, time2, lon1, lon2, 
                lat1, lat2, z1, z2, zkind, mass, parts, comment))
    
    print(f"Total number of particles: {np.sum(part_nums)*len(coords)}")
    
    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, args.out_name), 'w') as f:
        for item in RELEASES:
            f.write("%s\n" % item)
        f.write("%s\n" % "")
    shutil.copyfile(args.config, os.path.join(args.out_path, "config.yaml"))