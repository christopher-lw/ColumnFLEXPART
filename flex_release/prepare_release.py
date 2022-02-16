### file to prepare RELEASE file of a column receptor ###
if __name__ == "__main__":
    import argparse
    import yaml
    import os
    from utils import *

    parser = argparse.ArgumentParser(description="Construction tool for FLEXPART RELAESE file of column receptor")
    parser.add_argument("config", type=str,
        help="path to config file")
    parser.add_argument("--out_path", type=str, default="output",
        help="path of directory for output files")
    args = parser.parse_args()
    
    #print(args.config)
    with open(args.config) as f:
        config = yaml.full_load(f)
    
    coords = [config["coords"]] if type(config["coords"][0]) != list else config["coords"]
    regions = [0, config["regions"]] if type(config["regions"]) != list else config["regions"]
    dh = [config["dh"] for i in range(len(regions)-1)] if type(config["dh"]) != list else config["dh"]
    dn = [config["dn"] for i in range(len(regions)-1)] if type(config["dn"]) != list else config["dn"]

    assert len(regions) > len(dh), f"Incompatible shapes of regions and dh. Required: dh < region not len(regions)={len(regions)} and len(dh)={len(dh)}"
    assert len(regions) > len(dn), f"Incompatible shapes of regions and dn. Required: dn < region not len(regions)={len(regions)} and len(dn)={len(dn)}"

    height_levels = []
    part_nums = []
    height = 0

    for i, region in enumerate(regions[1:]):
        while height < region*1e3:
            height_levels.append(height)
            part_nums.append(dn[i])
            height += dh[i]

    RELEASES = load_header()
    for i, (lon1, lat1, lon2, lat2) in enumerate(config["coords"]):
        for j, parts in enumerate(part_nums[:-1]):
            #############CHANGE
            date1 = "20010909"
            time1 = "000000"
            date2 = "20010909"
            time2 = "000000"
            zkind = 1
            mass = 1
            comment = "lalal"
            ############CHANGE
            z1 = height_levels[j]
            z2 = height_levels[j+1]
            RELEASES.extend(write_to_dummy(date1, time1, date2, time2, lon1, lon2, 
                lat1, lat2, z1, z2, zkind, mass, parts, comment))
    
    with open(os.path.join(args.out_path,'RELEASES'), 'w') as f:
        for item in RELEASES:
            f.write("%s\n" % item)