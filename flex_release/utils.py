### Utilities for column receptor setup ###

import numpy as np

def pressure_factor(
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

def load_header(species=41):
    """Load header for RELEASE file
    Args:
        species (int): index of species to use in run
    Returns:
        list: List of lines in header for RELEASE file
    """    
    header_path = "dummies/HEADER_dummy.txt"
    with open(header_path) as f:
        header = f.read().splitlines()
    header[-2] = header[-2].replace("#", f"{species}")
    return header

def load_dummy():
    """Load dummy for RELEASE file.

    Returns:
        list: List of lines in dummy RELEASE file
    """    
    dummy_path = "dummies/RELEASES_dummy.txt"
    with open(dummy_path) as f:
        dummy = f.read().splitlines()
    return dummy

def write_to_dummy(date1, time1, date2, time2, lon1, lon2, 
    lat1, lat2, z1, z2, zkind, mass, parts, comment):
    """Write given params of into RELEASE fiel based on a dummy.

    Args:
        date1 (str): Start date (YYYYMMDD)
        time1 (str): Start time (HHMMSS)
        date2 (str): End date (YYYYMMDD)
        time2 (str): End time (HHMMSS)
        lon1 (float): min longitude
        lon2 (float): max longitude
        lat1 (float): min latitude
        lat2 (float): max latitude
        z1 (float): min height
        z2 (float): max height
        zkind (int): interpretation of height
        mass (float): emitted mass
        parts (int): number of particles
        comment (str): comment for release

    Returns:
        list: dummy with filled in values
    """      
    args = locals()
    dummy = load_dummy()
    
    for i, (_, value) in enumerate(args.items()):
        dummy[i+1] = dummy[i+1].replace("#", f"{value}")
    
    return dummy

def setup_column(config):

    mode = config["column_mode"]
    assert mode in ["wu", "pressure_full", "pressure_wu"], f"Only possible column_modes: wu, pressure_full, pressure_wu. You entered: {mode}"
    
    if mode == "wu":
        assert set(["regions", "dh", "dn"]).issubset(config.keys()), "For column mode wu give argumnets regions, dh, dn."
        
        regions = config["regions"]
        dh = config["dh"]
        dn = config["dn"]

        if not isinstance(dh, list): dh = [dh for i in range(len(regions)-1)] 
        if not isinstance(dn, list): dn = [dn for i in range(len(regions)-1)] 

        assert len(regions) == len(dh)+1, f"Incompatible shapes of regions and dh. Required: dh == region-1 not len(regions)={len(regions)} and len(dh)={len(dh)} (give single value as float not list)"
        assert len(regions) == len(dn)+1, f"Incompatible shapes of regions and dn. Required: dn == region-1 not len(regions)={len(regions)} and len(dn)={len(dn)} (give single value as float not list)"
        
        height = regions[0]*1e3
        height_levels = [height]
        part_nums = []
        

        for i, region in enumerate(regions[1:]):
            while height < region*1e3:
                height += dh[i]
                height_levels.append(height)
                part_nums.append(dn[i])
                 
        height_levels = np.array(height_levels)
        part_nums = np.array(part_nums)
    
    elif mode == "pressure_full":
        assert set(["regions", "lowest_region_n", "dh"]).issubset(config.keys()), "For column mode wu give argumnets regions, lowest_region_n, dh."
        
        regions = config["regions"]
        dh = config["dh"]
        lowest_region_n = config["lowest_region_n"]

        if not isinstance(dh, list): dh = [dh for i in range(len(regions)-1)]
        assert len(regions) == len(dh)+1, f"Incompatible shapes of regions and dh. Required: dh == region-1 not len(regions)={len(regions)} and len(dh)={len(dh)} (give single value as float not list)"

        height = regions[0]*1e3
        height_levels = [height]
        for i, region in enumerate(regions[1:]):
            while height < region*1e3:
                height += dh[i]
                height_levels.append(height)
        
        height_levels = np.array(height_levels)
        
        diff_heights = height_levels[1:] - height_levels[:-1]  
        mid_heights = (height_levels[:-1] + height_levels[1:])/2
        factors = pressure_factor(mid_heights)
        norm = factors[mid_heights <= regions[1]*1e3].sum()
        factors = factors/norm * diff_heights/dh[0]
        part_nums = lowest_region_n * factors
        part_nums = part_nums.round(0).astype(int)

    elif mode == "pressure_wu":
        assert set(["regions", "region_n", "dh"]).issubset(config.keys()), "For column mode wu give argumnets regions, region_n, dh."

        regions = config["regions"]
        dh = config["dh"]
        region_n = config["region_n"]

        if not isinstance(dh, list): dh = [dh for i in range(len(regions)-1)]
        assert len(regions) == len(dh)+1, f"Incompatible shapes of regions and dh. Required: len(dh) == len(region)-1 not len(regions)={len(regions)} and len(dh)={len(dh)} (give single value as float not list)"
        assert len(regions) == len(region_n)+1, f"Incompatible shapes of regions and region_n. Required: len(region_n) == len(region-1 not len(regions)={len(regions)} and len(dh)={len(dh)} (give single value as float not list)"

        height = regions[0]*1e3
        height_levels = [height]
        part_nums = []
        for i, region in enumerate(regions[1:]):
            new_heights = []
            while height < region*1e3:
                height += dh[i]
                new_heights.append(height)
            height_levels.extend(new_heights)
            
            factors = pressure_factor(np.array(new_heights))
            factors = factors/np.sum(factors)
            part_nums.extend(list(np.array(region_n[i]) * factors))
        
        height_levels = np.array(height_levels)
        part_nums = np.array(part_nums).round(0).astype(int)

    return height_levels, part_nums

def setup_times(config):
    if not config["times_from_file"]:
        times = config["times"]

    else:
        assert False, "times_from_file not yet implemented"
        assert "times_file_path" in config.keys(), "First set times_file_path in config or set times_from_file to 'true'"
        with open(config["times_file_path"], "r") as f:
            lines = f.read().splitlines()

        times = []
        for line in lines:
            times.append([line.split(",")])
    return times

def setup_coords(config):
    if not config["coords_from_file"]:
        coords = config["coords"]

    else:
        assert False, "coords_from_file not yet implemented"
        assert "coords_file_path" in config.keys(), "First set coords_file_path in config or set coords_from_file to 'true'"
        with open(config["coords_file_path"], "r") as f:
            lines = f.read().splitlines()

        coords = []
        for line in lines:
            coords.append(line.split(","))
    return coords
    