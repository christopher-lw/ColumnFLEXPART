### file to prepare RELEASE file of a column receptor ###
import argparse
from pathlib import Path
import yaml
import os
import shutil
import numpy as np
from columnflexpart.utils import yyyymmdd_to_datetime64 
from columnflexpart.classes	 import setup_column

def get_configs(config_path: str) -> tuple[str, str]:
    """Gets paths of config files

    Args:
        config_path (str): Path to config file of dir of config filesystem

    Returns:
        tuple[str, str]: directory and file names of configs
    """    
    if os.path.isdir(config_path):
        dir_path = config_path
        file_names = os.listdir(dir_path)
    else:
        dir_path, file_names = config_path.rsplit("/", 1)
        file_names = [file_names]
    return dir_path, file_names

def load_header(species: int=41) -> list[str]:
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

def load_dummy() -> list[str]:
    """Load dummy for RELEASE file.

    Returns:
        list: List of lines in dummy RELEASE file
    """    
    dummy_path = Path(__file__).parent / "dummies/RELEASES_dummy.txt"
    with open(dummy_path) as f:
        dummy = f.read().splitlines()
    return dummy

def write_to_dummy(date1: str, time1: str, date2: str, time2: str, lon1: float, lon2: float, 
    lat1: float, lat2: float, z1: float, z2: float, zkind: int, mass: float, parts: int, comment: str) -> list[str]:
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

def setup_times(config: dict) -> list[list[str]]:
    """Returns times either read from config or file given in config

    Args:
        config (dict): Loaded config as dict

    Returns:
        list[list[str]]: List of sets of [start_date, start_time, end_date, end_time]
    """    
    if not config["times_from_file"]:
        times = config["times"]

    else:
        assert "times_file_path" in config.keys(), "First set times_file_path in config or set times_from_file to 'true'"
        with open(config["times_file_path"], "r") as f:
            lines = f.read().splitlines()
        times = []
        for line in lines:
            times.append(line.split(","))
    return times

def setup_coords(config: dict) -> list[list[str]]:
    """Returns coordinates of release either read from config or file set in config.

    Args:
        config (dict): Loaded config as dictionary

    Returns:
        list[list[str]]: List of sets of [left right lower upper]
    """    
    if not config["coords_from_file"]:
        coords = config["coords"]

    else:
        assert "coords_file_path" in config.keys(), "First set coords_file_path in config or set coords_from_file to 'true'"
        with open(config["coords_file_path"], "r") as f:
            lines = f.read().splitlines()
        coords = []
        for line in lines:
            coords.append(line.split(","))
    return coords

def get_save_condition(counter: int, max_counter: int, multiple_days_per_file: bool, date: str, times: str, loop_ind: int) -> bool:
    """Return bool to state whether to save or not based on the current counter, the following date and the arguments given.

    Args:
        counter (int): Counter of releases since last save
        max_counter (int): Value of counter at which result releases should be saved
        multiple_days_per_file (bool): Flag if multiple days per releases file should be allowed
        date (str): date of current release
        times (str): list of times 
        loop_ind (int): current index in times to check if date of last value in times == date

    Returns:
        bool: Bool to state whether to save or not
    """    
    
    if len(times) - 1 == loop_ind:
        ret = True 
    else:
        next_date = times[loop_ind + 1][0]
        ret = (counter == max_counter) or (not multiple_days_per_file and next_date != date)
    return ret

def get_save_name(coords: np.ndarray, name: str, index: int) -> str:
    """Constructs save name based on input

    Args:
        coords (np.ndarray): To check if multiple files are in the output
        name (str): Base name of input
        index (int): index to add if multiple files are output

    Returns:
        str: Name for output directory
    """    
    save_name = name
    if len(coords) > 1:
        save_name += f"_{index}"
    return save_name

def save_release(dir_name: str, file_name: str, release_data: str, config: str=None):
    """Saves releases data to RELEASES file with name file_name in directory dir_name

    Args:
        dir_name (str): Directory to save to 
        file_name (str): File name of RELEASES file 
        release_data (str): Data to save
        config (str, optional): Path to config file if config should be saved together with RELEASES file. Defaults to None.
    """    
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, file_name), 'w') as f:
        for item in release_data:
            f.write("%s\n" % item)
        f.write("%s\n" % "")
    if config is not None:
        shutil.copyfile(config, os.path.join(dir_name, "config.yaml"))

def get_parser():
    parser = argparse.ArgumentParser(description="Construction tool for FLEXPART RELAESE file of column receptor")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("out_dir", type=str, default="output", help="path of directory for output files")
    parser.add_argument("--out_name", type=str, default="RELEASES", help="name for output file(default is 'RELEASES', ignored if config is dir)")
    parser.add_argument("--stations_per_file", type=int, default=1, help="Number of column setups per RELEASES file. (default is 1)")
    parser.add_argument("--mdpf", action="store_true", help="(Multiple Days Per File) Flag to allow different days in one RELEASES file")
    return parser
    
################################### MAIN ######################################

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    config_dir, config_files = get_configs(args.config)
    
    # start of file construction
    for file in config_files:
        file_path = os.path.join(config_dir, file)
        if len(config_files) > 1:
            # get output name without ".yaml"
            args.out_name = file.split(".")[0]
        
        with open(file_path) as f:
            config = dict(yaml.full_load(f))
        # GET PARAMETERS 
        zkind = config["zkind"]
        mass = config["mass"]
        height_levels, part_nums = setup_column(config)
        coords = setup_coords(config)
        times = setup_times(config)

        if len(coords) == 1:
            coords = coords*len(times)
        elif len(times) == 1:
            times = times*len(coords)

        assert len(coords) == len(times), f'Coords and times have the same number of elements. Your input: len(coords)={len(coords)}, len(times)={len(times)}'

        # list of strings to hold information for RELEASES file
        releases = load_header(config["species"])
        
        if config["discrete"]:
            height_levels = height_levels[1:]
        
        station_counter = 0
        save_counter = 0
        # SAVE PARAMETERS INTO RELEASES FILES
        for i, ((lon1, lon2, lat1, lat2), (date1, time1, date2, time2)) in enumerate(zip(coords, times)):
            date = date1
            station_counter += 1
            for j, parts in enumerate(part_nums):
                comment = f'"coords:{(lon1, lon2, lat1, lat2)}, height:{height_levels[j]}, time:{(date1, time1, date2, time2)}"'
                z1 = height_levels[j]
                z2 = height_levels[j if config["discrete"] else j+1]
                # insert paratmeters into RELEASES file 
                releases.extend(write_to_dummy(date1, time1, date2, time2, lon1, lon2,
                    lat1, lat2, z1, z2, zkind, mass, parts, comment))                

            save_condition = get_save_condition(station_counter, args.stations_per_file, args.mdpf, date, times, i)
            if save_condition:
                save_name = get_save_name(coords, args.out_name, save_counter)
                save_release(args.out_dir, save_name, releases)
                save_counter += 1
                station_counter = 0
                releases = load_header(config["species"])

        print(f"Save destination: {os.path.join(args.out_dir, args.out_name)}(_#)")
        print(f"Number of stations: {len(coords)}")
        print(f"Total number of particles: {np.sum(part_nums)*len(coords)}")
