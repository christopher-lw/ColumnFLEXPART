import os
import argparse
from filelock import FileLock
import subprocess
import shutil
from matplotlib.style import available
import yaml
import numpy as np
from utils import yyyymmdd_to_datetime64, hhmmss_to_timedelta64, datetime64_to_yyyymmdd_and_hhmmss

def get_paths(config: dict, releases_index: int)->list[str]:
    flex_path = config["flex_path"]
    releases_path = config["releases_path"]
    releases_files = os.listdir(releases_path)
    releases_files.sort()
    releases_file = os.path.join(releases_path, releases_files[releases_index])
    options_path = config["options_path"]
    output_path = config["output_path"]
    pathnames_file = os.path.join(flex_path, "pathnames")
    input_paths = config["input_paths"]
    if not isinstance(input_paths, list): input_paths = [input_paths]
    available_files = config["available_paths"]
    if not isinstance(available_files, list): available_files = [available_files]
    exe_path = config["exe_path"]
    return flex_path, releases_file, options_path, output_path, pathnames_file, input_paths, available_files, exe_path

def check_paths(paths: list[str]):
    for path in paths:
        if not isinstance(path, list):
            path = [path]
        for p in path:
            if not os.path.exists(p): raise FileNotFoundError(p)

def prepare_output_dir(output_path: str, releases_file: str):
    releases_name = os.path.basename(releases_file)
    output_path = os.path.join(output_path, releases_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def set_start(start: np.datetime64, step: int, shift: int) -> np.datetime64:
    assert np.timedelta64(1, "D") % np.timedelta64(step, "h") == 0, f"24 hours have to be devisable by value of step. (Your value: {step})"
    date = start.astype("datetime64[D]")
    shifts_late = np.arange(np.timedelta64(shift, "h"), np.timedelta64(shift+24, "h"), np.timedelta64(step, 'h'), dtype="timedelta64[h]")
    shifts_early = np.arange(np.timedelta64(shift, "h"), np.timedelta64(shift-24, "h"), - np.timedelta64(step, 'h'), dtype="timedelta64[h]")[::-1]
    shifts = np.concatenate([shifts_early, shifts_late])
    start_vals = date + shifts
    ret = start_vals[start_vals > start][0].astype("datetime64[s]")
    return ret

def get_start_stop(releases_file: str, step: int, shift: int, sim_lenght: int) -> tuple[np.datetime64, np.datetime64]:
    with open(releases_file) as f:
        lines = f.readlines()
    start_dates = [yyyymmdd_to_datetime64(line.split("=")[1].split(",")[0].replace(" ", ""))
        for line in lines if "IDATE2" in line]
    start_times = [hhmmss_to_timedelta64(line.split("=")[1].split(",")[0].replace(" ", "")) 
        for line in lines if "ITIME2" in line]
    stop_dates = [yyyymmdd_to_datetime64(line.split("=")[1].split(",")[0].replace(" ", ""))
        for line in lines if "IDATE1" in line]
    stop_times = [hhmmss_to_timedelta64(line.split("=")[1].split(",")[0].replace(" ", "")) 
        for line in lines if "ITIME1" in line]
    
    starts = np.array(start_dates) + np.array(start_times)
    stops = np.array(stop_dates) + np.array(stop_times)
    # max/min/- instead of min/max/+ since runs go backwards
    start = max(starts)
    stop = min(stops)
    start = set_start(start, step, shift)
    stop = stop - np.timedelta64(sim_lenght, "D")
    return start, stop

def prepare_command(options_path: str, start: np.datetime64, stop: np.datetime64):
    start_date, start_time = datetime64_to_yyyymmdd_and_hhmmss(start)
    stop_date, stop_time = datetime64_to_yyyymmdd_and_hhmmss(stop)
    with open(os.path.join(options_path, "COMMAND"), "r") as f:
        com_file = ""
        for line in f:
            addition = line
            if "IEDATE" in line:
                addition = f" IEDATE={start_date},\n"
            if "IETIME" in line:
                addition = f" IETIME={start_time},\n"
            if "IBDATE" in line:
                addition = f" IBDATE={stop_date},\n"
            if "IBTIME" in line:
                addition = f" IBTIME={stop_time},\n"
            com_file = com_file + addition
    with open(os.path.join(options_path, "COMMAND"), "w") as f:
        f.write(com_file)

def prepare_pathnames(
    pathnames_file: str, options_path: str, output_path: str, 
    input_paths: list[str], available_files: list[str]
    ):
    content = ""
    content += f"{options_path}\n"
    content += f"{output_path}\n"
    for input, avail in zip(input_paths, available_files):
        content += f"{input}\n"
        content += f"{avail}\n"
    with open(pathnames_file, "w") as f:
        f.write(content)

def check_for_run_start(process, log_file: str):
    with open(log_file, 'w') as log:
        for line in process.stdout:
            log.write(line)
            if "Simulated" in line:
                break

def continue_log(process, log_file):
    with open(log_file, 'a') as log:
        for line in process.stdout:
            log.write(line)
            log.flush()

################################### MAIN ######################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script to start FLEXPART run secure from interference with other runs.")
    parser.add_argument("config", type=str, help="Path to config file containing information for the runs")
    parser.add_argument("releases_index", type=int, help="Index of Release to start")
    args = parser.parse_args()

    with open(args.config) as f:
        config = dict(yaml.full_load(f))

    paths = get_paths(config, args.releases_index)
    check_paths(paths)
    flex_path, releases_file, options_path, output_path, pathnames_file, input_paths, available_files, exe_path = paths

    lock = FileLock(f"{pathnames_file}.lock")
    with lock:
        output_path = prepare_output_dir(output_path, releases_file)
        log_file = os.path.join(output_path, "log")
        shutil.copyfile(releases_file, os.path.join(options_path, "RELEASES"))
        # start means from point of the simulation (start > stop)
        start, stop = get_start_stop(releases_file, config["start_step"], config["start_shift"], config["sim_length"])
        prepare_command(options_path, start, stop)
        prepare_pathnames(pathnames_file, options_path, output_path, input_paths, available_files)

        os.chdir(flex_path)    
        process = subprocess.Popen([exe_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
        check_for_run_start(process, log_file)

    continue_log(process, log_file)
    process.wait()
    # unlock pathnames