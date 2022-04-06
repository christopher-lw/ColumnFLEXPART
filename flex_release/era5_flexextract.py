import os 
import argparse 
import errno
from multiflex import get_queue, get_run_states
import numpy as np
import time
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to Download consecutive days of ERA5 data in seperate jobs.")

    parser.add_argument("start", type=str, help="Start date of downloads YYYYMMDD")
    parser.add_argument("end", type=str, help="End date of downloads YYYYMMDD")
    parser.add_argument("extr_path", type=str, help="path to flexextract home directory")
    parser.add_argument("run_path", type=str, help="path to run file (if not absolute path it starts from extr_path/Run)")
    parser.add_argument("control_path", type=str, help="path to control file used in submit (if notabsolute path it starts from extr_path/Run/Control)")
    parser.add_argument("submit_path", type=str, help="path to slurm script for submission of jobs")
    parser.add_argument("--output_path", type=str, default="Run/Workspace", help="Directory for output (if not absolute path it starts from extr_path).")

    args = parser.parse_args()

    if args.run_path[0] != "/":
        args.run_path = os.path.join(args.extr_path, "Run", args.run_path)
    if args.output_path[0] != "/":
        args.output_path = os.path.join(args.extr_path, args.output_path)
    if args.control_path[0] != "/":
        args.control_path = os.path.join(args.extr_path, "Run/Control", args.control_path)

    run_dummy_path = "dummies/run_local.sh"

    #check if all paths exist
    for arg in vars(args): 
        if not "path" in arg:
            continue
        path = getattr(args, arg)
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        print(f"\n{arg} = {path}")

    inp = input("\nAre the run and submit file properly prepared? ([y]/n) ") or "y"
    assert inp == "y", "Insert y to continue."

    #set list of days to download data for
    start_date = np.datetime64(f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]}")
    end_date = np.datetime64(f"{args.end[:4]}-{args.end[4:6]}-{args.end[6:]}")

    dates = np.arange(start_date, end_date + np.timedelta64(1, "D"), dtype="datetime64[D]")
    
    print(f"\nERA 5 data for following dates will be downloaded: \n {dates}")
    inp = input("\nContinue? ([y]/n) ") or "y"
    assert inp == "y", "Insert y to continue."
    
    #start jobs
    for date in dates:
        shutil.copyfile(run_dummy_path, args.run_path)
        date = str(date).replace("-", "")
        run_file = ""
        with open(args.run_path) as f:
            #read each line and change dummies to values
            for line in f:
                addition = line
                addition = addition.replace("_startdate_", date)
                addition = addition.replace("_outputpath_", args.output_path)
                addition = addition.replace("_controlpath_", args.control_path)
                addition = addition.replace("_extrpath_", args.extr_path)
                run_file = run_file + addition
        #write manipulated file
        with open(args.run_path, "w") as f:
            f.write(run_file)
        #submit job
        os.system(f"sbatch {args.submit_path}")
        #wait for job to start
        states = get_run_states()
        while states.count("R") != len(states):
            time.sleep(2)
            states = get_run_states()
            r_num = states.count("R")
            print(f"{r_num}/{len(states)} tasks running")
        #buffer
        time.sleep(5)