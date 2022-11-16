# python script to start multiple flexpart runs with different releases
import os
import shutil
import argparse
import errno
import time
import subprocess
import numpy as np

#test
def get_queue():
    username = os.path.expanduser('~').split("/")[-1]
    command = ["squeue", "-u", username]
    queue = subprocess.Popen(command, stdout=subprocess.PIPE)
    queue = str(queue.stdout.read())[2:-1].replace("\\n","\n").splitlines()
    return queue

def get_run_states():
    queue = get_queue()[1:]
    states = []
    for q in queue:
        states.append(q.split()[-4])
    return states

def set_start(start, shift=0, step=None):
    if step is not None:
        assert np.timedelta64(1, "D") % np.timedelta64(step, "h") == 0, f"24 hours have to be devisable by value of step. (Your value: {step})"
        date = start.astype("datetime64[D]")
        shifts_late = np.arange(np.timedelta64(shift, "h"), np.timedelta64(shift+24, "h"), np.timedelta64(step, 'h'), dtype="timedelta64[h]")
        shifts_early = np.arange(np.timedelta64(shift, "h"), np.timedelta64(shift-24, "h"), - np.timedelta64(step, 'h'), dtype="timedelta64[h]")[::-1]
        shifts = np.concatenate([shifts_early, shifts_late])
        start_vals = date + shifts
        ret = start_vals[start_vals > start][0].astype("datetime64[s]")
    else: 
        ret = start + np.timedelta64(1, "h")
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to start multiple FLEXPART runs with different RELAESES files")
    parser.add_argument("flex_path", type=str, help="path to home directory of FLEXPART to be used")
    parser.add_argument("options_path", type=str, help="path to options directory for the runs (if not absolute path it starts from flex_path)")
    parser.add_argument("output_path", type=str, help="path for output directory (if not absolute path it starts from flex_path)")
    parser.add_argument("release_path", type=str, help="path of directory with RELEASES files (if not absolute path it starts from options_path)")
    parser.add_argument("submit_path", type=str, help="path to slurm script for submission of runs")
    parser.add_argument("--set_sim_length", type=int , default=0, help="sets simulation lenghts in days for each run automatically sets start one hour before release")
    parser.add_argument("--start_shift", type=int, default=0, help="if set_sim_lenght is set but only certain start values are admissable (e.g. every 3 hours starting at 1 o'clock set to 1 and start_step to 3)")
    parser.add_argument("--start_step", type=int, default=None, help="steps from start in which are admissable start values for simulation")
    parser.add_argument("--max_jobs", type=int, default=20, help="Determines maximum of slurm jobs be active at once.") 
    args = parser.parse_args()

    #append paths to be abolute
    if args.options_path[0] != "/":
        args.options_path = os.path.join(args.flex_path, args.options_path)

    if args.release_path[0] != "/":
        args.release_path = os.path.join(args.options_path, args.release_path)

    if args.output_path[0] != "/":
        args.output_path = os.path.join(args.flex_path, args.output_path)

    #check if all paths exist
    for arg in vars(args): 
        path = getattr(args, arg)
        if "path" in arg:
            if not os.path.exists(path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
            print(f"{arg} = {path}")

    inp = input("\nIs the submit file properly prepared? ([y]/n) ") or "y"
    assert inp == "y", "Insert y to continue."

    #get all release files to start jobs with
    rel_names = os.listdir(args.release_path)
    print(f"Found following files in release_path: {rel_names} \n")

    #possibility to save old RELEASE file
    if os.path.exists(os.path.join(args.options_path, "RELEASES")):
        inp = input("Save old RELEASE file? (y/[n]) ") or "n"
        if inp == "y":
            inp = input("New name of old RELEASES file: ")
            shutil.copyfile(os.path.join(args.options_path, "RELEASES"), os.path.join(args.options_path, inp))
            print(f"Saved to {os.path.join(args.options_path, inp)}")

    inp = input("Start runs? ([y]/n) ") or "y"
    assert inp == "y", "Insert y to continue."

    #start jobs
    for rel in rel_names:
        #copy RELEASE file to options directory
        shutil.copyfile(os.path.join(args.release_path, rel), os.path.join(args.options_path, "RELEASES"))
        #set current output dir
        new_output_path = os.path.join(args.output_path, rel)
        os.makedirs(new_output_path, exist_ok=True)
        if args.flex_path in new_output_path:
            new_output_path = new_output_path.replace(args.flex_path, "./")
        #set output path in pathnames file
        paths = ""
        with open(os.path.join(args.flex_path, "pathnames")) as f:
            for i, line in enumerate(f):
                addition = line
                if i == 1:
                    addition = f"{new_output_path}\n"
                paths = paths + addition
        #write in pathnames file
        with open(os.path.join(args.flex_path, "pathnames"), "w") as f:
            f.write(paths)
        #adjust COMMAND file to according to start step and shift and end args.set_sim_length days after
        if args.set_sim_length != 0:
            with open(os.path.join(args.options_path, "RELEASES"), "r") as f:
                for i, line in enumerate(f):
                    if "IDATE1" in line:
                        date_str = line.split("=")[1].split(",")[0].replace(" ", "")
                        start_date = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
                    if "ITIME1" in line:
                        time_str = line.split("=")[1].split(",")[0].replace(" ", "")
                        start_time = np.timedelta64(time_str[:2], "h") + np.timedelta64(time_str[2:4], "m") + np.timedelta64(time_str[4:], "s")
                        break
            start = start_date + start_time
            end = start - np.timedelta64(args.set_sim_length, "D")
            start = set_start(start, args.start_shift, args.start_step)
            start_date, start_time = str(start).split("T")
            start_date = start_date.replace("-", "")
            start_time = start_time.replace(":", "")
            end_date, end_time = str(end).split("T")
            end_date = end_date.replace("-", "")
            end_time = end_time.replace(":", "")
            com_file = ""
            with open(os.path.join(args.options_path, "COMMAND"), "r") as f:
                for line in f:
                    addition = line
                    if "IEDATE" in line:
                        addition = f" IEDATE={start_date},\n"
                    if "IETIME" in line:
                        addition = f" IETIME={start_time},\n"
                    if "IBDATE" in line:
                        addition = f" IBDATE={end_date},\n"
                    if "IBTIME" in line:
                        addition = f" IBTIME={end_time},\n"
                    com_file = com_file + addition
            with open(os.path.join(args.options_path, "COMMAND"), "w") as f:
                f.write(com_file)
        #submit job
        os.system(f"sbatch {args.submit_path}")
        #wait for job to start
        states = get_run_states()
        while len(states) >= args.max_jobs:
            time.sleep(10)
            print(f"Maximal number of jobs active ({len(states)})")
            states = get_run_states()   
        while states.count("R") != len(states):
            time.sleep(2)
            states = get_run_states()
            r_num = states.count("R")
            print(f"{r_num}/{len(states)} tasks running")
        #buffer
        time.sleep(7)
        

            

