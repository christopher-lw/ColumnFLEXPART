# python script to start multiple flexpart runs with different releases
import os
import shutil
import argparse
import errno
import time
import subprocess


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to start multiple FLEXPART runs with different RELAESES files")
    parser.add_argument("flex_path", type=str, help="path to home directory of FLEXPART to be used")
    parser.add_argument("options_path", type=str, help="path to options directory for the runs (if not absolute path it starts from flex_path)")
    parser.add_argument("output_path", type=str, help="path for output directory (if not absolute path it starts from flex_path)")
    parser.add_argument("release_path", type=str, help="path of directory with RELEASES files (if not absolute path it starts from options_path)")
    parser.add_argument("submit_path", type=str, help="path to slurm script for submission of runs")
    args = parser.parse_args()

    if args.options_path[0] != "/":
        args.options_path = os.path.join(args.flex_path, args.options_path)

    if args.release_path[0] != "/":
        args.release_path = os.path.join(args.options_path, args.release_path)

    if args.output_path[0] != "/":
        args.output_path = os.path.join(args.flex_path, args.output_path)

    for arg in vars(args): 
        path = getattr(args, arg)
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        print(f"{arg} = {path}")

    inp = input("\nIs the submit file properly prepared? ([y]/n) ") or "y"
    assert inp == "y", "Insert y to continue."

    #print(f"Looking for RELEASES files in :{args.release_path} ...  \n")
    rel_names = os.listdir(args.release_path)
    print(f"Found following files in release_path: {rel_names} \n")

    if os.path.exists(os.path.join(args.options_path, "RELEASES")):
        inp = input("Save old RELEASE file? (y/[n]) ") or "n"
        if inp == "y":
            inp = input("New name of old RELEASES file: ")
            shutil.copyfile(os.path.join(args.options_path, "RELEASES"), os.path.join(args.options_path, inp))
            print(f"Saved to {os.path.join(args.options_path, inp)}")

    inp = input("Start runs? ([y]/n) ") or "y"
    assert inp == "y", "Insert y to continue."

    for rel in rel_names:
        shutil.copyfile(os.path.join(args.release_path, rel), os.path.join(args.options_path, "RELEASES"))
        new_output_path = os.path.join(args.output_path, rel)
        os.makedirs(new_output_path, exist_ok=True)
        paths = ""
        with open(os.path.join(args.flex_path, "pathnames")) as f:
            for i, line in enumerate(f):
                addition = line
                if i == 1:
                    addition = f"{new_output_path}\n"
                paths = paths + addition
        
        with open(os.path.join(args.flex_path, "pathnames"), "w") as f:
            f.write(paths)
    
        os.system(f"sbatch {args.submit_path}")
        
        states = get_run_states()
        while states.count("R") != len(states):
            time.sleep(2)
            states = get_run_states()
            r_num = states.count("R")
            print(f"{r_num}/{len(states)} tasks running")
        time.sleep(2)
        

            

