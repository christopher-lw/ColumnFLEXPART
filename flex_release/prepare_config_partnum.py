import argparse
import os
from utils import config_total_parts
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manipulation tool for config files. Creates config files with desired total particle numbers")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("outpath", type=str, help="path for output")
    parser.add_argument("start", type=int, help="first input for np.arange")
    parser.add_argument("end", type=int, help="second input for np.arange")
    parser.add_argument("step", type=int, help="third input for np.arange")
    parser.add_argument("--name", type=str, default=None, help="start of name of output files if not set name of config is used")
    args = parser.parse_args()
    
    part_nums = np.arange(args.start, args.end, args.step)
    name = args.config.split("/")[-1].split(".")[0] if args.name is None else args.name
    
    
    for part_num in part_nums:
        num_str = str(int(part_num/1e3))
        num_str = "0"*(3 - len(num_str)) + num_str
        outpath = os.path.join(args.outpath, name + num_str + "k.yaml")
        
        config_total_parts(args.config, outpath, int(part_num))