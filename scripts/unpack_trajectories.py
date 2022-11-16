from master.scripts.split_releases import convert_partposit, convert_partposit, get_output_dirs, combine_to_trajectories

from argparse import ArgumentParser
from tqdm.auto import tqdm
import os


if __name__ == '__main__':
    parser = ArgumentParser(description = "Script to convert partposit files of FLEXPART output to xarray datasets and collects hole data in pickle file trajectories.pkl. Can e used on singel output directory or multiple ones")
    parser.add_argument("dir", type=str, help="Flexpart output directory to start search for partposit files.")
    parser.add_argument("-r", action="store_true", help="Flag to recursively search for flexpart output directories to split.")

    args = parser.parse_args()

    output_dirs = get_output_dirs(args.dir, args.r)

    for dir in tqdm(output_dirs, desc="Directories unpacked"):
        data, files = convert_partposit(dir)
        for d, f in zip(data, files):
            d.to_netcdf(os.path.join(dir, f"{f}.nc"))
        combine_to_trajectories(dir)

