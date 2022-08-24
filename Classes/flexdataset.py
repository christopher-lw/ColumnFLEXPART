from email import header
from typing import List, Union, Hashable, Any, Dict, Optional, Type, Tuple
from pathlib import Path
import numpy as np
import xarray as xr
import pickle
from copy import deepcopy
from hashlib import md5
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
from abc import ABCMeta, abstractmethod

from utils import yyyymmdd_to_datetime, hhmmss_to_timedelta

CastHashable = Union[Hashable, List, np.ndarray]

class Checkpoint():
    def __init__(self, dir: Union[Path, str], keys: List[str]):
        self.keys = keys
        self.dir = Path(dir)
        self.path = self.dir / "value_checkpoint.pkl"
        self.local = dict()
        self._blank = dict.fromkeys(self.keys)

    @staticmethod
    def get_save_hash(hashable: CastHashable):
        # Casting to a hashable format
        if isinstance(hashable, (list, np.ndarray)):
            hashable = tuple(hashable)
        elif isinstance(hashable, str):
            hashable = hashable.encode()
        #gettig appropriate hash
        if isinstance(hashable, bytes):
            save_hash = str(md5(hashable).digest())
        else:
            save_hash = str(hash(hashable))
        return save_hash        
        
    def __call__(self, hashable: CastHashable, key: str) -> Any:
        """Return value from local checkpoint"""
        save_hash = self.get_save_hash(hashable)
        value = self.local[save_hash][key]
        return value

    def set(self, hashable: CastHashable, key: str, value: Any):
        """Set value in self.local"""
        assert key in self.keys, f"Given key'{key}' not in checkpoint keys: {self.keys}"
        save_hash = self.get_save_hash(hashable)
        if not save_hash in self.local.keys():
            self.local[save_hash] = deepcopy(self._blank)
        self.local[save_hash][key] = value

    def load(self):
        """Loads the checkpoint from disc and stores value in self.local"""        
        with self.path.open("rb") as f:
            self.local = pickle.load(f)

    def save(self):
        """Saves the checkpoint self.local to disk adn replaces fromer save file"""
        with self.path.open("wb") as f:
            pickle.dump(self.local, f)
    
    def update(self):
        """Saves data to disc without removing data not set in self.local"""
        with self.path.open("rb") as f:
            disc_dict = pickle.load(f)
        disc_dict.update(self.local)
        with self.path.open("wb") as f:
            pickle.dump(disc_dict, f)

    def reset(self):
        """Sets self.local to empty dict"""
        self.local = dict()
    
    def delete(self):
        """Removes content of checkpoint on disc"""
        self.path.unlink()    

class DataLoaderBase(metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, dir: Union[Path, str], file_name: Union[Path, str], **kwargs):
        self.dir = Path(dir)
        self.file_name = file_name
        self.files = self.get_files(**kwargs)
        self.data = self.load_data(**kwargs)
    
    @abstractmethod
    def get_files(self, **kwargs):
        """Gets list of files to load"""
        pass
    
    @abstractmethod
    def load_data(self, **kwargs):
        """Loads files of self.files"""

# class DataLoaderAcos(DataLoaderBase):
#     """Dataloader for loading the sounding of """    
#     @staticmethod
#     def get_files(dir: Path, file_name: str, **kwargs):
#         """Gets list of files to load"""
#         start = None
#         stop = None
        


class FlexBase():
    """Class to extract metadata from flexpart output directory"""
    def __init__(self, dir: Union[Path, str]):
        self.dir = Path(dir).resolve()
        self.nc_path = self.get_nc_path(self.dir)
        self.start, self.stop, self.release = self.get_metadata(self.nc_path)
    
    @staticmethod
    def get_nc_path(dir: Path) -> Path:
        """Gets path of nc file of Flexpart output

        Args:
            dir (Path): Flexpart output directory

        Returns:
            Path: Path of file containing footprint
        """
        nc_path = None
        for file in dir.iterdir():
            if "grid_time" in file.stem and ".nc" == file.suffix:
                nc_path = file.resolve()
                break
        assert nc_path is not None, "No nc file found in directory"
        return nc_path

    @staticmethod
    def get_metadata(nc_path: Path) -> Tuple[datetime, datetime, dict]:
        """Reads metadata from flexpart output directory

        Args:
            nc_path (Path): Path of nc_file fo footprint

        Returns:
            Tuple[datetime, datetime, dict]: start, stop, releases
        """        
        dir = nc_path.parent
        header_path = dir / "header_txt"
        releases_path = dir / "RELEASES.namelist"
        with header_path.open("r") as f:
            lines = f.readlines()
        ibdate, ibtime, iedate, ietime = lines[1].strip().split()[:4]
        ibtime = ibtime.zfill(6)
        ietime = ietime.zfill(6)
        start = datetime.strptime(iedate + ietime, "%Y%m%d%H%M%S")
        stop = datetime.strptime(ibdate + ibtime, "%Y%m%d%H%M%S")
        with releases_path.open("r") as f:
            lines = f.readlines()[5:13]

        release = dict()
        for i, line in enumerate(lines):
            lines[i] = line.split("=")[1].strip()[:-1].strip()

        release["start"] = datetime.strptime(
            lines[2] + lines[3].zfill(6), "%Y%m%d%H%M%S"
        )
        release["stop"] = datetime.strptime(
            lines[0] + lines[1].zfill(6), "%Y%m%d%H%M%S"
        )
        release["lon1"] = float(lines[4])
        release["lon2"] = float(lines[5])
        release["lat1"] = float(lines[6])
        release["lat2"] = float(lines[7])
        release["lon"] = (release["lon1"] + release["lon2"]) / 2
        release["lat"] = (release["lat1"] + release["lat2"]) / 2
        dataset = xr.load_dataset(nc_path)
        release["boundary_low"] = dataset.RELZZ1.values
        release["boundary_up"] = dataset.RELZZ2.values
        release["heights"] = np.mean(
            [release["boundary_low"], release["boundary_up"]], axis=0
        )
        return start, stop, release
    
    @staticmethod
    def subplots(*args, **kwargs):
        """Same as matplotlib function only with projection=ccrs.PlateCarree() as default subplot_kw.

        Returns:
            Figure: figure
            Axes: axes of figure
        """
        default_kwargs = dict(subplot_kw=dict(projection=ccrs.PlateCarree()))
        for key, val in kwargs.items():
            default_kwargs[key] = val
        kwargs = default_kwargs
        fig, ax = plt.subplots(*args, **kwargs)
        return fig, ax

    @staticmethod
    def add_map(
        ax: plt.Axes = None,
        feature_list: list[cf.NaturalEarthFeature] = [
            cf.COASTLINE,
            cf.BORDERS,
            [cf.STATES, dict(alpha=0.1)],
        ],
        leave_lims: bool = False,
        **grid_kwargs,
    ) -> tuple[plt.Axes]:
        """Add map to axes using cartopy.

        Args:
            ax (Axes): Axes to add map to
            feature_list (list, optional): Features of cartopy to be added. Defaults to [cf.COASTLINE, cf.BORDERS, [cf.STATES, dict(alpha=0.1)]].
            extent (list, optional): list to define region ([lon1, lon2, lat1, lat2]). Defaults to None.

        Returns:
            Axes: Axes with added map
            Gridliner: cartopy.mpl.gridliner.Gridliner for further settings
        """
        if leave_lims:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        for feature in feature_list:
            feature, kwargs = (
                feature if isinstance(feature, list) else [feature, dict()]
            )
            ax.add_feature(feature, **kwargs)
        grid = True
        gl = None
        try:
            grid = grid_kwargs["grid"]
            grid_kwargs.pop("grid", None)
        except KeyError:
            pass
        if grid:
            grid_kwargs = (
                dict(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                if not bool(grid_kwargs)
                else grid_kwargs
            )
            gl = ax.gridlines(**grid_kwargs)
            gl.top_labels = False
            gl.right_labels = False
        if leave_lims:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        return ax, gl



class Footprint(FlexBase):
    """Class to access the footprint data of Flexpart output"""
    def __init__(self, dir: Union[Path, str], chunks: Optional[Dict] = None, datakey: str="spec001_mr"):
        super().__init__(dir)
        self.chunks = chunks
        self.datakey = datakey
        self.dataset = xr.open_dataset(self.nc_path, chunks=self.chunks)
        self.dataarray = self.dataset[self.datakey]

class Trajectories(FlexBase):
    def __init__(self, dir: Union[Path, str]):
        super().__init__(dir)
        self.dataframe = pd.read_pickle(self._dir / "trajectories.pkl").reset_index()

    
class FlexDataset(FlexBase):
    def __init__(self, dir: Union[Path, str], chunks: Optional[Dict]=None, datakey: str="spec001_mr"):
        super().__init__(dir)
        self.chunks = chunks
        self.datakey = datakey
        self.footprint = Footprint(dir, chunks, datakey)
        
        
        self.trajectories = self.get_trajectories()

    
    
    def get_trajectories(self) -> Optional[Type[Footprint]]:
        raise NotImplementedError
    
    def get_trajectories(self) -> Optional[Type[Trajectories]]:
        raise NotImplementedError

    


    def plot():
        raise NotImplementedError
