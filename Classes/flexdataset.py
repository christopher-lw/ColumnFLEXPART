from typing import List, Union, Hashable, Any, Dict, Optional
from pathlib import Path
import numpy as np
import xarray as xr
import pickle
from copy import deepcopy
from hashlib import md5

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
        
    
class FlexDataset():
    def __init__(self, dir: Union[Path, str], chunks: Optional[Dict]=None, datakey: str="spec001_mr"):
        self.dir = Path(dir)
        self.chunks = chunks
        self.nc_path = self.get_nc_path(self.dir)
        self.footprint = Footprint(self)

    @staticmethod
    def get_nc_path(dir: Path) -> Path:
        """Gets path of nc file of Flexpart output

        Args:
            dir (Path): Flexpart output directory

        Returns:
            Path: Path of file containing footprint
        """         

class Footprint():
    def __init__(self, flexdataset: FlexDataset):
        self.flexdataset = flexdataset
        self.dataset = xr.open_dataset(self.flexdataset.nc_path, chunks=self.flexdataset.chunks)
        self.dataarray = self.dataset[self.flexdataset.datakey]

    def plot():
        raise NotImplementedError