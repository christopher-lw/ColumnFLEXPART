import pytest
import os
from pathlib import Path
from flexdataset import Checkpoint, FlexDataset
import numpy as np
from copy import deepcopy
from flexdataset import FlexBase
from utils import yyyymmdd_to_datetime, hhmmss_to_timedelta
import xarray as xr
from datetime import datetime

#################################################################
#################### Checkkpoint Class ##########################
#################################################################
@pytest.fixture(params = [["a", "b"]])
def get_checkpoint(tmp_path, request):
    checkpoint = Checkpoint(dir = tmp_path, keys = request.param)
    return checkpoint

@pytest.fixture(params = [dict(hashable="test", keys=["a", "b"], values=[10, None])])
def get_nonempty_checkpoint(tmp_path, request):
    hashable = request.param["hashable"]
    keys = request.param["keys"]
    values  = request.param["values"]
    checkpoint = Checkpoint(dir = tmp_path, keys = keys)
    for key, value in zip(keys, values):
        checkpoint.set(hashable, key, value)
    return checkpoint

@pytest.fixture(params=["test", [0,1], np.arange(2), None])
def get_hashable(request):
    return request.param

@pytest.fixture(params = [1, "test", np.ones((2,2)), [1,2,4], dict(a=4)])
def get_value(request):
    return request.param

def test_get_save_hash(get_hashable):
    """Test if format of output has works"""
    hash_val1 = Checkpoint.get_save_hash(get_hashable)
    hash_val2 = Checkpoint.get_save_hash(get_hashable)
    assert isinstance(hash_val1, str)
    assert hash_val1 == hash_val2

def test_set_initialization(get_checkpoint, get_hashable, get_value):
    """Test if initialization by using set works as expected"""  
    checkpoint = get_checkpoint
    hashable = get_hashable
    value = get_value
    checkpoint.set(hashable, "a", value)
    assert checkpoint.local[checkpoint.get_save_hash(hashable)]["a"] is value
    assert checkpoint.local[checkpoint.get_save_hash(hashable)]["b"] is None

@pytest.mark.xfail
def test_set_wrong_key(get_checkpoint):
    checkpoint = get_checkpoint
    checkpoint.set("test", "neiter_a_nor_b", 10)

@pytest.mark.parametrize("hashable, key, value", [([1,2,3], "a", 10)])
def test_call(get_checkpoint, hashable, key, value):
    checkpoint = get_checkpoint
    checkpoint.set(hashable, key, value)
    assert checkpoint(hashable, key) is value 

def test_save(tmp_path):
    """Tests if loacl is saved to the right location"""
    assert tmp_path.exists()
    checkpoint = Checkpoint(tmp_path, ["a", "b"])
    checkpoint.set("test", "a", 10)
    checkpoint.save()
    assert checkpoint.path == tmp_path / "value_checkpoint.pkl"
    assert checkpoint.path.exists()

def test_save_load(tmp_path, get_value):
    """Tests if loaded dict is the same as saved one"""
    checkpoint = Checkpoint(tmp_path, ["a", "b"])
    value = get_value
    hashable = "test"
    key = "a"
    checkpoint.set(hashable, key, value)
    checkpoint.save()
    checkpoint.load()
    assert isinstance(checkpoint.local, dict)
    test_val = checkpoint(hashable, key) == value
    if isinstance(value, np.ndarray):
        assert test_val.all()
    else:
        assert test_val

def test_load_overwirte(get_nonempty_checkpoint):
    """Tests if load correctly overwirtes local"""
    checkpoint = get_nonempty_checkpoint
    old_local = deepcopy(checkpoint.local)
    checkpoint.save()
    checkpoint.set("test2", "b", 23)
    checkpoint.load()
    assert checkpoint.local == old_local


def test_reset(get_nonempty_checkpoint):
    """Tests if local checkpoint is resetted correctly"""
    checkpoint = get_nonempty_checkpoint
    checkpoint.reset()
    assert checkpoint.local == dict()

def test_delete(get_nonempty_checkpoint):
    """Tests if content of saved checkpoint is removed correctly"""
    checkpoint = get_nonempty_checkpoint
    checkpoint.save()
    checkpoint.delete()
    assert checkpoint.local != dict()
    assert not checkpoint.path.exists()

def test_update(get_nonempty_checkpoint):
    """Tests if data is correctly updated"""
    checkpoint = get_nonempty_checkpoint
    old_local = deepcopy(checkpoint.local)
    checkpoint.save()
    checkpoint.reset()
    checkpoint.set("test2", "b", 23)
    new_local = deepcopy(checkpoint.local)
    checkpoint.update()
    checkpoint.load()
    loaded_local = deepcopy(checkpoint.local)
    assert old_local.items() < loaded_local.items()
    assert new_local.items() < loaded_local.items()


#################################################################
#################### FlexBase Class #############################
#################################################################

@pytest.fixture
def output_directory_and_nc_file():
    directory = Path("test_data/flex_output").resolve()
    nc_file = "grid_time_20091130210000.nc"
    return directory, nc_file

@pytest.fixture
def get_dataset(output_directory_and_nc_file):
    directory, nc_file = output_directory_and_nc_file
    nc_path = directory / nc_file
    dataset = xr.load_dataset(nc_path)
    return dataset

# @pytest.fixture
# def get_FlexBase(output_directory):
#     fb = FlexBase(output_directory)

def test_get_nc_path(output_directory_and_nc_file):
    """Tests if FlexBase extracts correct nc_path"""
    directory, nc_file = output_directory_and_nc_file
    nc_path = FlexBase.get_nc_path(directory)
    assert nc_path == directory / nc_file
    # nc_file = None
    # for file in output_directory.iterdir():
    #     if "grid_time" in file.stem and ".nc"

def test_get_metadata(output_directory_and_nc_file, get_dataset):
    test_start = yyyymmdd_to_datetime("20091130") + hhmmss_to_timedelta("210000") 
    test_stop = yyyymmdd_to_datetime("20091116") + hhmmss_to_timedelta("195959")
    test_release = dict()
    test_release["start"] = yyyymmdd_to_datetime("20091130") + hhmmss_to_timedelta("200000")
    test_release["stop"] = yyyymmdd_to_datetime("20091130") + hhmmss_to_timedelta("195959")
    test_release["lon1"] = 150.878998
    test_release["lon2"] = 150.878998
    test_release["lat1"] = -34.4059982
    test_release["lat2"] = -34.4059982
    test_release["lon"] = (test_release["lon1"] + test_release["lon2"]) / 2
    test_release["lat"] = (test_release["lat1"] + test_release["lat2"]) / 2
    test_release["boundary_low"] = get_dataset.RELZZ1.values
    test_release["boundary_up"] = get_dataset.RELZZ2.values
    test_release["heights"] = np.mean(
            [test_release["boundary_low"], test_release["boundary_up"]], axis=0
        )

    directory, nc_file = output_directory_and_nc_file
    nc_path  = (directory / nc_file).resolve()
    start, stop, release = FlexBase.get_metadata(nc_path)

    assert isinstance(start, datetime)
    assert isinstance(stop, datetime)
    assert isinstance(release, dict)
    assert test_start == start
    assert test_stop == stop
    assert test_release.keys() == release.keys()
    for (_, test_value), (key, value) in zip(test_release.items(), release.items()):
        try: 
            assert (test_value == value).all()
        except AttributeError:
            assert test_value == value

#################################################################
#################### FlexBase Class #############################
#################################################################

