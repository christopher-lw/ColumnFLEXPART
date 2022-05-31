### Utilities for column receptor setup ###

import numpy as np
import yaml

def pressure_factor(
    h,
    Tb = 288.15,
    hb = 0,
    R = 8.3144598,
    g = 9.80665,
    M = 0.0289644,
    ):
    """Calculate factor of barrometric height formula as described here: https://en.wikipedia.org/wiki/Barometric_formula

    Args:
        h (fleat): height for factor calculation [m]
        Tb (float, optional): reference temperature [K]. Defaults to 288.15.
        hb (float, optional): height of reference [m]. Defaults to 0.
        R (float, optional): universal gas constant [J/(mol*K)]. Defaults to 8.3144598.
        g (float, optional): gravitational acceleration [m/s^2]. Defaults to 9.80665.
        M (float, optional): molar mass of Earth's air [kg/mol]. Defaults to 0.0289644.

    Returns:
        float: fraction of pressure at height h compared to height hb
    """    
    factor = np.exp(-g * M * (h - hb)/(R * Tb))
    return factor

def inv_pressure_factor(
    factor,
    Tb = 288.15,
    hb = 0,
    R = 8.3144598,
    g = 9.80665,
    M = 0.0289644,
    ):
    """Calculate height form inverse of barrometric height formula as described here: https://en.wikipedia.org/wiki/Barometric_formula

    Args:
        h (fleat): height for factor calculation [m]
        Tb (float, optional): reference temperature [K]. Defaults to 288.15.
        hb (float, optional): height of reference [m]. Defaults to 0.
        R (float, optional): universal gas constant [J/(mol*K)]. Defaults to 8.3144598.
        g (float, optional): gravitational acceleration [m/s^2]. Defaults to 9.80665.
        M (float, optional): molar mass of Earth's air [kg/mol]. Defaults to 0.0289644.

    Returns:
        float: height h at fraction of pressure relative to hb
    """    
    h = (R * Tb) / (-g * M) *  np.log(factor) + hb
    return h

def load_header(species=41):
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

def load_dummy():
    """Load dummy for RELEASE file.

    Returns:
        list: List of lines in dummy RELEASE file
    """    
    dummy_path = "dummies/RELEASES_dummy.txt"
    with open(dummy_path) as f:
        dummy = f.read().splitlines()
    return dummy

def write_to_dummy(date1, time1, date2, time2, lon1, lon2, 
    lat1, lat2, z1, z2, zkind, mass, parts, comment):
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

def setup_times(config):
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

def setup_coords(config):
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

class ColumnSetup():
    def __init__(self, config):
        self.config = config
        self.height_low = config["height_low"]
        self.height_up = config["height_up"]
        self.dh = config["dh"]
        self.n_total = config["n_total"]
        self.height_levels = []
        self.part_nums = []
        self.n_levels = int((self.height_up - self.height_low) * 1e3 / self.dh)

        self.get_height_levels()
        self.get_part_nums()
    
    def get_height_levels(self):
        self.height_levels = np.linspace(self.height_low*1e3, self.height_up*1e3, self.n_levels + 1).astype(int)

    def get_part_nums(self):
        pass

class UnitColumnSetup(ColumnSetup):
    def __init__(self, config):
        super().__init__(config)
    
    def get_part_nums(self):
        dn = self.n_total / self.n_levels
        self.part_nums = np.array([dn] * self.n_levels, dtype=int)

class PressureColumnSetup(ColumnSetup):
    def __init__(self, config):
        super().__init__(config)

    def get_part_nums(self):
        mid_heights = (self.height_levels[1:] + self.height_levels[:-1]) / 2
        factors = pressure_factor(mid_heights)
        factors = factors / np.sum(factors)
        self.part_nums = self.n_total * factors
        self.part_nums = self.part_nums.astype(int)

def setup_column(config):
    if config["column_mode"] == "unit":
        setup = UnitColumnSetup(config)
    elif config["column_mode"] == "pressure":
        setup = PressureColumnSetup(config)
    elif config["column_mode"] == "unit_single":
        setup = UnitSinglePartColumnSetup(config)
    elif config["column_mode"] == "pressure_single":
        setup = PressureSinglePartColumnSetup(config)
    return setup.height_levels, setup.part_nums

def config_total_parts(config_path, output_path, total_parts):
    with open(config_path) as f:
        config = dict(yaml.full_load(f))
    
    config["n_total"] = total_parts

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved to {output_path}.")

    
class SinglePartColumnSetup():
    def __init__(self, config):
        self.config = config
        self.regions = config["regions"]
        self.parts = config["n"]
        self.part_nums = [1]*self.parts
        self.height_levels = []
        self.prepare_height_levels()

    def prepare_height_levels(self):
        pass

class UnitSinglePartColumnSetup(SinglePartColumnSetup):
    def __init__(self, config):
        super().__init__(config)

    def prepare_height_levels(self):
        self.height_levels = list(np.linspace(self.regions[0]*1e3, self.regions[1]*1e3, self.parts + 1))

class PressureSinglePartColumnSetup(SinglePartColumnSetup):
    def __init__(self, config):
        super().__init__(config)

    def prepare_height_levels(self):
        pressure_regions = pressure_factor(np.array(self.regions) * 1e3)
        pressures = np.linspace(*pressure_regions, self.parts+1)
        self.height_levels = list(inv_pressure_factor(pressures))