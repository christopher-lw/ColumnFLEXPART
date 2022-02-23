### Utilities for column receptor setup ###

import numpy as np

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
        assert False, "times_from_file not yet implemented"
        assert "times_file_path" in config.keys(), "First set times_file_path in config or set times_from_file to 'true'"
        with open(config["times_file_path"], "r") as f:
            lines = f.read().splitlines()

        times = []
        for line in lines:
            times.append([line.split(",")])
    return times

def setup_coords(config):
    if not config["coords_from_file"]:
        coords = config["coords"]

    else:
        assert False, "coords_from_file not yet implemented"
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
        self.regions = config["regions"]
        self.dh = config["dh"]
        self.height = self.regions[0]*1e3
        self.height_levels = [self.height]
        self.part_nums = []
        self.new_part_nums = []
        self.new_heights = []
        
        self.prepare_dh()
        self.prepare_config()
        self.run()

    def run(self):
        for i, region in enumerate(self.regions[1:]):
            self.new_heights = []
            self.new_part_nums = []
            while self.height < region*1e3:
                self.get_height(i)
                self.add_new_height()
                self.add_part_nums(i)
            self.extend_height_levels()
            self.process_part_nums(i)
            self.extend_part_nums()
        
        self.height_levels = np.array(self.height_levels)
        self.part_nums = np.array(self.part_nums).round(0).astype(int)


    def prepare_dh(self):
        if not isinstance(self.dh, list): self.dh = [self.dh for i in range(len(self.regions)-1)]

    def add_new_height(self):
        self.new_heights.append(self.height)

    def get_height(self, i):
        self.height += self.dh[i]

    def extend_height_levels(self):
        self.height_levels.extend(self.new_heights)

    def extend_part_nums(self):
        self.part_nums.extend(self.new_part_nums)
    
    def prepare_config(self):
        pass
    
    def process_part_nums(self, i):
        pass
    
    def add_part_nums(self, i):
        pass

class WuColumnSetup(ColumnSetup):
    def __init__(self, config):
        super().__init__(config)

    def prepare_config(self):
        if not isinstance(self.config["dn"], list): self.config["dn"] = [self.config["dn"] for i in range(len(self.regions)-1)]
    
    def add_part_nums(self, i):
        self.part_nums.append(self.config["dn"][i])

class PressureColumnSetup(ColumnSetup):
    def __init__(self, config):
        super().__init__(config)

    def process_part_nums(self, i):
        if i == len(self.regions)-2:
            height_levels = np.array(self.height_levels)
            diff_heights = height_levels[1:] - height_levels[:-1]
            mid_heights = (height_levels[:-1] + height_levels[1:])/2
            factors = pressure_factor(mid_heights)
            norm = factors[mid_heights <= self.regions[1]*1e3].sum()
            factors = factors/norm * diff_heights/self.dh[0]
            self.new_part_nums = list(self.config["lowest_region_n"] * factors)

class PressureWuColumnSetup(ColumnSetup):
    def __init__(self, config):
        super().__init__(config)

    def process_part_nums(self, i):
        factors = pressure_factor(np.array(self.new_heights))
        factors = factors/np.sum(factors)
        self.new_part_nums = list(np.array(self.config["region_n"][i]) * factors)

def setup_column(config):
    if config["column_mode"] == "wu":
        setup = WuColumnSetup(config)
    elif config["column_mode"] == "pressure":
        setup = PressureColumnSetup(config)
    elif config["column_mode"] == "pressure_wu":
        setup = PressureWuColumnSetup(config)
    return setup.height_levels, setup.part_nums