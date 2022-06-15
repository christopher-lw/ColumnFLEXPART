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

def setup_column(config: dict) -> tuple[np.ndarray, np.ndarray]:
    if config["column_mode"] == "unit":
        setup = UnitColumnSetup(config)
    elif config["column_mode"] == "pressure":
        setup = PressureColumnSetup(config)
    return setup.height_levels, setup.part_nums

def config_total_parts(config_path, output_path, total_parts):
    with open(config_path) as f:
        config = dict(yaml.full_load(f))
    
    config["n_total"] = total_parts

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved to {output_path}.")

def yyyymmdd_to_datetime64(date_string: str) -> np.datetime64:
    date = np.datetime64(f"{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}")
    return date

def hhmmss_to_timedelta64(time_string: str) -> np.timedelta64:
    time = (np.timedelta64(time_string[:2], "h")
        + np.timedelta64(time_string[2:4], "m") 
        + np.timedelta64(time_string[4:], "s"))
    return time

def datetime64_to_yyyymmdd_and_hhmmss(time: np.datetime64) -> tuple[str, str]:
    string = str(time)
    string = string.replace("-", "").replace(":", "")
    date, time = string.split("T")
    return date, time