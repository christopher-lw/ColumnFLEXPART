import numpy as np

from master.Utils.utils import pressure_factor


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