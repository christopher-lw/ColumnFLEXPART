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

def load_header():
    """Load header for RELEASE file

    Returns:
        list: List of lines in header for RELEASE file
    """    
    header_path = "dummies/HEADER_dummy.txt"
    with open(header_path) as f:
        header = f.read().splitlines()
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

