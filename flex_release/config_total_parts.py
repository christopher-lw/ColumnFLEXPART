import yaml
import numpy as np
def config_total_parts(config_path, output_path, total_parts):
    with open(config_path) as f:
        config = dict(yaml.full_load(f))
    
    if config["column_mode"] == "wu":
        levels = 0
        for i, region in config["regions"][1:]:
            dh = config["dh"][i]
            levels += region*1e3//dh
        config["dn"] = int(total_parts/levels)

    elif config["column_mode"] == "pressure":
        assert len(config["regions"]) == 2, "For pressure setup use only one region [start, end]"
        config["lowest_region_n"] = total_parts
    
    elif config["column_mode"] == "pressure_wu":
        level_list = []
        levels = 0
        for i, region in config["regions"][1:]:
            dh = config["dh"][i]
            level_list.append(region*1e3//dh)
            levels += region*1e3//dh
        level_arr = np.array(level_list)
        region_n = total_parts * level_arr/levels
        config["region_n"] = list(region_n.astype(int))

    with open(output_path, "w") as f:
        yaml.dump(config, f)
    