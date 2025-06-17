import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # for large models
from run import run
from tvmc.utils.config import parse_config
import numpy as np

if __name__ == "__main__":
    config_path = "config.json"
    _, config = parse_config(config_path)
    print(config)

    system_sizes = [8 * 8]
    for L in system_sizes:
        config["TRAIN"]["L"] = L
        config["LPTF"]["L"] = L
        config["TRAIN"]["NLOOPS"] = int(L / (2 * 2))
        config["TRAIN"]["dir"] = f"output_{int(np.sqrt(L))}x{int(np.sqrt(L))}"
        run(config)
