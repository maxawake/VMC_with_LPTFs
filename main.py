import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # for large models
import sys

import numpy as np

from run import run
from tvmc.utils.config import parse_config, save_config

if __name__ == "__main__":
    # Parse resume boolean from command line arguments
    resume = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--resume":
        resume = True

    config_path = "config.json"
    _, config = parse_config(config_path, show=True)
    base_path = config["TRAIN"]["dir"]

    system_sizes = [8 * 8, 16 * 16, 24 * 24, 32 * 32, 40 * 40]
    for L in system_sizes:
        config["TRAIN"]["L"] = L
        config["LPTF"]["L"] = L
        config["TRAIN"]["NLOOPS"] = int(L / (2 * 2))
        config["TRAIN"]["dir"] = os.path.join(base_path, f"output_{int(np.sqrt(L))}x{int(np.sqrt(L))}")

        # save config and run
        save_config(config, config["TRAIN"]["dir"])
        run(config, resume=resume)
