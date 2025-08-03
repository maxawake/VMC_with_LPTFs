import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # for large models
from datetime import datetime
import sys

import numpy as np

from tvmc.utils.config import parse_config, save_config
from tvmc.utils.training import reg_train

if __name__ == "__main__":
    # Parse resume boolean from command line arguments
    resume = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--resume":
        resume = True

    config_path = "config.json"
    _, config = parse_config(config_path, show=True)
    base_path = config["TRAIN"]["dir"]

    start_time = datetime.now()

    system_sizes = [8 * 8, 16 * 16, 24 * 24, 32 * 32, 40 * 40]
    for L in system_sizes:
        config["TRAIN"]["L"] = L
        config["LPTF"]["L"] = L
        config["TRAIN"]["NLOOPS"] = int(L / (2 * 2))
        config["TRAIN"]["dir"] = os.path.join(base_path, f"output_{int(np.sqrt(L))}x{int(np.sqrt(L))}")

        # save config and run
        save_config(config, config["TRAIN"]["dir"])
        reg_train(
            config,
            plot_queue=None,
            printf=False,
            output_path=config["TRAIN"]["dir"],
            start_time=start_time,
            resume=resume,
        )
