import sys
import gc

import torch

from tvmc.utils.builder import build_model
from tvmc.utils.helper import setup_dir
from tvmc.utils.training import reg_train
from tvmc.utils.config import parse_config


def run(config):
    model, config = build_model(config)

    # Initialize optimizer
    beta1 = 0.9
    beta2 = 0.999
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["lr"], betas=(beta1, beta2))

    output_path = setup_dir(config)

    print("Starting Training...")
    reg_train(config, (model, optimizer), printf=True, output_path=output_path, resume=True)

    # Clean up after each run
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    config_path = "./config.json"
    _, config = parse_config(config_path)
    run(config)
