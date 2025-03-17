import sys

import torch

from tvmc.utils.builder import build_model
from tvmc.utils.helper import setup_dir
from tvmc.utils.training import reg_train
from tvmc.utils.config import *


if __name__ == "__main__":
    _, config = parse_config("config.json")
    
    model, config = build_model(config)
    train_opt = config["TRAIN"]

    # Initialize optimizer
    beta1 = 0.9
    beta2 = 0.999
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["lr"], betas=(beta1, beta2))

    mydir = setup_dir(config)
    orig_stdout = sys.stdout

    # config.save(mydir + "\\settings.json")

    print("Starting Training...")
    reg_train(config, (model, optimizer), printf=True, mydir=mydir)
