import sys

import torch

from tvmc.models.ModelBuilder import build_model
from tvmc.models.training import OptionManager, TrainOpt, reg_train, setup_dir


def helper(args):
    help(build_model)

    example = "Runtime Example:\n>>>python train.py --rydberg --train L=144"
    while True:
        if "--lptf" in args:
            print(LPTF.INFO)
            print(example + " --lptf patch=3x3 --rnn L=9 patch=3 Nh=128")
            break
        if "--rnn" in args:
            print(PRNN.INFO)
            print(example + " NLOOPS=36 --rnn patch=4")
            break
        if "--ptf" in args:
            print(PTF.INFO)
            print(example + " NLOOPS=24 --ptf patch=2x3")
            break
        if "--train" in args:
            print(TrainOpt.__doc__)
            print(example + " NLOOPS=36 sgrad=False steps=4000 --ptf patch=2x2")
            break

        args = ["--" + input("What Model do you need help with?\nOptions are rnn, lptf, ptf, and train:\n".lower())]


if __name__ == "__main__":
    if "--help" in sys.argv:
        print()
        helper(sys.argv)
    else:
        args = [
            "--train",
            "L=64",
            "NLOOPS=16",
            "K=1024",
            "sub_directory=2x2",
            "--ptf",
            "patch=2x2",
            "--rydberg",
            "V=7",
            "delta=1",
            "Omega=1",
        ]

        model, full_opt, opt_dict = build_model(args)
        train_opt = opt_dict["TRAIN"]

        OptionManager.register("train", TrainOpt())

        # Initialize optimizer
        beta1 = 0.9
        beta2 = 0.999
        optimizer = torch.optim.Adam(model.parameters(), lr=train_opt.lr, betas=(beta1, beta2))

        mydir = setup_dir(opt_dict)
        orig_stdout = sys.stdout

        full_opt.save(mydir + "\\settings.json")

        print("Starting Training...")
        reg_train(opt_dict, (model, optimizer), printf=True, mydir=mydir)
