import gc
import multiprocessing as mp
import sys

import torch
from PyQt5.QtWidgets import QApplication
from threading import Thread
from tvmc.utils.builder import build_model
from tvmc.utils.config import parse_config
from tvmc.utils.training import reg_train
from tvmc.utils.view import LivePlotWidget


def run(config, plot_queue=None, resume=False):
    model, config = build_model(config)

    # Initialize optimizer
    beta1 = 0.9
    beta2 = 0.999
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["lr"], betas=(beta1, beta2))

    print("Starting Training...")
    reg_train(config, (model, optimizer), plot_queue=plot_queue, printf=True, resume=resume)

    # Clean up after each run
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    config_path = "./config.json"
    _, config = parse_config(config_path)

    L = 8 * 8
    config["TRAIN"]["L"] = L
    config["LPTF"]["L"] = L
    config["TRAIN"]["NLOOPS"] = int(L / (2 * 2))

    mp.set_start_method("spawn")  # safer for PyTorch + Qt on some OSes

    plot_queue = mp.Queue()

    # Launch training in a thread
    def train():
        run(config, plot_queue=plot_queue)

    # Start training in a separate process
    t = Thread(target=train)
    t.start()

    # Start the GUI in the main process
    app = QApplication(sys.argv)
    window = LivePlotWidget(plot_queue)
    window.show()
    app.exec_()
