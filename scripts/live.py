import multiprocessing as mp
import sys

from PyQt5.QtWidgets import QApplication
from threading import Thread
from tvmc.utils.config import parse_config
from tvmc.utils.training import reg_train
from tvmc.utils.view import LivePlotWidget

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
        reg_train(
            config,
            plot_queue=plot_queue,
            printf=True,
            output_path=config["TRAIN"]["dir"],
            start_time=None,  # No start time needed for live plotting
            resume=False,
        )

    # Start training in a separate process
    t = Thread(target=train)
    t.start()

    # Start the GUI in the main process
    app = QApplication(sys.argv)
    window = LivePlotWidget(plot_queue)
    window.show()
    app.exec_()
