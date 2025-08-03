import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget
import matplotlib.pyplot as plt


# Dummy 2D reshaper: reshape 1D sample to 2D (e.g., for plotting)
def reshape_to_2d(sample):
    length = sample.shape[-2]
    size = int(np.sqrt(length))
    return sample.reshape(size, size)


# GUI class
class LivePlotWidget(QWidget):
    def __init__(self, plot_queue):
        super().__init__()
        self.plot_queue = plot_queue
        self.setWindowTitle("Live Sample Viewer")

        # self.figure = Figure(figsize=(10, 10))
        self.figure, ax = plt.subplots(1, 2, figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.ax_sample = ax[0]
        self.ax_stagmag = ax[1]

        self.toggle_button = QPushButton("Switch to Single Sample")
        self.sample_selector = QSpinBox()
        self.sample_selector.setMaximum(1000)
        self.mode_label = QLabel("Mode: Batch Average")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.toggle_button)
        layout.addWidget(QLabel("Sample ID:"))
        layout.addWidget(self.sample_selector)
        self.setLayout(layout)

        self.mode = "average"  # or "single"
        self.toggle_button.clicked.connect(self.toggle_mode)

        # Timer for refreshing
        self.timer = self.startTimer(200)
        self.stag_steps = []
        self.stag_mags = []

    def toggle_mode(self):
        if self.mode == "average":
            self.mode = "single"
            self.toggle_button.setText("Switch to Batch Average")
        else:
            self.mode = "average"
            self.toggle_button.setText("Switch to Single Sample")
        self.mode_label.setText(f"Mode: {'Single Sample' if self.mode == 'single' else 'Batch Average'}")

    def timerEvent(self, event):
        try:
            while not self.plot_queue.empty():
                data = self.plot_queue.get_nowait()
                self.plot_sample(data["samples"])
                self.update_staggered(data["step"], data["stag_mag"])
        except Exception as e:
            print("Plotting error:", e)

    def plot_sample(self, samplebatch):
        self.ax_sample.clear()
        if self.mode == "average":
            avg_sample = samplebatch.mean(axis=0)
            image = reshape_to_2d(avg_sample)
            self.ax_sample.imshow(image, cmap="viridis")
            self.ax_sample.set_title("Average Sample")
        else:
            idx = self.sample_selector.value()
            if 0 <= idx < samplebatch.shape[0]:
                sample = reshape_to_2d(samplebatch[idx])
                self.ax_sample.imshow(sample, cmap="plasma")
                self.ax_sample.set_title(f"Sample ID: {idx}")
            else:
                self.ax_sample.text(0.5, 0.5, "Invalid Sample ID", ha="center")
        self.canvas.draw()

    def update_staggered(self, step, stag_mag):
        self.stag_steps.append(step)
        self.stag_mags.append(stag_mag)

        self.ax_stagmag.clear()
        self.ax_stagmag.plot(self.stag_steps, self.stag_mags, marker=None, linestyle="-")
        self.ax_stagmag.hlines(0, 0, max(self.stag_steps), colors="black", linestyles="dashed")

        self.ax_stagmag.set_ylabel("Staggered Magnetization")
        self.ax_stagmag.set_xlabel("Training Step")
        self.ax_stagmag.set_title("Staggered Magnetization vs Step")
        self.ax_stagmag.grid(True)
        self.canvas.draw()
