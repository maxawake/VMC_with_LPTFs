import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QSlider, QLabel, QFrame
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Simulation parameters
L = 128
step_size = 0.1
update_interval_ms = 10  # ~20 FPS
phi = np.random.normal(0, 0.1, size=(L, L))  # Initial field

# Default action coefficients
params = {"c1": 0.0, "a2": -1.0, "a3": 0.0, "a4": 1.0, "a5": 0.0, "a6": 0.0, "beta": 2.0}


def metropolis_step(phi, p):
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        old = phi[i, j]
        new = old + np.random.normal(0, step_size)

        # Neighbor coupling (gradient term)
        neighbors = phi[(i + 1) % L, j] + phi[(i - 1) % L, j] + phi[i, (j + 1) % L] + phi[i, (j - 1) % L]
        grad_old = 0.5 * (4 * old**2 - 2 * old * neighbors)
        grad_new = 0.5 * (4 * new**2 - 2 * new * neighbors)

        # Potential terms (up to 6th order)
        pot_old = (
            p["c1"] * old + p["a2"] * old**2 + p["a3"] * old**3 + p["a4"] * old**4 + p["a5"] * old**5 + p["a6"] * old**6
        )
        pot_new = (
            p["c1"] * new + p["a2"] * new**2 + p["a3"] * new**3 + p["a4"] * new**4 + p["a5"] * new**5 + p["a6"] * new**6
        )

        dS = p["beta"] * ((grad_new + pot_new) - (grad_old + pot_old))

        if np.random.rand() < np.exp(-dS):
            phi[i, j] = new


# PyQt5 GUI
class SimulationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("φ⁶ Effective Action Sampling")
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(update_interval_ms)

    def init_ui(self):
        layout = QHBoxLayout(self)

        # Sliders
        controls = QVBoxLayout()
        for name, (vmin, vmax, step) in {
            "c1": (-10.0, 10.0, 0.01),
            "a2": (-10.0, 10.0, 0.01),
            "a3": (-10.0, 10.0, 0.01),
            "a4": (-10.0, 10.0, 0.01),
            "a5": (-10.0, 10.0, 0.01),
            "a6": (-10.0, 10.0, 0.01),
            "beta": (0.01, 5.0, 0.01),
        }.items():
            controls.addWidget(self.make_slider(name, vmin, vmax, step))

        # Reset button
        reset_btn = QPushButton("Reset Field")
        reset_btn.clicked.connect(self.reset_field)
        controls.addWidget(reset_btn)

        layout.addLayout(controls)

        # Plot area
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.im = self.ax.imshow(phi, cmap="coolwarm", vmin=-1, vmax=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        layout.addWidget(self.canvas)

    def make_slider(self, name, vmin, vmax, step):
        label = QLabel(f"{name}: {params[name]:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(int((vmax - vmin) / step))
        slider.setValue(int((params[name] - vmin) / step))

        def update(val):
            value = vmin + val * step
            params[name] = value
            label.setText(f"{name}: {value:.2f}")

        slider.valueChanged.connect(update)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(slider)
        frame = QFrame()
        frame.setLayout(layout)
        return frame

    def reset_field(self):
        global phi
        phi = np.random.normal(0, 0.1, size=(L, L))
        self.update_canvas()

    def update_simulation(self):
        metropolis_step(phi, params)
        self.update_canvas()

    def update_canvas(self):
        self.im.set_data(phi)
        self.im.set_clim(vmin=-1, vmax=1)
        self.canvas.draw()


# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulationWindow()
    window.show()
    sys.exit(app.exec_())
