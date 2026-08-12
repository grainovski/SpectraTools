import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matrix_cut import compute_projection
from mtx_io import load_mtx
from theme import style_axes


class MatrixPanel(QMainWindow):
    """A separate top-level window for TV-style matrix gate/cut
    analysis. Loads a matrix and computes both its X and Y
    projections up front (this app has the whole matrix in memory,
    unlike TV which required opening a second, transposed matrix file
    for the other axis). The user picks which projection to work on,
    marks a cut region and background regions on it (Task 4), and
    activates the cut to hand a resulting spectrum off to the main
    window. Coexists with MainWindow -- activating one disables, but
    does not close, the other (wired in Task 6)."""

    def __init__(self, main_window, path):
        super().__init__()
        self.main_window = main_window
        self.path = path
        self.matrix = load_mtx(path)
        self.projections = {
            "x": compute_projection(self.matrix, "x"),
            "y": compute_projection(self.matrix, "y"),
        }
        self.working_axis = "x"

        self.setWindowTitle(f"Matrix -- {os.path.basename(path)}")
        self.resize(800, 500)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)

        self.axis_selector = QComboBox()
        self.axis_selector.addItem("X projection", "x")
        self.axis_selector.addItem("Y projection", "y")
        self.axis_selector.currentIndexChanged.connect(self._on_axis_changed)

        self.heatmap_button = QPushButton("Show Heatmap...")

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Working on:"))
        top_bar.addWidget(self.axis_selector)
        top_bar.addStretch()
        top_bar.addWidget(self.heatmap_button)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self._plot_projection()

    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self._plot_projection()

    def _plot_projection(self):
        self.axes.clear()
        style_axes(self.axes, self.main_window._theme)
        data = self.projections[self.working_axis]
        self.axes.plot(range(len(data)), data, linewidth=0.8)
        self.axes.set_xlabel(f"{self.working_axis.upper()} channel")
        self.axes.set_ylabel("Counts")
        self.canvas.draw()
