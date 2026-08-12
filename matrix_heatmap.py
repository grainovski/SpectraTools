import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget


class MatrixHeatmapWindow(QMainWindow):
    """A separate, non-modal, view-only visualization of a loaded
    matrix -- opened on demand from the matrix panel. No markers, no
    interaction beyond the standard matplotlib zoom/pan/reset toolbar.
    TV itself never had any 2D matrix rendering at all (confirmed
    absent from its entire source tree during design); this is a
    genuine enhancement, not a port."""

    def __init__(self, matrix, path):
        super().__init__()
        self.setWindowTitle(f"Heatmap -- {os.path.basename(path)}")
        self.resize(700, 700)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # The stock (untrimmed) toolbar -- unlike this app's 1D
        # spectrum view, a 2D image needs rectangle-select Zoom, which
        # _TrimmedNavigationToolbar deliberately removes for the 1D
        # case (main_window.py's own _zoom_x buttons cover that
        # instead). Home/Pan/Zoom/Save all apply naturally to an image.
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        # SymLogNorm handles the negative values this app's matrix
        # data can genuinely contain (random-coincidence subtraction)
        # without error -- plain LogNorm requires strictly positive
        # data and would raise on any matrix with a negative cell.
        norm = SymLogNorm(linthresh=1.0, vmin=matrix.min(), vmax=max(matrix.max(), 1))
        self.image = self.axes.imshow(matrix, norm=norm, origin="lower", aspect="auto")
        self.figure.colorbar(self.image, ax=self.axes)
        self.axes.set_xlabel("X channel")
        self.axes.set_ylabel("Y channel")
        self.canvas.draw()
