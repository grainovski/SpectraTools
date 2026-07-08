import os

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from histogram_io import ParseError, load_histogram


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Viewer")
        self.resize(900, 600)

        self.data = None

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self._build_menu()

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Histogram", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            data = load_histogram(path)
        except ParseError as exc:
            QMessageBox.warning(self, "No histogram data found in file", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Could not open file", str(exc))
            return

        self.data = data
        self.setWindowTitle(f"Histogram Viewer - {os.path.basename(path)}")
        self._plot_data()

    def _plot_data(self):
        self.axes.clear()
        channels = np.arange(len(self.data))
        self.axes.plot(channels, self.data, drawstyle="steps-mid")
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.canvas.draw()
