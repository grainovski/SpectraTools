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

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

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

        view_menu = self.menuBar().addMenu("&View")
        self.log_scale_action = QAction("Log scale Y", self)
        self.log_scale_action.setCheckable(True)
        self.log_scale_action.toggled.connect(self._on_log_scale_toggled)
        view_menu.addAction(self.log_scale_action)

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
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        self.canvas.draw()

    def _on_mouse_move(self, event):
        if self.data is None or event.inaxes != self.axes or event.xdata is None:
            self.statusBar().clearMessage()
            return
        channel = int(round(event.xdata))
        if 0 <= channel < len(self.data):
            self.statusBar().showMessage(f"Channel: {channel}  Counts: {self.data[channel]}")
        else:
            self.statusBar().clearMessage()

    def _on_log_scale_toggled(self, checked):
        if self.data is not None:
            self._plot_data()
