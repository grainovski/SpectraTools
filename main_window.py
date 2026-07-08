import os

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from histogram_io import ParseError, load_histogram
from settings import Settings

ZOOM_FACTOR = 1.5


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

        self.settings = Settings()

        self._build_menu()
        self._update_recent_menu()

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._build_zoom_buttons()

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Recent Files")

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
            self, "Open Histogram", self.settings.last_folder(), "Text files (*.txt);;All files (*)"
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
        self.settings.set_last_folder(os.path.dirname(path))
        self.settings.add_recent_file(path)
        self._update_recent_menu()
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

    def _update_recent_menu(self):
        self.recent_menu.clear()
        for path in self.settings.recent_files():
            action = QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self.recent_menu.addAction(action)

    def _open_recent(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File not found", f"{path} no longer exists.")
            self.settings.remove_recent_file(path)
            self._update_recent_menu()
            return
        self._load_file(path)

    def _build_zoom_buttons(self):
        self.nav_toolbar.addSeparator()

        zoom_in_action = QAction("Zoom In X", self)
        zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out X", self)
        zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(zoom_out_action)

    def _on_scroll(self, event):
        if self.data is None or event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)

    def _zoom_x(self, factor, center=None):
        if self.data is None:
            return
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = (xlim[1] - xlim[0]) / 2 * factor
        new_xlim = (center - half_width, center + half_width)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()

    def _autoscale_y(self, xlim):
        lo = max(0, int(np.floor(xlim[0])))
        hi = min(len(self.data), int(np.ceil(xlim[1])) + 1)
        if lo >= hi:
            return
        visible = self.data[lo:hi]
        y_min = float(np.min(visible))
        y_max = float(np.max(visible))
        margin = (y_max - y_min) * 0.05 or 1.0
        self.axes.set_ylim(y_min - margin, y_max + margin)
