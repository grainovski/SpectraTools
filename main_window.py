import os

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from histogram_io import ParseError, load_histogram
from settings import Settings
from spectrum import LoadedSpectrum, next_color

ZOOM_FACTOR = 1.5
_ICON_SIZE = 24


def _magnifier_icon(sign):
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(Qt.GlobalColor.black)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.drawEllipse(3, 3, 12, 12)
    painter.drawLine(13, 13, 20, 20)
    painter.drawLine(6, 9, 12, 9)
    if sign == "+":
        painter.drawLine(9, 6, 9, 12)
    painter.end()
    return QIcon(pixmap)


def _full_spectrum_icon():
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(Qt.GlobalColor.black)
    painter.setPen(Qt.PenStyle.NoPen)
    heights = (6, 14, 9, 18, 11)
    bar_width = 3
    gap = 1
    base_y = 21
    x = 2
    for height in heights:
        painter.drawRect(x, base_y - height, bar_width, height)
        x += bar_width + gap
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Viewer")
        self.resize(900, 600)

        self.spectra = []

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
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Histogram", self.settings.last_folder(), "Text files (*.txt);;All files (*)"
        )
        if paths:
            self._load_files(paths)

    def _try_load_spectrum(self, path):
        try:
            data = load_histogram(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        color = next_color(len(self.spectra))
        return LoadedSpectrum(path, data, color), None

    def _load_files(self, paths):
        failures = []
        loaded_any = False
        for path in paths:
            if any(s.path == path for s in self.spectra):
                continue
            spectrum, error = self._try_load_spectrum(path)
            if error:
                failures.append(error)
                continue
            self.spectra.append(spectrum)
            self.settings.set_last_folder(os.path.dirname(path))
            self.settings.add_recent_file(path)
            loaded_any = True

        if loaded_any:
            self._update_recent_menu()
            self._plot_data()

        if failures:
            QMessageBox.warning(self, "Some files could not be loaded", "\n".join(failures))

    def _plot_data(self):
        self.axes.clear()
        channels = np.arange(len(self.data))
        self.axes.plot(channels, self.data, drawstyle="steps-mid")
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        # Channel numbers can't be negative; override Matplotlib's default
        # 5% autoscale margin, which would otherwise show them as such.
        self.axes.set_xlim(0, len(self.data) - 1)
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path, so
        # without this the Home/Back/Forward buttons wouldn't know about
        # this view. Reset the navigation history and record this full view
        # as the new "home" baseline.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()

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
        self._load_files([path])

    def _build_zoom_buttons(self):
        self.nav_toolbar.addSeparator()

        zoom_in_action = QAction(_magnifier_icon("+"), "Zoom In X", self)
        zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction(_magnifier_icon("-"), "Zoom Out X", self)
        zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(zoom_out_action)

        full_spectrum_action = QAction(_full_spectrum_icon(), "Show Full Spectrum", self)
        full_spectrum_action.triggered.connect(self._show_full_spectrum)
        self.nav_toolbar.addAction(full_spectrum_action)

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
        max_channel = len(self.data) - 1
        # Clamp to valid channel numbers -- zooming/scrolling must never
        # show negative channels or channels past the end of the data.
        new_lo = max(0.0, center - half_width)
        new_hi = min(float(max_channel), center + half_width)
        if new_hi <= new_lo:
            new_hi = min(float(max_channel), new_lo + 1)
        new_xlim = (new_lo, new_hi)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _show_full_spectrum(self):
        if self.data is None:
            return
        full_xlim = (0, len(self.data) - 1)
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _autoscale_y(self, xlim):
        lo = max(0, int(np.floor(xlim[0])))
        hi = min(len(self.data), int(np.ceil(xlim[1])) + 1)
        if lo >= hi:
            return
        visible = self.data[lo:hi]
        y_min = float(np.min(visible))
        y_max = float(np.max(visible))
        if self.log_scale_action.isChecked():
            # A log-scaled axis silently ignores set_ylim() with a
            # non-positive bound, so a plain "y_min - margin" (common
            # here since spectra often have zero-count channels) would
            # leave the Y-axis un-rescaled. Floor to a small positive
            # value and use a multiplicative margin instead.
            y_min = max(y_min, 1.0)
            y_max = max(y_max, y_min * 1.1)
            self.axes.set_ylim(y_min / 1.1, y_max * 1.1)
        else:
            margin = (y_max - y_min) * 0.05 or 1.0
            self.axes.set_ylim(y_min - margin, y_max + margin)
