import os

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QRadioButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
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


def _color_swatch_pixmap(color):
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(color))
    return pixmap


_APP_ICON_SIZES = (16, 32, 48, 128, 256)
_APP_ICON_BAR_HEIGHTS = (0.35, 0.7, 0.5, 0.9, 0.6, 0.8, 0.45)


def _app_icon_pixmap(size):
    # Bars colored from the app's own spectrum color cycle -- the icon
    # doubles as a visual reminder of how loaded spectra are distinguished
    # from one another in the app itself.
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    margin = size * 0.1
    gap = size * 0.03
    n = len(_APP_ICON_BAR_HEIGHTS)
    bar_width = (size - 2 * margin - (n - 1) * gap) / n
    base_y = size - margin
    radius = bar_width * 0.15
    for i, fraction in enumerate(_APP_ICON_BAR_HEIGHTS):
        painter.setBrush(QColor(next_color(i)))
        height = fraction * (size - 2 * margin)
        x = margin + i * (bar_width + gap)
        painter.drawRoundedRect(x, base_y - height, bar_width, height, radius, radius)
    painter.end()
    return pixmap


def _app_icon():
    icon = QIcon()
    for size in _APP_ICON_SIZES:
        icon.addPixmap(_app_icon_pixmap(size))
    return icon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraTools")
        self.setWindowIcon(_app_icon())
        self.resize(900, 600)

        self.spectra = []
        # Monotonically increasing, never reused -- unlike len(self.spectra),
        # this can't collide with a still-loaded spectrum's color after one
        # is removed (there's no plot legend, so color is the only way to
        # tell traces apart; a collision would make two spectra
        # indistinguishable on the plot).
        self._next_color_index = 0

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

        self._build_spectrum_panel()
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

        view_menu.addAction(self.toggle_spectrum_panel_action)

    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Histogram",
            self.settings.last_folder(),
            "Spectrum files (*.txt *.spe);;Text files (*.txt);;SPE files (*.spe);;All files (*)",
        )
        if paths:
            self._load_files(paths)

    def _try_load_spectrum(self, path):
        loader = load_spe if path.lower().endswith(".spe") else load_histogram
        try:
            data = loader(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        color = next_color(self._next_color_index)
        self._next_color_index += 1
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
            if not self.spectra:
                spectrum.active = True
            self.spectra.append(spectrum)
            self.settings.set_last_folder(os.path.dirname(path))
            self.settings.add_recent_file(path)
            loaded_any = True

        if loaded_any:
            self._update_recent_menu()
            self._update_spectrum_list()
            self._plot_data()

        if failures:
            QMessageBox.warning(self, "Some files could not be loaded", "\n".join(failures))

    def _plot_data(self):
        self.axes.clear()
        visible = [s for s in self.spectra if s.visible]
        for spectrum in visible:
            channels = np.arange(len(spectrum.data))
            self.axes.plot(channels, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        if visible:
            # Channel numbers can't be negative; override Matplotlib's
            # default 5% autoscale margin, which would otherwise show them
            # as such. Span the widest currently-visible spectrum, since
            # loaded files can have different channel counts.
            max_channel = max(len(s.data) for s in visible) - 1
            self.axes.set_xlim(0, max_channel)
            self._autoscale_y((0, max_channel))
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path, so
        # without this the Home/Back/Forward buttons wouldn't know about
        # this view. Reset the navigation history and record this full view
        # as the new "home" baseline.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()

    def _on_mouse_move(self, event):
        visible = [s for s in self.spectra if s.visible]
        if not visible or event.inaxes != self.axes or event.xdata is None:
            self.statusBar().clearMessage()
            return
        channel = int(round(event.xdata))
        parts = [f"Channel: {channel}"]
        for spectrum in visible:
            if 0 <= channel < len(spectrum.data):
                parts.append(f"{os.path.basename(spectrum.path)}: {spectrum.data[channel]}")
            else:
                parts.append(f"{os.path.basename(spectrum.path)}: -")
        self.statusBar().showMessage("  |  ".join(parts))

    def _on_log_scale_toggled(self, checked):
        if self.spectra:
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

    def _build_spectrum_panel(self):
        self.spectrum_list = QListWidget()
        self.spectrum_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.spectrum_list.customContextMenuRequested.connect(self._on_spectrum_context_menu)
        self.active_button_group = QButtonGroup(self)

        self.spectrum_dock = QDockWidget("Loaded Spectra", self)
        self.spectrum_dock.setWidget(self.spectrum_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.spectrum_dock)

        self.toggle_spectrum_panel_action = QAction("Spectra", self)
        self.toggle_spectrum_panel_action.setCheckable(True)
        self.toggle_spectrum_panel_action.setToolTip("Show/hide the loaded spectra list")
        self.toggle_spectrum_panel_action.toggled.connect(self.spectrum_dock.setVisible)
        self.spectrum_dock.visibilityChanged.connect(self.toggle_spectrum_panel_action.setChecked)
        # Starts closed -- the dock is only ever a click away via the
        # always-visible tab below, never left open by default.
        self.spectrum_dock.setVisible(False)

        spectrum_tab_bar = QToolBar("Spectra Tab", self)
        spectrum_tab_bar.setMovable(False)
        spectrum_tab_bar.setFloatable(False)
        spectrum_tab_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        spectrum_tab_bar.addAction(self.toggle_spectrum_panel_action)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, spectrum_tab_bar)

    def _update_spectrum_list(self):
        self.spectrum_list.clear()
        # Recreated each rebuild -- the old group (and its buttons) are
        # discarded along with the list items they belonged to.
        self.active_button_group = QButtonGroup(self)

        for spectrum in self.spectra:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, spectrum.path)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(4, 2, 4, 2)

            show_checkbox = QCheckBox()
            show_checkbox.setToolTip("Show")
            show_checkbox.setChecked(spectrum.visible)
            show_checkbox.toggled.connect(
                lambda checked, p=spectrum.path: self._on_show_toggled(p, checked)
            )
            row_layout.addWidget(show_checkbox)

            active_radio = QRadioButton()
            active_radio.setToolTip("Active (for future fitting/peak-finding operations)")
            active_radio.setChecked(spectrum.active)
            active_radio.toggled.connect(
                lambda checked, p=spectrum.path: self._on_active_toggled(p, checked)
            )
            self.active_button_group.addButton(active_radio)
            row_layout.addWidget(active_radio)

            swatch = QLabel()
            swatch.setPixmap(_color_swatch_pixmap(spectrum.color))
            row_layout.addWidget(swatch)

            row_layout.addWidget(QLabel(os.path.basename(spectrum.path)))
            row_layout.addStretch()

            item.setSizeHint(row.sizeHint())
            self.spectrum_list.addItem(item)
            self.spectrum_list.setItemWidget(item, row)

    def _on_show_toggled(self, path, checked):
        for spectrum in self.spectra:
            if spectrum.path == path:
                spectrum.visible = checked
                break
        self._plot_data()

    def _on_active_toggled(self, path, checked):
        if not checked:
            return
        for spectrum in self.spectra:
            spectrum.active = spectrum.path == path

    def _on_spectrum_context_menu(self, position):
        item = self.spectrum_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove")
        chosen = menu.exec(self.spectrum_list.viewport().mapToGlobal(position))
        if chosen == remove_action:
            path = item.data(Qt.ItemDataRole.UserRole)
            removed_was_active = any(s.path == path and s.active for s in self.spectra)
            self.spectra = [s for s in self.spectra if s.path != path]
            if removed_was_active and self.spectra:
                self.spectra[0].active = True
            self._update_spectrum_list()
            self._plot_data()

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
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)

    def _zoom_x(self, factor, center=None):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = (xlim[1] - xlim[0]) / 2 * factor
        max_channel = max(len(s.data) for s in visible) - 1
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
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        full_xlim = (0, max(len(s.data) for s in visible) - 1)
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _autoscale_y(self, xlim):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        lo_bound = max(0, int(np.floor(xlim[0])))
        hi_bound = int(np.ceil(xlim[1])) + 1
        slices = []
        for spectrum in visible:
            lo = min(lo_bound, len(spectrum.data))
            hi = min(hi_bound, len(spectrum.data))
            if lo < hi:
                slices.append(spectrum.data[lo:hi])
        if not slices:
            return
        combined = np.concatenate(slices)
        y_min = float(np.min(combined))
        y_max = float(np.max(combined))
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
