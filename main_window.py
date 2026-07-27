import os
import time

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
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

from calibration import Calibration
from calibration_dialog import CalibrationDialog
from fit_mode import FitModeController
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spectrum import LoadedSpectrum, next_color
from spk_io import load_spk
from theme import qt_stylesheet, style_axes

ZOOM_FACTOR = 1.5
_ICON_SIZE = 24


def _magnifier_icon(sign, dark=False):
    # A filled silhouette (ring + handle, both solid-filled, no stroked
    # outlines) rather than pen-drawn lines -- matplotlib's own toolbar
    # icons (Home/Pan/Save/...) are all flat filled shapes, so matching
    # that construction is what actually makes these icons "look like"
    # the built-in ones side by side, not just sharing a color.
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("white") if dark else QColor("black")

    lens_rect = QRectF(2.5, 2.5, 12.0, 12.0)
    ring = 2.2
    cx, cy = lens_rect.center().x(), lens_rect.center().y()
    bar_len, bar_w = 6.0, 1.8

    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    path.addEllipse(lens_rect)
    path.addEllipse(lens_rect.adjusted(ring, ring, -ring, -ring))
    path.addRect(QRectF(cx - bar_len / 2, cy - bar_w / 2, bar_len, bar_w))
    if sign == "+":
        path.addRect(QRectF(cx - bar_w / 2, cy - bar_len / 2, bar_w, bar_len))
    painter.fillPath(path, color)

    # Handle: a short filled bar rotated off the lens's lower-right edge.
    painter.save()
    painter.translate(lens_rect.right() - 1.0, lens_rect.bottom() - 1.0)
    painter.rotate(45)
    painter.fillRect(QRectF(0, -1.1, 8.0, 2.2), color)
    painter.restore()

    painter.end()
    return QIcon(pixmap)


def _full_spectrum_icon(dark=False):
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("white") if dark else QColor("black"))
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


class _TrimmedNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib's default toolbar (Home/Back/Forward/Pan/Zoom/
    Subplots/Customize/Save) with the items this app doesn't use
    dropped: Back/Forward (this app's own zoom already keeps its own
    history via push_current(), and duplicate view-history controls are
    confusing), Zoom-to-rectangle/Subplots/Customize (box-zoom, subplot
    layout, and curve/image styling aren't relevant to a fixed
    single-axes spectrum view). Leaves Home, Pan, and Save -- this
    app's own zoom-in/zoom-out/full-spectrum actions are then appended
    right after Save (see MainWindow._build_zoom_buttons)."""

    toolitems = [
        item for item in NavigationToolbar2QT.toolitems
        if item[0] not in ("Back", "Forward", "Zoom", "Subplots", "Customize")
    ]


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
        # coordinates=False: matplotlib's built-in toolbar otherwise adds
        # an expanding mouse-position readout label after Save, which
        # stretches to fill the toolbar and pushes anything appended
        # afterward (this app's own Zoom In X/Zoom Out X/Show Full
        # Spectrum actions) to hug the far-right edge instead of sitting
        # next to Save. This app already shows the mouse position in its
        # own status bar (_on_mouse_move), making that label redundant.
        self.nav_toolbar = _TrimmedNavigationToolbar(self.canvas, self, coordinates=False)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self.settings = Settings()
        self._theme = self.settings.theme()
        self._apply_theme(self._theme)

        self._calibration = None  # Calibration | None -- last-set coefficients, persist across on/off toggles
        self._calibration_active = False

        self._build_spectrum_panel()
        self._build_menu()
        self._update_recent_menu()

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._build_zoom_buttons()

        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.fit_controller.build_parameters_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._update_fit_mode_availability()

    def showEvent(self, event):
        super().showEvent(event)
        # b/r/p marking depends entirely on the canvas holding keyboard
        # focus (see FitModeController.eventFilter). The Fit Parameters
        # dock is visible from app startup and contains focusable
        # checkboxes/a table, so on some window managers the initial
        # focus assignment on first show can land there instead of the
        # canvas -- explicitly reclaim it every time the window is shown
        # rather than relying solely on figure_enter_event (mouse-move
        # into the plot), which only fires after the user has already
        # moved the cursor there.
        self.canvas.setFocus()

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

        view_menu.addSeparator()
        self.dark_theme_action = QAction("Dark theme", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(self._theme == "dark")
        self.dark_theme_action.toggled.connect(self._on_theme_toggled)
        view_menu.addAction(self.dark_theme_action)

        view_menu.addSeparator()
        self.calibration_action = QAction("Calibration...", self)
        self.calibration_action.triggered.connect(self._open_calibration_dialog)
        view_menu.addAction(self.calibration_action)

    def _apply_theme(self, theme):
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qt_stylesheet(theme))
        style_axes(self.axes, theme)
        self._theme = theme
        self._style_nav_toolbar_palette(theme)
        self._refresh_zoom_icons()
        # Re-derive each already-loaded spectrum's trace color from the
        # new theme's palette, using the index remembered at load time --
        # spectra loaded via a direct LoadedSpectrum(...) construction
        # (e.g. in tests) have no color_index and are left as-is.
        for spectrum in self.spectra:
            color_index = getattr(spectrum, "color_index", None)
            if color_index is not None:
                spectrum.color = next_color(color_index, theme)

    def channel_to_display(self, channel):
        """Converts a channel number (or numpy array of channel numbers)
        to whatever's on the x-axis right now: the same value if no
        calibration is active, or its calibrated keV equivalent if one
        is. Every place that draws an x-coordinate routes through this."""
        if not self._calibration_active or self._calibration is None:
            return channel
        return self._calibration.apply(channel)

    def display_to_channel(self, display_x):
        """Inverse of channel_to_display -- converts an x-axis
        coordinate (channel or keV, whichever is currently displayed)
        back to a channel number. Every place that reads a click/hover
        x-coordinate routes through this."""
        if not self._calibration_active or self._calibration is None:
            return display_x
        return self._calibration.invert(display_x)

    def _open_calibration_dialog(self):
        dialog = CalibrationDialog(
            self, initial=self._calibration, initially_active=self._calibration_active
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._calibration = dialog.result_calibration
            self._calibration_active = dialog.result_active
            if self.spectra:
                # preserve_view intentionally omitted (defaults to
                # False): the previous xlim was in the other unit
                # (channels vs keV) and carrying it over would show a
                # nonsensical view.
                self._plot_data()

    def _style_nav_toolbar_palette(self, theme):
        """Sets the navigation toolbar's actual QPalette -- not just this
        app's own QSS, which changes the toolbar's *paint* but leaves
        `.palette()` queries returning Qt's original light-mode colors.
        matplotlib's own icon loader (NavigationToolbar2QT._icon, via
        _IconEngine._is_dark_mode) reads exactly that palette background
        to decide whether to recolor Home/Pan/Save white for a dark
        background, so this is what makes those built-in icons adapt
        under this app's dark theme; without it they'd stay black
        (barely visible) regardless of theme."""
        palette = self.nav_toolbar.palette()
        if theme == "dark":
            from theme import DARK_PANEL, DARK_TEXT
            for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button):
                palette.setColor(role, QColor(DARK_PANEL))
            for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
                palette.setColor(role, QColor(DARK_TEXT))
        else:
            palette = QPalette()
        self.nav_toolbar.setPalette(palette)

    def _on_theme_toggled(self, checked):
        theme = "dark" if checked else "light"
        self._apply_theme(theme)
        self.settings.set_theme(theme)
        if hasattr(self, "spectrum_list"):
            self._update_spectrum_list()
        if self.spectra:
            self._plot_data(preserve_view=True)
        else:
            self.canvas.draw()

    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Histogram",
            self.settings.last_folder(),
            "Spectrum files (*.txt *.spe *.spk);;Text files (*.txt);;"
            "SPE files (*.spe);;SPK files (*.spk);;All files (*)",
        )
        if paths:
            self._load_files(paths)

    def _try_load_spectrum(self, path):
        lower = path.lower()
        if lower.endswith(".spe"):
            loader = load_spe
        elif lower.endswith(".spk"):
            loader = load_spk
        else:
            loader = load_histogram
        try:
            data = loader(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        color_index = self._next_color_index
        self._next_color_index += 1
        spectrum = LoadedSpectrum(path, data, next_color(color_index, self._theme))
        # Remembered so _apply_theme can re-derive this spectrum's color
        # from the new theme's palette without needing to reload the file
        # or renumber already-loaded spectra.
        spectrum.color_index = color_index
        return spectrum, None

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

    def _plot_data(self, preserve_view=False):
        # Captured before axes.clear() (which resets limits) -- used to
        # keep the user's current zoom when a replot is triggered by
        # something unrelated to loading/showing a spectrum, e.g.
        # committing or removing a peak fit.
        saved_xlim = self.axes.get_xlim() if preserve_view else None
        self.axes.clear()
        style_axes(self.axes, self._theme)
        visible = [s for s in self.spectra if s.visible]
        for spectrum in visible:
            channels = np.arange(len(spectrum.data))
            x = self.channel_to_display(channels)
            self.axes.plot(x, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
            self.fit_controller.draw_committed_fits(spectrum)
        self.axes.set_xlabel("Energy (keV)" if self._calibration_active else "Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        if visible:
            # Channel numbers can't be negative; override Matplotlib's
            # default 5% autoscale margin, which would otherwise show them
            # as such. Span the widest currently-visible spectrum, since
            # loaded files can have different channel counts.
            max_channel = max(len(s.data) for s in visible) - 1
            if saved_xlim is not None:
                xlim = saved_xlim
            else:
                xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
            self.axes.set_xlim(xlim)
            self._autoscale_y(xlim)
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path, so
        # without this the Home/Back/Forward buttons wouldn't know about
        # this view. Reset the navigation history and record this full view
        # as the new "home" baseline.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
        self._update_fit_mode_availability()
        self.fit_controller.update_results_list()

    def _on_mouse_move(self, event):
        if time.monotonic() < self.fit_controller._status_message_until:
            return
        visible = [s for s in self.spectra if s.visible]
        if not visible or event.inaxes != self.axes or event.xdata is None:
            self.statusBar().clearMessage()
            return
        channel = int(round(self.display_to_channel(event.xdata)))
        if self._calibration_active:
            parts = [f"Channel: {channel}  Energy: {event.xdata:.2f} keV"]
        else:
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
        self._update_fit_mode_availability()
        self.fit_controller.update_results_list()

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
        dark = self._theme == "dark"

        self.zoom_in_action = QAction(_magnifier_icon("+", dark), "Zoom In X", self)
        self.zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(_magnifier_icon("-", dark), "Zoom Out X", self)
        self.zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_out_action)

        self.full_spectrum_action = QAction(_full_spectrum_icon(dark), "Show Full Spectrum", self)
        self.full_spectrum_action.triggered.connect(self._show_full_spectrum)
        self.nav_toolbar.addAction(self.full_spectrum_action)

    def _refresh_zoom_icons(self):
        """Re-renders this app's own zoom/full-spectrum toolbar icons for
        the current theme -- called after _build_zoom_buttons has run;
        a no-op before that (e.g. the initial _apply_theme call in
        __init__, which runs before _build_zoom_buttons), since those
        icons are already built with the correct initial color."""
        if not hasattr(self, "zoom_in_action"):
            return
        dark = self._theme == "dark"
        self.zoom_in_action.setIcon(_magnifier_icon("+", dark))
        self.zoom_out_action.setIcon(_magnifier_icon("-", dark))
        self.full_spectrum_action.setIcon(_full_spectrum_icon(dark))

    def _build_fit_mode_buttons(self):
        self.fit_button = QAction("Fit", self)
        self.fit_button.setShortcut("Ctrl+F")
        self.fit_button.setEnabled(False)
        self.fit_button.triggered.connect(self.fit_controller.run_fit)
        self.addAction(self.fit_button)

        self.clear_fit_button = QAction("Clear", self)
        self.clear_fit_button.setShortcut("Ctrl+C")
        self.clear_fit_button.triggered.connect(self.fit_controller.clear)
        self.addAction(self.clear_fit_button)

        self.integrate_button = QAction("Integrate", self)
        self.integrate_button.setShortcut("Ctrl+I")
        self.integrate_button.setEnabled(False)
        self.integrate_button.triggered.connect(self.fit_controller.run_integration)
        self.addAction(self.integrate_button)

    def _on_canvas_click(self, event):
        self.fit_controller.on_click(event)

    def _update_fit_mode_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        available = active is not None and active.visible
        self.fit_button.setEnabled(available and self.fit_controller.state.ready_to_fit())
        self.integrate_button.setEnabled(available and self.fit_controller.state.ready_to_integrate())

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
        display_lo = self.channel_to_display(0)
        display_hi = self.channel_to_display(max_channel)
        display_lo, display_hi = min(display_lo, display_hi), max(display_lo, display_hi)
        # Clamp to the valid displayed range -- zooming/scrolling must
        # never show channels outside the data, whether the axis is
        # currently in raw channels or calibrated keV.
        new_lo = max(display_lo, center - half_width)
        new_hi = min(display_hi, center + half_width)
        if new_hi <= new_lo:
            new_hi = min(display_hi, new_lo + 1)
        new_xlim = (new_lo, new_hi)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _show_full_spectrum(self):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        max_channel = max(len(s.data) for s in visible) - 1
        full_xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _autoscale_y(self, xlim):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        channel_lo = self.display_to_channel(xlim[0])
        channel_hi = self.display_to_channel(xlim[1])
        channel_lo, channel_hi = min(channel_lo, channel_hi), max(channel_lo, channel_hi)
        lo_bound = max(0, int(np.floor(channel_lo)))
        hi_bound = int(np.ceil(channel_hi)) + 1
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
