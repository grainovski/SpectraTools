import math
import os
import time
from datetime import datetime

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QRectF, QThread, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QProgressDialog,
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

from calibration import turning_point
from calibration_plot_dialog import CalibrationPlotDialog
from calibration_view import (
    CalibrationViewMixin,
    wheel_zoom_factor,
    zoomed_limits,
)
from goto_view import GoToMixin
from combine_dialog import CombineDialog
from energy_assignments import EnergyAssignments
from factor_dialog import FactorDialog
from fit_mode import FitModeController
from help_content import (
    build_about_html,
    build_howto_html,
    build_knowledge_database_html,
    open_help_page,
)
import fit_persist
from histogram_io import ParseError, load_histogram, save_histogram
from matrix_panel import MatrixPanel
import matrix_cache
from mtx_io import load_mtx
from lzs_io import load_lzs
from n42_io import load_n42
from peak_fit import FitError, channel_indices
from settings import Settings
from spe_io import load_spe, save_spe
from spectrum import (
    LoadedSpectrum, active_spectrum, next_color, pan_button_is_active, panned_xlim,
)
from spectrum_operations import (
    add, combined_variance, multiply, normalize_factors, rebin, rebinned_variance,
    reference_value, scaled_variance, subtract,
)
from spk_io import load_spk, save_spk
from theme import qt_stylesheet, refresh_builtin_toolbar_icons, style_axes, style_nav_toolbar_palette

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


def _calibration_icon(dark=False):
    # A diagonal ruler with notched tick marks -- evokes calibration/
    # measurement, visually distinct from the magnifying-glass zoom
    # icons and the full-spectrum bar chart already in this toolbar.
    # Ticks are cut as transparent notches out of the solid ruler body
    # via OddEvenFill (the same technique _magnifier_icon uses for its
    # ring), which reads correctly on any toolbar background.
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("white") if dark else QColor("black")

    painter.translate(_ICON_SIZE / 2, _ICON_SIZE / 2)
    painter.rotate(-40)

    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    path.addRect(QRectF(-9.5, -3.0, 19.0, 6.0))
    for x in (-6.5, -2.5, 1.5, 5.5):
        path.addRect(QRectF(x, -3.0, 1.2, 3.0))
    painter.fillPath(path, color)

    painter.end()
    return QIcon(pixmap)


def _calibration_active_icon(active, dark=False):
    # A toggle-switch glyph -- outlined pill track with a filled knob on
    # the right when active, left when inactive. Two distinct pixmaps
    # (not a single icon relying on Qt's checked-state highlight, which
    # this project has found unreliable across themes -- see the
    # matplotlib/Qt toolbar gotchas already worked around for the
    # built-in icons) so on/off stays unambiguous regardless of theme.
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("white") if dark else QColor("black")

    track = QRectF(2.0, 8.0, 20.0, 8.0)
    radius = track.height() / 2
    # Stroked outline rather than this file's usual flat-filled
    # silhouette (see _magnifier_icon) -- a filled pill would read as a
    # solid bar, not a switch; the outline is what makes the track
    # legible as a track for the knob below to sit in.
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(track, radius, radius)

    knob_d = 6.0
    knob_x = track.right() - knob_d - 1.2 if active else track.left() + 1.2
    knob_y = track.center().y() - knob_d / 2
    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QRectF(knob_x, knob_y, knob_d, knob_d))

    painter.end()
    return QIcon(pixmap)


def _color_swatch_pixmap(color):
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(color))
    return pixmap


_APP_ICON_SIZES = (16, 32, 48, 128, 256)
_APP_ICON_BAR_HEIGHTS = (0.35, 0.7, 0.5, 0.9, 0.6, 0.8, 0.45)
# The icon's own palette. It used to read the light spectrum cycle, but
# that became a red-to-blue ramp in 5.2.6 and the icon has a different job:
# it is the application's mark, on the taskbar and in the installer, and
# should not shift when the way traces are coloured changes. These are the
# ten values the light cycle held before that change, so the icon renders
# exactly as it always has.
_APP_ICON_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _app_icon_pixmap(size):
    # Bars coloured from the icon's own palette, _APP_ICON_COLORS.
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
        painter.setBrush(QColor(_APP_ICON_COLORS[i % len(_APP_ICON_COLORS)]))
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


class _MatrixLoadWorker(QThread):
    """Decodes a matrix off the GUI thread.

    The decode is several seconds of pure-Python CPU that cannot be
    vectorised (the tag stream's variable-length encoding forces a
    sequential scan -- see lc_codec.decode_row). Run on the GUI thread it
    froze the whole window: no repaint, no menu, an unresponsive title
    bar, which is what made a merely-slow operation feel broken.

    Only decodes and reports; every widget touched in response lives in
    the GUI thread's slots. The decoded numpy array is handed over via
    the `loaded` signal -- safe to pass between threads, since ownership
    moves with it and this thread never looks at it again.
    """

    progressed = Signal(int, int)
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        # A cache hit skips the decode entirely, which is the whole cost
        # of opening a matrix -- about 4.9 s for the reference file. Read
        # on the worker thread, not before starting it, so a slow or
        # stalled disk cannot freeze the GUI where the decode no longer
        # does. No progress is reported for a hit: it is a single array
        # read, and a bar that jumps straight to done says less than no
        # bar at all.
        cached = matrix_cache.load(self._path)
        if cached is not None:
            self.loaded.emit(cached)
            return
        try:
            matrix = load_mtx(self._path, progress=self.progressed.emit)
        except ParseError as exc:
            # ParseError already carries a user-facing message, and
            # load_mtx converts OSError into one too, so this is the
            # single failure channel. Caught here rather than left to
            # propagate because an exception escaping run() would take
            # down the thread with no way for the GUI to report it.
            self.failed.emit(str(exc))
            return
        # Stored after a successful decode only, so a matrix that failed
        # to parse is never cached as if it had worked. store() reports
        # failure rather than raising -- a cache that cannot be written is
        # not something to interrupt the user for.
        matrix_cache.store(self._path, matrix)
        self.loaded.emit(matrix)


class MainWindow(CalibrationViewMixin, GoToMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraTools")
        self.setWindowIcon(_app_icon())
        self.resize(900, 600)

        self.spectra = []
        self._matrix_panels = []  # currently-open MatrixPanel windows -- see _on_activated
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

        self._pan_last_px = None
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_pan_press)
        self.canvas.mpl_connect("button_release_event", self._on_pan_release)

        self._build_zoom_buttons()
        self._build_calibration_toolbar_buttons()

        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.fit_controller.build_parameters_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._update_fit_mode_availability()
        self._update_operations_availability()

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

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowActivate:
            self._on_activated()

    def closeEvent(self, event):
        # MatrixPanel windows are parentless top-level windows (deliberately,
        # so they behave independently rather than as Qt-modal children --
        # see matrix_panel.py). That means Qt's quitOnLastWindowClosed
        # doesn't quit the app just because MainWindow closes: any matrix
        # panel left open keeps counting as a visible top-level window, so
        # app.exec() would never return and the process would linger with
        # no visible explanation. list(...) copies before iterating because
        # each panel.close() synchronously mutates self._matrix_panels via
        # its own closeEvent (MatrixPanel.closeEvent removes itself).
        for panel in list(self._matrix_panels):
            panel.close()
        super().closeEvent(event)

    def _build_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        self.file_menu.addAction(open_action)

        self.open_matrix_action = QAction("Open Matrix...", self)
        self.open_matrix_action.setShortcut("Ctrl+Shift+O")
        self.open_matrix_action.triggered.connect(self._open_matrix_dialog)
        self.file_menu.addAction(self.open_matrix_action)

        # Separate from "Open..." because a ROOT file needs a second
        # question -- which of its objects -- that no other format does.
        self.open_root_action = QAction("Open ROOT File...", self)
        self.open_root_action.triggered.connect(self._open_root_dialog)
        self.file_menu.addAction(self.open_root_action)

        self.save_fits_action = QAction("Save Fits...", self)
        self.save_fits_action.setEnabled(False)
        self.save_fits_action.triggered.connect(self._save_fits_dialog)
        self.file_menu.addAction(self.save_fits_action)

        self.load_fits_action = QAction("Load Fits...", self)
        self.load_fits_action.setEnabled(False)
        self.load_fits_action.triggered.connect(self._load_fits_dialog)
        self.file_menu.addAction(self.load_fits_action)

        self.reload_spectrum_action = QAction("Reload Spectrum", self)
        self.reload_spectrum_action.setShortcut("F5")
        self.reload_spectrum_action.setEnabled(False)
        self.reload_spectrum_action.setToolTip(
            "Re-read the active spectrum's file, keeping its fits and marks"
        )
        self.reload_spectrum_action.triggered.connect(self._reload_active_spectrum)
        self.file_menu.addAction(self.reload_spectrum_action)

        self.save_spectrum_action = QAction("Save Spectrum...", self)
        self.save_spectrum_action.setShortcut("Ctrl+S")
        self.save_spectrum_action.setEnabled(False)
        self.save_spectrum_action.triggered.connect(self._open_save_spectrum_dialog)
        self.file_menu.addAction(self.save_spectrum_action)

        self.close_spectrum_action = QAction("Close Spectrum", self)
        self.close_spectrum_action.setShortcut("Ctrl+W")
        self.close_spectrum_action.setEnabled(False)
        self.close_spectrum_action.triggered.connect(self._close_active_spectrum)
        self.file_menu.addAction(self.close_spectrum_action)

        self.recent_menu = self.file_menu.addMenu("Recent Files")

        self.file_menu.addSeparator()

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("&View")
        self.goto_action = QAction("Go To...", self)
        # Ctrl+G is what every editor and browser uses for "go to", and it
        # was worth reclaiming: Log scale Y moved to Ctrl+Y, which is the
        # better mnemonic for it anyway (it scales the Y axis).
        self.goto_action.setShortcut("Ctrl+G")
        self.goto_action.triggered.connect(self.open_goto_dialog)
        view_menu.addAction(self.goto_action)

        self.log_scale_action = QAction("Log scale Y", self)
        self.log_scale_action.setShortcut("Ctrl+Y")
        self.log_scale_action.setCheckable(True)
        self.log_scale_action.toggled.connect(self._on_log_scale_toggled)
        view_menu.addAction(self.log_scale_action)

        self.toggle_spectrum_panel_action.setShortcut("Ctrl+1")
        view_menu.addAction(self.toggle_spectrum_panel_action)

        view_menu.addSeparator()
        self.dark_theme_action = QAction("Dark theme", self)
        self.dark_theme_action.setShortcut("Ctrl+D")
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(self._theme == "dark")
        self.dark_theme_action.toggled.connect(self._on_theme_toggled)
        view_menu.addAction(self.dark_theme_action)

        self.operations_menu = self.menuBar().addMenu("&Operations")
        self.calibration_action = QAction("Calibration...", self)
        self.calibration_action.setShortcut("Ctrl+L")
        self.calibration_action.triggered.connect(self._open_calibration_dialog)
        self.operations_menu.addAction(self.calibration_action)

        self.calibration_active_menu_action = QAction("Toggle Calibration Active", self)
        self.calibration_active_menu_action.setShortcut("Ctrl+T")
        self.calibration_active_menu_action.setCheckable(True)
        self.calibration_active_menu_action.setEnabled(self._calibration is not None)
        self.calibration_active_menu_action.toggled.connect(self._on_calibration_toggle_action)
        self.operations_menu.addAction(self.calibration_active_menu_action)

        # After the two existing calibration entries rather than between
        # them: those two are the primitives (set the coefficients, turn
        # them on) and belong together, while this one is a way of
        # arriving at coefficients.
        self.calibrate_from_peaks_action = QAction("Calibrate from Fitted Peaks...", self)
        self.calibrate_from_peaks_action.setEnabled(False)
        self.calibrate_from_peaks_action.setToolTip(
            "Assign known energies to peaks you have already fitted, and "
            "calibrate from their fitted centroids"
        )
        self.calibrate_from_peaks_action.triggered.connect(
            self._open_calibrate_from_peaks_dialog
        )
        self.operations_menu.addAction(self.calibrate_from_peaks_action)

        # The automatic route to the same dialog: it finds and fits the
        # peaks itself and identifies them against a source file, so it
        # needs a spectrum but no fits. Ctrl+Shift+L beside Calibration's
        # Ctrl+L, the way Subtract's Ctrl+Shift+A sits beside Add's Ctrl+A;
        # L is not a marking key, so the Ctrl+letter trap described in
        # fit_mode.is_bare_key_event does not arise.
        self.auto_calibrate_action = QAction("Automatic Calibration...", self)
        self.auto_calibrate_action.setShortcut("Ctrl+Shift+L")
        self.auto_calibrate_action.setEnabled(False)
        self.auto_calibrate_action.setToolTip(
            "Find and fit the peaks of the active spectrum, identify them "
            "against a .sou source file, and calibrate from them -- no "
            "calibration needed to start"
        )
        self.auto_calibrate_action.triggered.connect(self._open_auto_calibrate_dialog)
        self.operations_menu.addAction(self.auto_calibrate_action)

        self.operations_menu.addSeparator()

        self.multiply_action = QAction("Multiply by Factor...", self)
        self.multiply_action.setShortcut("Ctrl+M")
        self.multiply_action.setEnabled(False)
        self.multiply_action.triggered.connect(self._open_multiply_dialog)
        self.operations_menu.addAction(self.multiply_action)

        self.rebin_action = QAction("Rebin by Factor...", self)
        self.rebin_action.setShortcut("Ctrl+R")
        self.rebin_action.setEnabled(False)
        self.rebin_action.triggered.connect(self._open_rebin_dialog)
        self.operations_menu.addAction(self.rebin_action)

        self.normalize_action = QAction("Normalize Spectra", self)
        self.normalize_action.setShortcut("Ctrl+N")
        self.normalize_action.setEnabled(False)
        self.normalize_action.triggered.connect(self._normalize_spectra)
        self.operations_menu.addAction(self.normalize_action)

        self.add_action = QAction("Add Spectra...", self)
        self.add_action.setShortcut("Ctrl+A")
        self.add_action.setEnabled(False)
        self.add_action.triggered.connect(self._open_add_dialog)
        self.operations_menu.addAction(self.add_action)

        self.subtract_action = QAction("Subtract Spectra...", self)
        self.subtract_action.setShortcut("Ctrl+Shift+A")
        self.subtract_action.setEnabled(False)
        self.subtract_action.triggered.connect(self._open_subtract_dialog)
        self.operations_menu.addAction(self.subtract_action)

        self.help_menu = self.menuBar().addMenu("&Help")

        self.howto_action = QAction("HowTo", self)
        self.howto_action.setShortcut("F1")
        self.howto_action.triggered.connect(self._open_howto)
        self.help_menu.addAction(self.howto_action)

        self.knowledge_database_action = QAction("Knowledge Database", self)
        self.knowledge_database_action.triggered.connect(self._open_knowledge_database)
        self.help_menu.addAction(self.knowledge_database_action)

        self.about_action = QAction("About", self)
        self.about_action.triggered.connect(self._open_about)
        self.help_menu.addAction(self.about_action)

    def _open_howto(self):
        if not open_help_page(build_howto_html()):
            self.fit_controller._show_status_message("Couldn't open the HowTo page in your browser.", 5000)

    def _open_knowledge_database(self):
        if not open_help_page(build_knowledge_database_html()):
            self.fit_controller._show_status_message(
                "Couldn't open the Knowledge Database page in your browser.", 5000
            )

    def _open_about(self):
        if not open_help_page(build_about_html()):
            self.fit_controller._show_status_message("Couldn't open the About page in your browser.", 5000)

    def _apply_theme(self, theme):
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qt_stylesheet(theme))
        style_axes(self.axes, theme)
        self._theme = theme
        style_nav_toolbar_palette(self.nav_toolbar, theme)
        refresh_builtin_toolbar_icons(self.nav_toolbar)
        self._refresh_zoom_icons()
        self._refresh_calibration_icons()
        # Re-derive each already-loaded spectrum's trace color from the
        # new theme's palette, using the index remembered at load time --
        # spectra loaded via a direct LoadedSpectrum(...) construction
        # (e.g. in tests) have no color_index and are left as-is.
        for spectrum in self.spectra:
            color_index = getattr(spectrum, "color_index", None)
            if color_index is not None:
                spectrum.color = next_color(color_index, theme)
        # Matrix panels (and any heatmap windows opened from them) each
        # build their own separate nav_toolbar/axes, entirely independent
        # of this window's own -- push the new theme out to each one
        # explicitly, the same way _apply_calibration_change (below)
        # already loops over self._matrix_panels to share a calibration
        # change with every open panel. Without this loop, an
        # already-open panel (and any heatmap window opened from it)
        # would keep showing whatever theme was active when IT was
        # constructed and never follow a later toggle here -- confirmed
        # empirically before this fix: both the toolbar's built-in icon
        # bytes and the plot axes' facecolor stayed byte-for-byte
        # unchanged across a toggle.
        for panel in self._matrix_panels:
            panel._refresh_theme()
            for heatmap_window in panel._heatmap_windows:
                heatmap_window._refresh_theme(theme)

    def _apply_calibration_change(self, new_calibration, new_active):
        """Applies a new calibration/active state and replots, keeping
        the same detector-channel region in view across the unit change
        -- re-expresses the current view's bounds (captured in channel
        space using the OLD calibration state, below) through the NEW
        calibration, rather than resetting to the full spectrum or
        reusing the old view's raw x-values verbatim (which would show
        a nonsensical region once the axis units change). Shared by the
        calibration dialog's OK handler and the toolbar's quick Active
        toggle, so either path keeps the plot, the toolbar button, and
        the Fit Results table in sync."""
        # Captured before the overwrite below so the fold check can tell a
        # genuinely NEW calibration from the Active toggle, which passes the
        # very same object back and must not re-warn on every flip.
        previous_calibration = self._calibration
        channel_bounds = None
        if self.spectra:
            old_xlim = self.axes.get_xlim()
            channel_bounds = (
                self.display_to_channel(old_xlim[0]),
                self.display_to_channel(old_xlim[1]),
            )
        # Same reasoning as channel_bounds above, applied to every open
        # matrix panel too -- captured under the OLD calibration, before
        # it's overwritten below, so a zoomed-in panel doesn't silently
        # reset to full view just because the change came from here
        # (or from a SIBLING panel) rather than from that panel's own
        # calibration dialog.
        panel_channel_bounds = {
            panel: (
                panel.display_to_channel(panel.axes.get_xlim()[0]),
                panel.display_to_channel(panel.axes.get_xlim()[1]),
            )
            for panel in self._matrix_panels
        }
        self._calibration = new_calibration
        self._calibration_active = new_active
        self.calibration_toggle_action.setEnabled(new_calibration is not None)
        self.calibration_toggle_action.setChecked(new_active)
        self.calibration_active_menu_action.setEnabled(new_calibration is not None)
        self.calibration_active_menu_action.setChecked(new_active)
        if self.spectra:
            new_xlim = (
                self.channel_to_display(channel_bounds[0]),
                self.channel_to_display(channel_bounds[1]),
            )
            self._plot_data(xlim_override=new_xlim)
        self.fit_controller.refresh_parameters_panel_calibration()
        if (new_calibration is not None and new_active
                and new_calibration is not previous_calibration):
            self._warn_if_calibration_folds(new_calibration)
        # Calibration is shared with any open MatrixPanel (its
        # _calibration/_calibration_active are properties delegating to
        # these exact attributes, not a separate copy) -- a change made
        # here needs to visibly redraw those windows too, not just leave
        # their already-drawn plot showing the old units until something
        # else happens to trigger a redraw. Each panel's own zoom is
        # restored via panel_channel_bounds captured above, the same way
        # this window's own zoom is preserved just above -- not just
        # reset to full view.
        for panel, old_bounds in panel_channel_bounds.items():
            panel_new_xlim = (
                panel.channel_to_display(old_bounds[0]),
                panel.channel_to_display(old_bounds[1]),
            )
            panel._plot_data(xlim_override=panel_new_xlim)

    def _open_multiply_dialog(self):
        active = active_spectrum(self.spectra)
        if active is None:
            return
        dialog = FactorDialog(
            self, "Multiply by Factor", "Factor:",
            parse=float,
            validate=lambda v: None if (math.isfinite(v) and v > 0) else "Factor must be a finite number greater than zero.",
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_multiply(active, dialog.result_factor)

    def _apply_multiply(self, spectrum, factor):
        try:
            data = multiply(spectrum.data, factor)
        except ValueError as exc:
            QMessageBox.warning(self, "Multiply by Factor", f"Could not multiply: {exc}")
            return
        # Variance before counts: scaled_variance materializes the Poisson
        # assumption from the PRE-multiply counts when the spectrum carries
        # no propagated variance, so it must still be able to see them.
        # The overflow branch above returns before touching either, so the
        # two can never end up describing different data.
        spectrum.variance = scaled_variance(spectrum.variance, factor, spectrum.data)
        spectrum.data = data
        self.fit_controller.reset_marks()
        spectrum.fits.clear()
        self._plot_data(preserve_view=True)

    def _open_rebin_dialog(self):
        active = active_spectrum(self.spectra)
        if active is None:
            return
        dialog = FactorDialog(
            self, "Rebin by Factor", "Factor:",
            parse=int,
            validate=lambda v: None if v >= 2 else "Rebin factor must be an integer of at least 2.",
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_rebin(active, dialog.result_factor)

    def _apply_rebin(self, spectrum, factor):
        spectrum.data = rebin(spectrum.data, factor)
        # Grouped and summed exactly as the counts were. Without this the
        # variance kept its pre-rebin length and fit_peaks' own length check
        # then refused the spectrum outright, leaving a rebinned cut
        # impossible to fit for the rest of the session.
        spectrum.variance = rebinned_variance(spectrum.variance, factor)
        if self._calibration is not None:
            # Calibration is a single MainWindow-level object shared by
            # every loaded spectrum (see __init__), but rebinning changes
            # only THIS spectrum's channel count -- rescaling it here keeps
            # the just-rebinned spectrum's keV axis correct at the cost of
            # desyncing it for any OTHER already-loaded spectrum, which
            # still has its original channel scale. Rescaling is the right
            # default (not rescaling would immediately break the spectrum
            # that was just rebinned), so this is flagged to the user via
            # a status message rather than blocked or silently skipped.
            # fit_controller._show_status_message (not statusBar()
            # directly) -- it arms _status_message_until, which
            # _on_mouse_move checks before overwriting the status bar
            # with the ordinary hover readout. Without this, the warning
            # gets wiped by the very next mouse move over the canvas,
            # which is nearly guaranteed to happen right after rebinning.
            self._calibration = self._calibration.rescaled(factor)
            if len(self.spectra) > 1:
                self.fit_controller._show_status_message(
                    "Rebinned. Calibration was rescaled for this spectrum -- "
                    "it may no longer be correct for other loaded spectra.",
                    8000,
                )
        self.fit_controller.reset_marks()
        spectrum.fits.clear()
        self._plot_data()

    def _normalize_spectra(self):
        # Both status messages below use fit_controller._show_status_message
        # (not statusBar() directly) -- it arms _status_message_until, which
        # _on_mouse_move checks before overwriting the status bar with the
        # ordinary hover readout. Without this, either message gets wiped by
        # the very next mouse move over the canvas -- and for the first
        # message in particular, moving the mouse onto the canvas to make a
        # mark is the literal next thing the message tells the user to do.
        # (Same class of bug found and fixed for Rebin's warning in Task 10;
        # applied here proactively rather than waiting to rediscover it.)
        visible = [s for s in self.spectra if s.visible]
        if len(visible) < 2:
            return
        state = self.fit_controller.state
        if state.fit_region is not None:
            region, channel = state.fit_region, None
        elif state.pending_fit_click is not None:
            region, channel = None, state.pending_fit_click
        else:
            self.fit_controller._show_status_message(
                "Mark a channel or region (hold r, click) to normalize against.", 5000
            )
            return

        values = [reference_value(s.data, channel=channel, region=region) for s in visible]
        if all(v == 0 for v in values):
            self.fit_controller._show_status_message(
                "Nothing to normalize -- every visible spectrum reads zero at the marked channel/region.",
                5000,
            )
            self.fit_controller.reset_marks()
            self._plot_data(preserve_view=True)
            return
        factors = normalize_factors(values)

        skipped = []
        for spectrum, factor in zip(visible, factors):
            if factor is None:
                skipped.append(os.path.basename(spectrum.path))
                continue
            if factor == 1.0:
                continue
            scaled = multiply(spectrum.data, factor)
            # Normalize scales through the same multiply(), so it carries
            # the same obligation -- and the same ordering as
            # _apply_multiply: scaled_variance reads the PRE-multiply
            # counts, so it runs before the counts are replaced.
            spectrum.variance = scaled_variance(spectrum.variance, factor, spectrum.data)
            spectrum.data = scaled
            spectrum.fits.clear()

        self.fit_controller.reset_marks()
        self._plot_data(preserve_view=True)

        if skipped:
            self.fit_controller._show_status_message(
                f"Skipped (zero reference value): {', '.join(skipped)}", 5000
            )

    def _open_add_dialog(self):
        if len(self.spectra) < 2:
            return
        active = active_spectrum(self.spectra)
        dialog = CombineDialog(self, "Add Spectra", self.spectra, active)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_add(dialog.result_spectrum_a, dialog.result_spectrum_b, dialog.result_factor)

    def _apply_add(self, spectrum_a, spectrum_b, factor):
        try:
            data = add(spectrum_a.data, spectrum_b.data, factor)
        except ValueError as exc:
            QMessageBox.warning(self, "Add Spectra", f"Could not add: {exc}")
            return
        name_a = os.path.basename(spectrum_a.path)
        name_b = os.path.basename(spectrum_b.path)
        label = f"{name_a} + {name_b}" if factor == 1 else f"{name_a} + {factor}x{name_b}"
        # Anchored at Spectrum A's directory (the operand named first, and
        # the dialog's default "active" pick) so fit_export.auto_log_path
        # and fit_mode.py's "Export Fit Report" default directory -- both
        # of which read os.path.dirname of a spectrum's .path -- resolve
        # somewhere real instead of silently falling back to the process's
        # cwd. Spectrum B's directory is dropped; there is no single
        # correct choice when the two operands live in different places.
        path = os.path.join(os.path.dirname(spectrum_a.path), label)
        self._add_combined_spectrum(path, data, variance=combined_variance(
            spectrum_a.data, spectrum_b.data, factor,
            spectrum_a.variance, spectrum_b.variance,
        ))

    def _open_subtract_dialog(self):
        if len(self.spectra) < 2:
            return
        active = active_spectrum(self.spectra)
        dialog = CombineDialog(self, "Subtract Spectra", self.spectra, active)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_subtract(dialog.result_spectrum_a, dialog.result_spectrum_b, dialog.result_factor)

    def _apply_subtract(self, spectrum_a, spectrum_b, factor):
        try:
            data = subtract(spectrum_a.data, spectrum_b.data, factor)
        except ValueError as exc:
            QMessageBox.warning(self, "Subtract Spectra", f"Could not subtract: {exc}")
            return
        name_a = os.path.basename(spectrum_a.path)
        name_b = os.path.basename(spectrum_b.path)
        label = f"{name_a} - {name_b}" if factor == 1 else f"{name_a} - {factor}x{name_b}"
        # See _apply_add's comment: anchored at Spectrum A's directory.
        path = os.path.join(os.path.dirname(spectrum_a.path), label)
        # Variance ADDS on subtraction too -- the factor is squared. Two
        # spectra that cancel to zero leave a result more uncertain than
        # either input, not less.
        path_variance = combined_variance(
            spectrum_a.data, spectrum_b.data, factor,
            spectrum_a.variance, spectrum_b.variance,
        )
        self._add_combined_spectrum(path, data, variance=path_variance)

    def _add_combined_spectrum(self, path, data, variance=None):
        # Shared by _apply_add/_apply_subtract and by the matrix panel's
        # Activate Cut -- inserting a newly computed spectrum into the
        # loaded list and refreshing every UI surface that depends on it
        # is identical bookkeeping either way; only the operation and
        # naming above differ.
        #
        # `variance` is the propagated per-channel variance where the
        # caller knows it. Every spectrum reaching this method is DERIVED,
        # so none of them is Poisson in its own counts; passing it through
        # is what lets a fit weight them correctly.
        existing_paths = {s.path for s in self.spectra}
        if path in existing_paths:
            suffix = 2
            while f"{path} ({suffix})" in existing_paths:
                suffix += 1
            path = f"{path} ({suffix})"
        color_index = self._next_color_index
        self._next_color_index += 1
        spectrum = LoadedSpectrum(path, data, next_color(color_index, self._theme), variance=variance)
        spectrum.color_index = color_index
        for s in self.spectra:
            s.active = False
        spectrum.active = True
        self.spectra.append(spectrum)
        # Appending one row rather than rebuilding every existing one:
        # this path runs once per Add/Subtract, so repeated use used to
        # pay O(N^2) widget construction. _sync_active_radios() then
        # moves the checked radio onto the new spectrum, since the loop
        # above just deactivated the one that had it.
        self._append_spectrum_row(spectrum)
        self._sync_active_radios()
        self._plot_data()

    def _open_save_spectrum_dialog(self):
        active = active_spectrum(self.spectra)
        if active is None:
            return
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save Spectrum", os.path.dirname(active.path),
            "SPE files (*.spe);;SPK files (*.spk);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        self._write_spectrum(active, path, chosen_filter)

    def _write_spectrum(self, spectrum, path, chosen_filter):
        lower = path.lower()
        if not lower.endswith((".spe", ".spk", ".txt")):
            # QFileDialog.getSaveFileName() has no setDefaultSuffix
            # equivalent, and unlike the native Windows/macOS Save dialogs,
            # Qt's own cross-platform dialog -- what Linux gets without
            # native GTK auto-suffixing -- does not append the chosen
            # filter's extension to a bare filename on its own. Left alone,
            # that file would stay extension-less on disk, and
            # _try_load_spectrum dispatches purely by extension, so
            # reloading it later would silently fall through to the
            # plain-text histogram loader regardless of the binary format
            # actually written. Append the extension implied by the chosen
            # filter so the file is always loadable again afterward.
            #
            # "All files (*)" (and any other unrecognized filter string)
            # falls back to .spe rather than .txt: SPE is both the
            # first-listed filter in _open_save_spectrum_dialog's filter
            # string and -- since getSaveFileName is never given a
            # selectedFilter there -- the dialog's actual pre-selected
            # default, so it's the more consistent "no clearly chosen
            # format" default than an arbitrary second special case.
            if chosen_filter.startswith("SPK"):
                path += ".spk"
            elif chosen_filter.startswith("Text"):
                path += ".txt"
            else:
                path += ".spe"
            lower = path.lower()
        # By this point `lower` always ends in one of the three recognized
        # extensions (either it already did, or the block above just
        # appended one), so the writer is fully determined by the
        # extension alone -- a path with a recognized-but-mismatched
        # extension (e.g. "out.spe" saved with the SPK filter selected)
        # deliberately keeps its own extension's writer and filename
        # rather than being forced to match the chosen filter.
        if lower.endswith(".spe"):
            writer = save_spe
        elif lower.endswith(".spk"):
            writer = save_spk
        else:
            writer = save_histogram
        try:
            writer(path, spectrum.data)
        except OSError as exc:
            QMessageBox.warning(self, "Save Spectrum", f"Could not save: {exc}")
            return
        except ValueError as exc:
            QMessageBox.warning(
                self, "Save Spectrum",
                f"Could not save in this format: {exc}\n\n"
                "Try a different format (e.g. Text), or Multiply by a smaller factor first.",
            )
            return
        if spectrum.variance is not None:
            # A derived spectrum (matrix cut, Add/Subtract, Multiply result)
            # carries a propagated per-channel variance that no spectrum
            # file format can hold -- the save quietly kept only the counts,
            # and reloading the file will re-assume Poisson uncertainties
            # read off them. That silent downgrade is exactly what the
            # variance propagation exists to prevent, so say it happened
            # rather than let a later analysis trust the wrong error bars.
            QMessageBox.information(
                self, "Save Spectrum",
                "Saved counts only. This spectrum carries propagated "
                "uncertainties (it is a derived spectrum), and no spectrum "
                "file format stores them -- reloading the saved file will "
                "assume Poisson uncertainties from the counts instead, "
                "which are not the ones this spectrum carries now.",
            )

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
            "Spectrum files (*.txt *.spe *.spk *.n42 *.lzs);;Text files (*.txt);;"
            "SPE files (*.spe);;SPK files (*.spk);;N42 files (*.n42);;"
            "LZS files (*.lzs);;All files (*)",
        )
        if paths:
            self._load_files(paths)

    def _open_matrix_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Matrix", self.settings.last_folder(), "Matrix files (*.mtx);;All files (*)"
        )
        if path:
            matrix, error = self._load_matrix_with_progress(path)
            if error is not None:
                QMessageBox.warning(self, "Could not open matrix", error)
                return
            self._open_matrix_panel(path, matrix=matrix)
            self.settings.set_last_folder(os.path.dirname(path))

    def fitted_peak_choices(self):
        """(label, channel, channel_err, fwhm, area, area_err) for every
        committed peak on the active spectrum, in Fit Results order.

        The uncertainty travels with the channel so the calibration can be
        weighted by it -- an unweighted fit lets a barely-visible line pull
        exactly as hard as the strongest peak in the spectrum, and the two
        routinely differ in precision by an order of magnitude.

        FWHM rides along so a restored assignment can be matched back to
        the peak it came from after a refit nudges the centroid (see
        energy_assignments.restore). Area is peak.area/area_err -- the
        NET, background-subtracted value, not full_area -- because the
        calibration plot hands these to CalEnEff, which divides counts by
        intensity to get efficiency; a gross area would fold the
        background into that curve.

        Always in CHANNELS, even when a calibration is already active: the
        user is assigning energies in order to determine the calibration,
        so offering them positions that a previous calibration already
        converted would be circular.
        """
        active = active_spectrum(self.spectra)
        if active is None:
            return []
        choices = []
        for fit_index, result in enumerate(active.fits, start=1):
            # active.fits holds integration results alongside fits, and an
            # IntegrationResult has no peaks at all -- it reports a region's
            # gross/background/net totals, not fitted centroids. The index
            # still advances over them so the numbering here matches what
            # the Fit Results panel shows.
            for peak_index, peak in enumerate(getattr(result, "peaks", []) or [], start=1):
                choices.append(
                    (f"Fit {fit_index}, peak {peak_index}", peak.position,
                     peak.position_err, peak.fwhm, peak.area, peak.area_err)
                )
        return choices

    def _open_calibrate_from_peaks_dialog(self):
        choices = self.fitted_peak_choices()
        if not choices:
            return
        # Fetched once and reused below: _apply_calibration_change does not
        # touch which spectrum is active, so a second lookup after it would
        # only ever repeat this same answer.
        active = active_spectrum(self.spectra)
        self._assign_energies(
            active, choices, active.energy_assignments if active else None
        )

    def _assign_energies(self, active, choices, assignments, status=None,
                         show_plot=True):
        """Open the Calibrate from Fitted Peaks dialog on `choices`,
        prefilled from `assignments`, and apply what comes back. Returns
        the applied Calibration, or None if the dialog was cancelled.

        Shared by the manual entry point and the automatic one, which
        differ only in where the assignments come from -- the dialog, its
        live plot, the ticks and the stored record are the same either
        way. `status` is a first line for the dialog's status label; the
        automatic run puts its summary there. `show_plot=False` lets the
        automatic run open the plot itself, on the refit that follows.
        """
        from energy_assign_dialog import EnergyAssignDialog

        dialog = EnergyAssignDialog(
            self, choices,
            quadratic=(self._calibration is not None
                       and self._calibration.kind == "quadratic"),
            settings=self.settings,
            assignments=assignments,
            # Enables the live preview, which needs the channel range to
            # draw over and a name to offer its export under.
            max_channel=(len(active.data) - 1) if active is not None else None,
            export_default_path=(
                self._caleneff_default_path(active) if active is not None else None
            ),
        )
        if status:
            dialog.status.setText(status)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        calibration = dialog.result_calibration
        self._apply_calibration_change(calibration, True)
        if active is not None:
            self._store_energy_assignments(active, dialog)
            if show_plot:
                self._show_calibration_plot(
                    active, calibration, dialog.export_points(),
                    dialog.source_lines, dialog.excluded_energies(),
                )
        return calibration

    def _open_auto_calibrate_dialog(self):
        """Find, fit and identify the peaks of the active spectrum, then
        hand the result to the ordinary Calibrate from Fitted Peaks dialog
        for review -- prefilled when the identification was confident,
        blank with the reason on show when it was not."""
        from auto_calibrate_dialog import AutoCalibrateDialog

        active = active_spectrum(self.spectra)
        if active is None:
            return
        remembered = active.energy_assignments
        dialog = AutoCalibrateDialog(
            self, active.data, settings=self.settings,
            # None for a spectrum read from a file; set for a matrix cut
            # or an Add/Subtract result, exactly as run_fit passes it.
            variance=getattr(active, "variance", None),
            # The source already chosen for this spectrum, if any, so it
            # need not be picked twice.
            source_path=remembered.source_path if remembered is not None else None,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        outcome = dialog.outcome
        existing = len(active.fits)
        # Appended, never replacing: whatever was fitted by hand stays, and
        # the automatic fits join it in Fit Results. Committed through the
        # same path as the Fit button, so each is logged and drawn exactly
        # as a hand fit would be.
        stamp = datetime.now().isoformat(timespec="seconds")
        for result in outcome.results:
            result.timestamp = stamp
            # NOT logged yet. If the review is accepted, every one of
            # these is replaced by a refit of the same peak, and logging
            # both would put two records of each line in the spectrum's
            # `_fits.jsonl` -- the same double count the in-memory list
            # goes to some trouble to avoid, in the file the HowTo tells
            # people to process with other tools. Logged below, once it
            # is known which pass the user is keeping.
            self.fit_controller._commit_result(active, result, log=False)
        self._plot_data(preserve_view=True)
        summary = outcome.summary(existing, dialog.source_name())
        self.fit_controller._show_status_message(summary, 10000)
        # Stored now rather than only on OK, so cancelling the review does
        # not throw the identifications away -- running again would append
        # a second copy of every fit. A refused match stores the source
        # alone, which is still worth remembering. Points the efficiency
        # test doubts open UNTICKED: assigned and visible, hollow on the
        # plot, but out of the fit until the user says otherwise.
        active.energy_assignments = EnergyAssignments(
            source_path=dialog.source_path(), pairs=tuple(outcome.pairs),
            excluded=outcome.suspect_energies,
        )
        calibration = self._assign_energies(
            active, self.fitted_peak_choices(), active.energy_assignments,
            status=summary, show_plot=False,
        )
        if calibration is None or dialog.source_lines is None:
            # No refit is coming, so these fits are the ones being kept.
            for result in outcome.results:
                self.fit_controller.append_auto_log(active, result)
            return
        self._refit_for_caleneff(active, dialog, outcome, calibration)

    def _refit_for_caleneff(self, active, dialog, outcome, calibration):
        """With the calibration applied, fit every source line that is
        visibly present, one at a time, and make THOSE the fits on the
        spectrum -- the input CalEnEff needs is one clean area per line
        of the source.

        The fits from the identification pass are replaced, not kept
        beside the new ones: they were the same peaks, and a spectrum
        carrying two fits of every line would export every efficiency
        point twice. Fits made by hand before the run are untouched.

        A point the user unticked in the review is left out of this pass
        as well. Unticking here means the area does not belong on the
        efficiency curve, and the whole purpose of the refit is to
        measure areas for that curve.
        """
        import auto_calibrate

        remembered = active.energy_assignments
        unticked = tuple(remembered.excluded) if remembered is not None else ()
        refit = auto_calibrate.refit_source_lines(
            channel_indices(len(active.data)), active.data, dialog.source_lines,
            calibration, sensitivity=dialog.sensitivity.value(),
            variance=getattr(active, "variance", None), excluded=unticked,
        )
        superseded = {id(result) for result in outcome.results}
        active.fits[:] = [f for f in active.fits if id(f) not in superseded]
        stamp = datetime.now().isoformat(timespec="seconds")
        for result in refit.results:
            result.timestamp = stamp
            self.fit_controller._commit_result(active, result)
        self.fit_controller.update_results_list()
        self._plot_data(preserve_view=True)
        active.energy_assignments = EnergyAssignments(
            source_path=dialog.source_path(), pairs=tuple(refit.pairs),
            # Kept though none of these lines is now on the spectrum:
            # the user's decision outlives the fits it was made about.
            excluded=unticked,
        )
        points = [
            (result.peaks[0].position, result.peaks[0].position_err or 0.0,
             result.peaks[0].area, result.peaks[0].area_err, energy)
            for result, (_channel, energy) in zip(refit.results, refit.pairs)
        ]
        self._show_calibration_plot(active, calibration, points, dialog.source_lines, ())
        self.fit_controller._show_status_message(refit.summary(dialog.source_name()), 10000)

    def _store_energy_assignments(self, spectrum, dialog):
        """Remember what the dialog was told, so a refit does not throw it
        away.

        Derived from the table's live contents rather than from the Clear
        button's flag: Clear followed by fresh energies is the ordinary way
        to start over, and a flag that only ever latches True would erase
        exactly the assignments the user had just retyped.
        """
        channels, energies, _errors = dialog.assignments()
        if not channels:
            spectrum.energy_assignments = None
            return
        spectrum.energy_assignments = EnergyAssignments(
            source_path=dialog.source_path(),
            pairs=tuple(zip(channels, energies)),
            # Which points the user left out of the fit is part of what
            # they told the dialog, so it is stored with the energies
            # rather than being rediscovered after every refit.
            excluded=dialog.excluded_energies(),
        )

    def _caleneff_default_path(self, spectrum):
        """Filename the CalEnEff export is offered under, derived from
        the spectrum's own name. Shared by the live preview and the
        window opened on OK so the two never propose different files
        for the same spectrum."""
        source = spectrum.path.split("::", 1)[0]
        stem = os.path.splitext(os.path.basename(source))[0]
        return os.path.join(os.path.dirname(source), f"{stem}_En_Area.txt")

    def _show_calibration_plot(self, spectrum, calibration, points, source_lines,
                               excluded):
        """The coefficients alone look equally plausible whether or not
        a line was misidentified; the plot's residual strip is where
        that shows."""
        default_path = self._caleneff_default_path(spectrum)
        # Qt keeps a parented dialog alive after the attribute is rebound,
        # so without this a second calibration leaves the first window on
        # screen showing different coefficients and offering to export to
        # the same filename.
        previous = getattr(self, "_calibration_plot", None)
        if previous is not None:
            previous.close()
            previous.deleteLater()
        self._calibration_plot = CalibrationPlotDialog(
            self, calibration, points, source_lines,
            len(spectrum.data) - 1, default_path, excluded=excluded,
        )
        self._calibration_plot.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._calibration_plot.show()

    def _save_fits_dialog(self):
        active = active_spectrum(self.spectra)
        if active is None or not fit_persist.savable(active.fits):
            return
        stem = os.path.splitext(os.path.basename(active.path.split("::", 1)[0]))[0]
        directory = os.path.dirname(active.path.split("::", 1)[0])
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Fits", os.path.join(directory, f"{stem}_fits.json"),
            "Fit files (*.json);;All files (*)",
        )
        if not path:
            return
        saved = fit_persist.savable(active.fits)
        try:
            fit_persist.save(path, active.fits, active.path)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save fits", str(exc))
            return
        skipped = len(active.fits) - len(saved)
        message = f"Saved {len(saved)} fit(s)."
        if skipped:
            # Said out loud rather than left for the user to discover on
            # reload: integrations are visible in the same panel, so
            # "Saved 2 fits" when the panel shows 5 rows needs explaining.
            message += f" {skipped} integration result(s) were not saved."
        self.fit_controller._show_status_message(message, 5000)

    def _load_fits_dialog(self):
        active = active_spectrum(self.spectra)
        if active is None:
            return
        directory = os.path.dirname(active.path.split("::", 1)[0])
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Fits", directory, "Fit files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            results, saved_spectrum = fit_persist.load(path)
        except fit_persist.FitFileError as exc:
            QMessageBox.warning(self, "Could not load fits", str(exc))
            return
        if not results:
            QMessageBox.warning(self, "Could not load fits", "That file holds no fits.")
            return

        # Loading fits saved against a DIFFERENT spectrum is allowed --
        # comparing one run's fits against another's is a real thing to
        # want -- but it is worth saying out loud, because the marks will
        # land wherever those channel numbers fall in this spectrum.
        note = ""
        if saved_spectrum and saved_spectrum != active.path:
            note = (
                f"\n\nThese fits were saved against:\n{saved_spectrum}\n"
                f"and will be applied to:\n{active.path}"
            )

        choice = QMessageBox.question(
            self, "Load Fits",
            f"Load {len(results)} fit(s)?\n\n"
            "Yes  - restore the saved results exactly as they were\n"
            "No   - re-run each fit from its saved marks using the current "
            "fitting code" + note,
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return

        if choice == QMessageBox.StandardButton.Yes:
            loaded = results
            failures = []
        else:
            loaded, failures = [], []
            x = channel_indices(len(active.data))
            for index, result in enumerate(results, start=1):
                try:
                    refitted = fit_persist.refit(
                        result, x, active.data, getattr(active, "variance", None)
                    )
                except FitError as exc:
                    failures.append(f"Fit {index}: {exc}")
                    continue
                refitted.timestamp = datetime.now().isoformat(timespec="seconds")
                loaded.append(refitted)

        active.fits.extend(loaded)
        self.fit_controller.update_results_list()
        self._plot_data(preserve_view=True)
        self.fit_controller._show_status_message(
            f"Loaded {len(loaded)} fit(s)"
            + (f"; {len(failures)} could not be refitted." if failures else "."),
            6000,
        )
        if failures:
            QMessageBox.warning(self, "Some fits could not be refitted", "\n".join(failures))

    def _reload_active_spectrum(self):
        """Re-read the active spectrum's file in place, keeping its
        calibration, fits and marks.

        For watching an acquisition that is still counting: the file on
        disk grows, and this picks up the new counts without losing the
        analysis set up around them. Manual rather than polled -- a timer
        would redraw under the user's hands mid-measurement, and
        file-watching behaves differently on Windows and Linux.
        """
        active = active_spectrum(self.spectra)
        if active is None:
            return

        path = active.path
        object_path = None
        if "::" in path:
            # A ROOT spectrum is one object inside a file; both halves are
            # needed to find it again.
            path, object_path = path.split("::", 1)

        if not os.path.exists(path):
            QMessageBox.warning(
                self, "Could not reload",
                f"{os.path.basename(path)} is no longer on disk.",
            )
            return

        previous_length = len(active.data)
        try:
            if object_path is not None:
                from root_io import RootError, list_objects, load_spectrum
                try:
                    # The cycle suffix is not in the displayed path, so the
                    # object is matched on its stem.
                    key = next(
                        k for k, _cls, kind in list_objects(path)
                        if kind == "spectrum" and k.split(";")[0] == object_path
                    )
                except StopIteration:
                    raise RootError(f"{object_path!r} is no longer in the file")
                data, _calibration = load_spectrum(path, key)
            else:
                spectrum, error, _calibration = self._try_load_spectrum(path)
                if error:
                    QMessageBox.warning(self, "Could not reload", error)
                    return
                data = spectrum.data
                # _try_load_spectrum allocates a colour for the throwaway
                # spectrum it builds; hand it back so reloading repeatedly
                # does not walk the palette.
                self._next_color_index -= 1
        except Exception as exc:
            QMessageBox.warning(self, "Could not reload", f"{os.path.basename(path)}: {exc}")
            return

        active.data = data
        # A spectrum that changed length invalidates anything anchored to a
        # channel number: the marks and fits describe positions that may no
        # longer mean the same thing. Dropping them silently would be worse
        # than saying so -- the user would keep reading fits that no longer
        # match the data under them.
        if len(data) != previous_length:
            active.fits.clear()
            self.fit_controller.reset_marks()
            self.fit_controller.update_results_list()
            self.fit_controller._show_status_message(
                f"Reloaded: channel count changed {previous_length} -> {len(data)}, "
                "so fits and marks were cleared.", 8000,
            )
        else:
            self.fit_controller._show_status_message("Reloaded from disk.", 4000)
        self._plot_data(preserve_view=True)

    def _open_root_dialog(self):
        """Open a ROOT file: pick the file, then pick what to take out of
        it. Spectra join the spectrum list; a matrix opens its own panel."""
        from root_dialog import RootObjectDialog
        from root_io import RootError, list_objects

        path, _ = QFileDialog.getOpenFileName(
            self, "Open ROOT File", self.settings.last_folder(),
            "ROOT files (*.root);;All files (*)",
        )
        if not path:
            return
        try:
            objects = list_objects(path)
        except RootError as exc:
            QMessageBox.warning(self, "Could not open ROOT file", str(exc))
            return
        if not objects:
            QMessageBox.warning(
                self, "Could not open ROOT file",
                f"{os.path.basename(path)} contains no 1D or 2D histograms.",
            )
            return

        dialog = RootObjectDialog(self, path, objects)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.settings.set_last_folder(os.path.dirname(path))
        if dialog.result_matrix is not None:
            self._open_root_matrix(path, dialog.result_matrix)
        elif dialog.result_spectra:
            self._load_root_spectra(path, dialog.result_spectra)

    def _load_root_spectra(self, path, object_paths):
        """Loads the chosen 1D histograms as spectra.

        Deliberately mirrors _load_files' sequencing rather than reusing
        it: _load_files is keyed on a file path per spectrum, and here one
        file yields several. The parts that matter are kept -- the first
        embedded calibration wins and is applied ONCE after every spectrum
        is in the list, not per file, which is the v3.1.0 M18 fix.
        """
        from root_io import RootError, load_spectrum

        failures = []
        added = []
        pending_calibration = None
        for object_path in object_paths:
            display = f"{path}::{object_path.split(';')[0]}"
            if any(s.path == display for s in self.spectra):
                continue
            try:
                data, calibration = load_spectrum(path, object_path)
            except RootError as exc:
                failures.append(f"{object_path}: {exc}")
                continue
            color_index = self._next_color_index
            self._next_color_index += 1
            spectrum = LoadedSpectrum(display, data, next_color(color_index, self._theme))
            spectrum.color_index = color_index
            if not self.spectra:
                spectrum.active = True
            self.spectra.append(spectrum)
            added.append(spectrum)
            if calibration is not None and pending_calibration is None:
                pending_calibration = calibration

        if added:
            self.settings.add_recent_file(path)
            self._update_recent_menu()
            for spectrum in added:
                self._append_spectrum_row(spectrum)
            self._sync_active_radios()
            if pending_calibration is not None and not self._calibration_active:
                # Redraws by itself, so it replaces the _plot_data() below
                # rather than adding to it -- same reasoning as _load_files.
                self._apply_calibration_change(pending_calibration, True)
            else:
                self._plot_data()
        if failures:
            QMessageBox.warning(
                self, "Could not open ROOT file", "\n".join(failures)
            )

    def _open_root_matrix(self, path, object_path):
        from root_io import RootError, load_matrix

        try:
            matrix, _x_cal, _y_cal = load_matrix(path, object_path)
        except RootError as exc:
            QMessageBox.warning(self, "Could not open matrix", str(exc))
            return
        # No progress dialog: a ROOT TH2 is already decoded by uproot into
        # a numpy array in one step, so there is no row-by-row pass to
        # report on the way .mtx has. The axis calibrations are read but
        # not yet applied -- the matrix panel works in channels, and
        # wiring a per-axis calibration into it is its own change.
        self._open_matrix_panel(f"{path}::{object_path.split(';')[0]}", matrix=matrix)

    def _load_matrix_with_progress(self, path):
        """Decodes `path` in a worker thread behind a progress dialog.

        Returns (matrix, None) or (None, error_message).

        This used to run on the GUI thread behind a wait cursor and a
        status message flushed with processEvents(). That told the user
        something was happening but the window still could not repaint
        for the several seconds the decode takes, so it still looked
        hung. A worker thread keeps the event loop alive, and because
        load_mtx reports row progress the bar shows real progress rather
        than an indeterminate spinner.

        The dialog has no cancel button on purpose: load_mtx builds the
        matrix in one pass with no unwind path, so a cancel could only be
        honoured between rows and would leave a half-decoded array to
        throw away. A few seconds does not warrant that machinery.

        Deliberately NOT folded into _open_matrix_panel: that method is
        the synchronous seam every other caller (and every test) uses to
        get a panel back immediately, and threading it would turn a
        simple call into an event-loop dependency.
        """
        dialog = QProgressDialog("Reading matrix...", None, 0, 100, self)
        dialog.setWindowTitle("Open Matrix")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)

        outcome = {"matrix": None, "error": None}
        worker = _MatrixLoadWorker(path, self)

        def on_progress(done, total):
            if total:
                dialog.setValue(int(done * 100 / total))

        worker.progressed.connect(on_progress)
        worker.loaded.connect(lambda m: outcome.__setitem__("matrix", m))
        worker.failed.connect(lambda msg: outcome.__setitem__("error", msg))
        # close(), not accept()/reject(): this dialog is only ever
        # dismissed by the load finishing, so there is no accepted vs
        # rejected distinction to preserve.
        worker.finished.connect(dialog.close)
        worker.start()
        dialog.exec()
        # The nested event loop above ends when the dialog closes, which
        # the finished signal drives -- but wait() makes the hand-off
        # explicit rather than relying on that ordering, and guarantees
        # the thread is done before its results are read.
        worker.wait()
        return outcome["matrix"], outcome["error"]

    def _open_matrix_panel(self, path, matrix=None):
        panel = MatrixPanel(self, path, matrix=matrix)
        self._matrix_panels.append(panel)
        panel.show()
        return panel

    def _on_activated(self):
        for panel in self._matrix_panels:
            panel.setEnabled(False)
        self.setEnabled(True)

    def _try_load_spectrum(self, path):
        lower = path.lower()
        calibration = None
        try:
            if lower.endswith(".spe"):
                data = load_spe(path)
            elif lower.endswith(".spk"):
                data = load_spk(path)
            elif lower.endswith(".n42"):
                data, calibration = load_n42(path)
            elif lower.endswith(".lzs"):
                data, calibration = load_lzs(path)
            else:
                data = load_histogram(path)
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
        # The calibration is RETURNED rather than applied here: applying
        # it mid-load fired _apply_calibration_change while this spectrum
        # was still not in self.spectra, redrawing a plot that didn't yet
        # contain the file that supplied the calibration, and then getting
        # redrawn again by _load_files moments later. _load_files applies
        # it once the list is complete instead.
        return spectrum, None, calibration

    def _load_files(self, paths):
        failures = []
        loaded_any = False
        # First embedded calibration wins, matching the previous
        # per-file behaviour: the old code applied one as soon as it saw
        # it, which set _calibration_active and made every later file in
        # the same batch skip its own.
        pending_calibration = None
        added = []
        for path in paths:
            if any(s.path == path for s in self.spectra):
                continue
            spectrum, error, calibration = self._try_load_spectrum(path)
            if error:
                failures.append(error)
                continue
            if not self.spectra:
                spectrum.active = True
            self.spectra.append(spectrum)
            added.append(spectrum)
            self.settings.set_last_folder(os.path.dirname(path))
            self.settings.add_recent_file(path)
            if calibration is not None and pending_calibration is None:
                pending_calibration = calibration
            loaded_any = True

        if loaded_any:
            self._update_recent_menu()
            # Only the newly loaded spectra need rows built; opening a
            # batch into an already-populated list no longer reconstructs
            # the rows that were already there.
            for spectrum in added:
                self._append_spectrum_row(spectrum)
            self._sync_active_radios()
            if pending_calibration is not None and not self._calibration_active:
                # Redraws by itself, so it replaces the _plot_data()
                # below rather than adding to it.
                self._apply_calibration_change(pending_calibration, True)
            else:
                self._plot_data()

        if failures:
            QMessageBox.warning(self, "Some files could not be loaded", "\n".join(failures))

    def _plot_data(self, preserve_view=False, xlim_override=None):
        # Captured before axes.clear() (which resets limits) -- used to
        # keep the user's current zoom when a replot is triggered by
        # something unrelated to loading/showing a spectrum, e.g.
        # committing or removing a peak fit.
        saved_xlim = self.axes.get_xlim() if preserve_view else None
        self.axes.clear()
        style_axes(self.axes, self._theme)
        visible = [s for s in self.spectra if s.visible]
        # The final limits are resolved BEFORE anything is drawn, purely
        # so draw_committed_fits can skip fits that fall outside them.
        # They are still applied below, in the original place, so the
        # autoscale behaviour is unchanged -- this only moves the
        # decision earlier. It has to be: set_xlim happens after the
        # loop, so a fit-culling test reading axes.get_xlim() from inside
        # the loop would see matplotlib's provisional data-derived
        # limits, not the view the user ends up looking at, and would
        # drop fits that belong on screen.
        view_xlim = None
        if visible:
            max_channel = max(len(s.data) for s in visible) - 1
            if xlim_override is not None:
                view_xlim = xlim_override
            elif saved_xlim is not None:
                view_xlim = saved_xlim
            else:
                view_xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
        for spectrum in visible:
            # channel_indices rather than a fresh np.arange: this runs
            # for every visible spectrum on every replot, and the cached
            # read-only axis exists for exactly this.
            channels = channel_indices(len(spectrum.data))
            x = self.channel_to_display(channels)
            self.axes.plot(x, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
            self.fit_controller.draw_committed_fits(spectrum, view_xlim=view_xlim)
        self.draw_goto_marker()
        self.axes.set_xlabel("Energy (keV)" if self._calibration_active else "Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        if visible:
            # Channel numbers can't be negative; override Matplotlib's
            # default 5% autoscale margin, which would otherwise show them
            # as such. Span the widest currently-visible spectrum, since
            # loaded files can have different channel counts.
            # Same value the fit-culling above already resolved.
            xlim = view_xlim
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
        self._update_operations_availability()
        self.fit_controller.update_results_list()

    def _pan_button_is_active(self, event):
        """Whether this press starts a drag-pan -- see
        spectrum.pan_button_is_active, which the matrix panel shares."""
        return pan_button_is_active(
            event, self.axes, self.nav_toolbar, (self.fit_controller,)
        )

    def _on_pan_press(self, event):
        if not self._pan_button_is_active(event):
            return
        self._pan_last_px = event.x

    def _on_pan_release(self, event):
        if self._pan_last_px is None:
            return
        self._pan_last_px = None
        # Record the panned view so Home/Back/Forward know about it, the
        # same way the zoom actions do.
        self.nav_toolbar.push_current()

    def _pan_to(self, event):
        """Drags the view with the cursor. Returns True if it panned.

        Works from the PIXEL delta between motion events, not from the
        data coordinate the drag started at: xlim changes as the drag
        proceeds, so event.xdata is expressed in a frame that has already
        moved, and differencing against the original press point drifts.
        An incremental pixel delta converted through the current span is
        stable however far the drag goes.
        """
        if self._pan_last_px is None or event.x is None:
            return False
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return False
        width_px = self.axes.bbox.width
        if not width_px:
            return False
        xlim = self.axes.get_xlim()
        # Content follows the cursor: dragging right moves the view left.
        delta = -(event.x - self._pan_last_px) * (xlim[1] - xlim[0]) / width_px
        self._pan_last_px = event.x
        max_channel = max(len(s.data) for s in visible) - 1
        new_xlim = panned_xlim(
            xlim, delta, self.channel_to_display(0), self.channel_to_display(max_channel)
        )
        if new_xlim == xlim:
            return True
        self.axes.set_xlim(new_xlim)
        # The X span is deliberately preserved while Y re-fits whatever
        # is now on screen -- the whole point of the gesture is to walk
        # along a spectrum at one zoom level without small peaks being
        # flattened by a tall one that has scrolled off.
        self._autoscale_y(new_xlim)
        self.canvas.draw_idle()
        return True

    def _on_mouse_move(self, event):
        if self._pan_to(event):
            return
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
            self._plot_data(preserve_view=True)

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
        # Recreated each rebuild. Detaching the old group from `self`
        # first is load-bearing, not tidiness: QButtonGroup(self) hands
        # C++ ownership to the main window, so merely rebinding the
        # Python attribute leaks the old group -- it survives as a child
        # of the window forever (measured: 21 live instances after 20
        # rebuilds, even after gc.collect()). setParent(None) hands
        # ownership back to Python, so the rebind below drops the last
        # reference and the C++ object goes with it. Deterministic, unlike
        # deleteLater(), which needs an event-loop turn that a rebuild
        # triggered from a menu action doesn't necessarily reach.
        old_group = getattr(self, "active_button_group", None)
        if old_group is not None:
            old_group.setParent(None)
        self.active_button_group = QButtonGroup(self)

        for spectrum in self.spectra:
            self._append_spectrum_row(spectrum)

    def _append_spectrum_row(self, spectrum):
        """Builds and appends ONE spectrum's row.

        Used both by _update_spectrum_list()'s full rebuild and, on its
        own, by the add/load paths -- appending a row for a newly loaded
        spectrum leaves every existing row untouched, so N sequential
        single-spectrum operations cost O(N) row constructions instead
        of O(N^2). (Measured before this: rebuilding the whole list took
        5.5 ms at 20 spectra, 17 ms at 50, 96 ms at 200, on every single
        add or remove.)

        Order matters in two places, both load-bearing: each widget's
        state is set BEFORE its signal is connected, so populating a row
        never emits toggled() and never re-enters the handlers; and the
        radio joins the exclusive button group only after its checked
        state is set, so the group never briefly sees two checked
        buttons.
        """
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

    def _sync_active_radios(self):
        """Re-points the checked radio at whichever spectrum is active,
        without rebuilding any row. Needed after an incremental removal,
        which can promote a different spectrum to active. Signals are
        blocked because these radios already reflect a decision the
        model has made -- letting them re-emit would drive
        _on_active_toggled and re-enter the very update that is running.
        Exclusivity is lifted for the duration for the same reason it is
        in the append path: an exclusive group refuses to have zero
        checked buttons mid-update."""
        group = self.active_button_group
        was_exclusive = group.exclusive()
        group.setExclusive(False)
        try:
            for index, spectrum in enumerate(self.spectra):
                item = self.spectrum_list.item(index)
                if item is None:
                    continue
                widget = self.spectrum_list.itemWidget(item)
                if widget is None:
                    continue
                for radio in widget.findChildren(QRadioButton):
                    radio.blockSignals(True)
                    radio.setChecked(spectrum.active)
                    radio.blockSignals(False)
        finally:
            group.setExclusive(was_exclusive)

    def _remove_spectrum_row(self, path):
        """Drops the one row whose spectrum has `path`, leaving the rest
        in place. Returns True if a row was found and removed; a False
        return means the list and self.spectra have diverged, and the
        caller should fall back to a full rebuild rather than carry on
        with a stale list."""
        for index in range(self.spectrum_list.count()):
            item = self.spectrum_list.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == path:
                widget = self.spectrum_list.itemWidget(item)
                if widget is not None:
                    for radio in widget.findChildren(QRadioButton):
                        self.active_button_group.removeButton(radio)
                self.spectrum_list.takeItem(index)
                return True
        return False

    def _on_show_toggled(self, path, checked):
        for spectrum in self.spectra:
            if spectrum.path == path:
                spectrum.visible = checked
                break
        self._plot_data(preserve_view=True)

    def _on_active_toggled(self, path, checked):
        if not checked:
            return
        for spectrum in self.spectra:
            spectrum.active = spectrum.path == path
        self._update_fit_mode_availability()
        self._update_operations_availability()
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
            self._remove_spectrum(path)

    def _remove_spectrum(self, path):
        removed_was_active = any(s.path == path and s.active for s in self.spectra)
        self.spectra = [s for s in self.spectra if s.path != path]
        if removed_was_active and self.spectra:
            self.spectra[0].active = True
        # Drop just this row; every other row is unaffected by a removal.
        # Falls back to a full rebuild if the row isn't found, rather
        # than leaving the list out of step with self.spectra.
        if self._remove_spectrum_row(path):
            self._sync_active_radios()
        else:
            self._update_spectrum_list()
        self._plot_data(preserve_view=True)

    def _close_active_spectrum(self):
        active = active_spectrum(self.spectra)
        if active is None:
            return
        self._remove_spectrum(active.path)

    def _build_zoom_buttons(self):
        self.nav_toolbar.addSeparator()
        dark = self._theme == "dark"

        self.zoom_in_action = QAction(_magnifier_icon("+", dark), "Zoom In X", self)
        self.zoom_in_action.setShortcut("Ctrl+=")
        self.zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(_magnifier_icon("-", dark), "Zoom Out X", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_out_action)

        self.full_spectrum_action = QAction(_full_spectrum_icon(dark), "Show Full Spectrum", self)
        self.full_spectrum_action.setShortcut("Ctrl+0")
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

    def _build_calibration_toolbar_buttons(self):
        self.nav_toolbar.addSeparator()
        dark = self._theme == "dark"

        self.calibration_load_action = QAction(_calibration_icon(dark), "Load Calibration...", self)
        self.calibration_load_action.triggered.connect(self._open_calibration_dialog)
        self.nav_toolbar.addAction(self.calibration_load_action)

        self.calibration_toggle_action = QAction(
            _calibration_active_icon(False, dark), "Calibration Active", self
        )
        self.calibration_toggle_action.setCheckable(True)
        self.calibration_toggle_action.setEnabled(self._calibration is not None)
        self.calibration_toggle_action.toggled.connect(self._on_calibration_toggle_action)
        self.nav_toolbar.addAction(self.calibration_toggle_action)

    def _refresh_calibration_icons(self):
        """Re-renders this app's own calibration toolbar icons for the
        current theme -- mirrors _refresh_zoom_icons; a no-op before
        _build_calibration_toolbar_buttons has run (e.g. the initial
        _apply_theme call in __init__)."""
        if not hasattr(self, "calibration_load_action"):
            return
        dark = self._theme == "dark"
        self.calibration_load_action.setIcon(_calibration_icon(dark))
        self.calibration_toggle_action.setIcon(
            _calibration_active_icon(self.calibration_toggle_action.isChecked(), dark)
        )

    def _on_calibration_toggle_action(self, checked):
        self.calibration_toggle_action.setIcon(
            _calibration_active_icon(checked, self._theme == "dark")
        )
        if self._calibration is None or checked == self._calibration_active:
            return
        self._apply_calibration_change(self._calibration, checked)

    def _warn_if_calibration_folds(self, calibration):
        """Warns when `calibration` reverses direction inside the loaded
        spectra's channel range, which makes the energy axis fold back on
        itself: two channels then share one energy, and converting an
        energy to a channel has two answers rather than one.

        Only reachable with a quadratic. It is worth interrupting for
        because nothing else about such a calibration looks wrong -- it
        still passes through every assigned point, so the coefficients
        and the residuals can both look reasonable while Go To, the
        keV-space fit parameters and the axis labels quietly disagree
        about which channel an energy means. A mistyped energy or a peak
        assigned to the wrong line is enough to produce one.

        Deliberately silent when the turning point lies OUTSIDE the data:
        a parabola has one everywhere, and a curve that only bends beyond
        the last channel is an ordinary, perfectly usable calibration.
        """
        turning = turning_point(calibration)
        if turning is None or not self.spectra:
            return
        max_channel = max(len(spectrum.data) for spectrum in self.spectra) - 1
        if not (0.0 <= turning <= max_channel):
            return
        QMessageBox.warning(
            self, "Calibration",
            f"This calibration reverses direction at channel {turning:.1f}, "
            f"which is inside the loaded data (0-{max_channel}).\n\n"
            "Above and below that channel the energy axis runs opposite "
            "ways, so two different channels share the same energy and "
            "converting an energy back to a channel is ambiguous. Go To "
            "and any energy you type into the Fit Parameters panel may "
            "resolve to the wrong side.\n\n"
            "A quadratic that turns inside the data usually means one "
            "assigned energy is wrong, or that the points do not support "
            "a quadratic -- check the residuals, or fit a line instead.",
        )

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

        self.background_preview_button = QAction("Preview Background Fit", self)
        self.background_preview_button.setShortcut("Ctrl+B")
        self.background_preview_button.triggered.connect(self.fit_controller.toggle_background_preview)
        self.addAction(self.background_preview_button)

    def _on_canvas_click(self, event):
        self.fit_controller.on_click(event)

    def _update_fit_mode_availability(self):
        active = active_spectrum(self.spectra)
        available = active is not None and active.visible
        self.fit_button.setEnabled(available and self.fit_controller.state.ready_to_fit())
        self.integrate_button.setEnabled(available and self.fit_controller.state.ready_to_integrate())

    def _update_operations_availability(self):
        active = active_spectrum(self.spectra)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        self.close_spectrum_action.setEnabled(active is not None)
        # A spectrum computed in memory -- an Add/Subtract result, or a
        # matrix cut -- has no file behind it to re-read, so offering
        # Reload for it would be an action that can only fail.
        self.reload_spectrum_action.setEnabled(
            active is not None and os.path.exists(active.path.split("::", 1)[0])
        )
        # Saving needs something to save; loading only needs somewhere to
        # put it.
        self.save_fits_action.setEnabled(
            active is not None and bool(fit_persist.savable(active.fits))
        )
        self.load_fits_action.setEnabled(active is not None)
        # Needs at least two fitted peaks, since that is the fewest a line
        # can be drawn through.
        self.calibrate_from_peaks_action.setEnabled(
            len(self.fitted_peak_choices()) >= 2
        )
        # Finds and fits its own peaks, so a spectrum is all it needs.
        self.auto_calibrate_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
        self.add_action.setEnabled(len(self.spectra) >= 2)
        self.subtract_action.setEnabled(len(self.spectra) >= 2)

    def _on_scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        # Gentler than the toolbar buttons, and proportional to how
        # far the wheel actually turned -- see wheel_zoom_factor.
        self._zoom_x(wheel_zoom_factor(event), center=event.xdata)

    def _zoom_x(self, factor, center=None):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        xlim = self.axes.get_xlim()
        max_channel = max(len(s.data) for s in visible) - 1
        display_lo = self.channel_to_display(0)
        display_hi = self.channel_to_display(max_channel)
        display_lo, display_hi = min(display_lo, display_hi), max(display_lo, display_hi)
        # Anchored on `center`, clamped to the data, orientation
        # preserved -- see calibration_view.zoomed_limits, which the
        # matrix panel's projection shares so the two cannot drift.
        new_xlim = zoomed_limits(xlim, factor, center, display_lo, display_hi)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        # draw_idle, not draw: this is the one redraw path a user can
        # trigger in a burst -- a mouse wheel emits events far faster
        # than a full canvas render completes. draw() renders each one
        # synchronously, so N wheel ticks cost N full renders and the
        # view visibly lags behind the wheel; draw_idle() coalesces the
        # burst into a single render at the next event-loop pass.
        #
        # Measured on Windows, one render is ~22 ms of the ~21 ms zoom
        # step -- i.e. the arithmetic here is free and rendering is the
        # entire cost. That is tolerable on a GPU-accelerated desktop
        # and is NOT on a software renderer (WSLg passes no GPU through
        # at all), which is where the lag was reported.
        self.canvas.draw_idle()
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
