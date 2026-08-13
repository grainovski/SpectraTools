import math
import os
import time

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QRectF
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

from calibration_dialog import CalibrationDialog
from combine_dialog import CombineDialog
from factor_dialog import FactorDialog
from fit_mode import FitModeController
from help_content import (
    build_about_html,
    build_howto_html,
    build_knowledge_database_html,
    open_help_page,
)
from histogram_io import ParseError, load_histogram, save_histogram
from matrix_panel import MatrixPanel
from n42_io import load_n42
from settings import Settings
from spe_io import load_spe, save_spe
from spectrum import LoadedSpectrum, next_color
from spectrum_operations import add, multiply, normalize_factors, rebin, reference_value, subtract
from spk_io import load_spk, save_spk
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

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

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
        self.log_scale_action = QAction("Log scale Y", self)
        self.log_scale_action.setShortcut("Ctrl+G")
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
        self._style_nav_toolbar_palette(theme)
        self._refresh_builtin_toolbar_icons()
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
            self._apply_calibration_change(dialog.result_calibration, dialog.result_active)

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
        active = next((s for s in self.spectra if s.active), None)
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
        spectrum.data = multiply(spectrum.data, factor)
        self.fit_controller.reset_marks()
        spectrum.fits.clear()
        self._plot_data(preserve_view=True)

    def _open_rebin_dialog(self):
        active = next((s for s in self.spectra if s.active), None)
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
            spectrum.data = multiply(spectrum.data, factor)
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
        active = next((s for s in self.spectra if s.active), None)
        dialog = CombineDialog(self, "Add Spectra", self.spectra, active)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_add(dialog.result_spectrum_a, dialog.result_spectrum_b, dialog.result_factor)

    def _apply_add(self, spectrum_a, spectrum_b, factor):
        data = add(spectrum_a.data, spectrum_b.data, factor)
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
        self._add_combined_spectrum(path, data)

    def _open_subtract_dialog(self):
        if len(self.spectra) < 2:
            return
        active = next((s for s in self.spectra if s.active), None)
        dialog = CombineDialog(self, "Subtract Spectra", self.spectra, active)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_subtract(dialog.result_spectrum_a, dialog.result_spectrum_b, dialog.result_factor)

    def _apply_subtract(self, spectrum_a, spectrum_b, factor):
        data = subtract(spectrum_a.data, spectrum_b.data, factor)
        name_a = os.path.basename(spectrum_a.path)
        name_b = os.path.basename(spectrum_b.path)
        label = f"{name_a} - {name_b}" if factor == 1 else f"{name_a} - {factor}x{name_b}"
        # See _apply_add's comment: anchored at Spectrum A's directory.
        path = os.path.join(os.path.dirname(spectrum_a.path), label)
        self._add_combined_spectrum(path, data)

    def _add_combined_spectrum(self, path, data):
        # Shared by _apply_add/_apply_subtract -- inserting a newly
        # computed spectrum into the loaded list and refreshing every
        # UI surface that depends on it is identical bookkeeping either
        # way; only the operation and naming above differ.
        existing_paths = {s.path for s in self.spectra}
        if path in existing_paths:
            suffix = 2
            while f"{path} ({suffix})" in existing_paths:
                suffix += 1
            path = f"{path} ({suffix})"
        color_index = self._next_color_index
        self._next_color_index += 1
        spectrum = LoadedSpectrum(path, data, next_color(color_index, self._theme))
        spectrum.color_index = color_index
        for s in self.spectra:
            s.active = False
        spectrum.active = True
        self.spectra.append(spectrum)
        self._update_spectrum_list()
        self._plot_data()

    def _open_save_spectrum_dialog(self):
        active = next((s for s in self.spectra if s.active), None)
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
        if lower.endswith(".spe"):
            writer = save_spe
        elif lower.endswith(".spk"):
            writer = save_spk
        elif lower.endswith(".txt"):
            writer = save_histogram
        elif chosen_filter.startswith("SPE"):
            writer = save_spe
        elif chosen_filter.startswith("SPK"):
            writer = save_spk
        else:
            writer = save_histogram
        try:
            writer(path, spectrum.data)
        except OSError as exc:
            QMessageBox.warning(self, "Save Spectrum", f"Could not save: {exc}")
        except ValueError as exc:
            QMessageBox.warning(
                self, "Save Spectrum",
                f"Could not save in this format: {exc}\n\n"
                "Try a different format (e.g. Text), or Multiply by a smaller factor first.",
            )

    def _style_nav_toolbar_palette(self, theme):
        """Sets the navigation toolbar's actual QPalette -- not just this
        app's own QSS, which changes the toolbar's *paint* but leaves
        `.palette()` queries returning Qt's original light-mode colors.
        matplotlib's own icon loader (NavigationToolbar2QT._icon) reads
        exactly that palette's background/foreground to decide whether
        and which color to recolor Home/Pan/Save for a dark background --
        but only the one time each button icon is first built, in
        NavigationToolbar2QT.__init__; it never re-reads the palette on
        its own afterward, so _refresh_builtin_toolbar_icons() below
        re-invokes it on every theme change. The foreground is set to
        plain white here (not this app's usual DARK_TEXT) so those
        re-rendered icons come out the same exact color as this app's
        own hand-drawn zoom/calibration icons, which are also plain
        white/black rather than DARK_TEXT -- this toolbar has no visible
        text labels (icon-only buttons), so there's no legibility
        trade-off to plain white over DARK_TEXT here."""
        palette = self.nav_toolbar.palette()
        if theme == "dark":
            from theme import DARK_PANEL
            for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button):
                palette.setColor(role, QColor(DARK_PANEL))
            for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
                palette.setColor(role, QColor("white"))
        else:
            palette = QPalette()
        self.nav_toolbar.setPalette(palette)

    def _refresh_builtin_toolbar_icons(self):
        """Re-renders matplotlib's own Home/Pan/Save toolbar icons for the
        current theme. NavigationToolbar2QT._icon() only reads the
        toolbar's QPalette once, when each button is first created in
        __init__ -- unlike this app's own zoom/calibration icons (see
        _refresh_zoom_icons/_refresh_calibration_icons below), matplotlib
        never re-renders them on its own, so without this they'd stay
        whatever color matched the palette at toolbar-construction time
        and never follow later theme toggles."""
        image_files = {
            text: image_file
            for text, _tooltip, image_file, _callback in self.nav_toolbar.toolitems
            if text is not None
        }
        for action in self.nav_toolbar.actions():
            image_file = image_files.get(action.text())
            if image_file is not None:
                action.setIcon(NavigationToolbar2QT._icon(self.nav_toolbar, image_file + ".png"))

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
            "Spectrum files (*.txt *.spe *.spk *.n42);;Text files (*.txt);;"
            "SPE files (*.spe);;SPK files (*.spk);;N42 files (*.n42);;All files (*)",
        )
        if paths:
            self._load_files(paths)

    def _open_matrix_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Matrix", self.settings.last_folder(), "Matrix files (*.mtx);;All files (*)"
        )
        if path:
            # Decoding a real matrix (e.g. 8192x8192 lc-compressed) takes
            # several seconds on the GUI thread -- without feedback the app
            # appears to hang. The status message is posted before the wait
            # cursor and flushed with processEvents() so it actually paints
            # before the blocking load starts (the event loop can't repaint
            # once _open_matrix_panel is on the stack).
            self.statusBar().showMessage("Loading matrix...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                self._open_matrix_panel(path)
            except ParseError as exc:
                QMessageBox.warning(self, "Could not open matrix", str(exc))
            else:
                self.settings.set_last_folder(os.path.dirname(path))
            finally:
                QApplication.restoreOverrideCursor()
                self.statusBar().clearMessage()

    def _open_matrix_panel(self, path):
        panel = MatrixPanel(self, path)
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
            else:
                data = load_histogram(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"

        if calibration is not None and not self._calibration_active:
            self._apply_calibration_change(calibration, True)

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

    def _plot_data(self, preserve_view=False, xlim_override=None):
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
            if xlim_override is not None:
                xlim = xlim_override
            elif saved_xlim is not None:
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
        self._update_operations_availability()
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
        self._update_spectrum_list()
        self._plot_data()

    def _close_active_spectrum(self):
        active = next((s for s in self.spectra if s.active), None)
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
        active = next((s for s in self.spectra if s.active), None)
        available = active is not None and active.visible
        self.fit_button.setEnabled(available and self.fit_controller.state.ready_to_fit())
        self.integrate_button.setEnabled(available and self.fit_controller.state.ready_to_integrate())

    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        self.close_spectrum_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
        self.add_action.setEnabled(len(self.spectra) >= 2)
        self.subtract_action.setEnabled(len(self.spectra) >= 2)

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
        half_width = abs(xlim[1] - xlim[0]) / 2 * factor
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
        # A negative-b calibration makes channel_to_display decreasing,
        # so the axis may currently be "inverted" (xlim[0] > xlim[1], a
        # legitimate matplotlib feature -- see _plot_data/_show_full_spectrum).
        # Preserve that orientation on write-back rather than always
        # writing ascending order, which would flip the axis direction
        # on every zoom.
        new_xlim = (new_lo, new_hi) if xlim[0] <= xlim[1] else (new_hi, new_lo)
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
