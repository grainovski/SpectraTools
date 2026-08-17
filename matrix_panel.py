import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from calibration_dialog import CalibrationDialog
from fit_mode import FitModeController
from matrix_cut import compute_projection
from mtx_io import load_mtx
from spectrum import LIGHT_COLOR_CYCLE, LoadedSpectrum, panned_xlim
from theme import refresh_builtin_toolbar_icons, style_axes, style_nav_toolbar_palette

CUT_REGION_COLOR = "tab:red"
CUT_REGION_ALPHA = 0.25
BG_REGION_COLOR = "tab:green"
BG_REGION_ALPHA = 0.3
ZOOM_FACTOR = 1.5


class MatrixCutState:
    """Tracks in-progress cut-region/background-region marks on a
    matrix's working projection -- exactly one cut region, any number
    of background regions (unlike fit_mode.py's FitModeState, which
    caps background regions at 2; TV's own philosophy is "the more
    background you mark, the better"). Two-click pairing, modeled on
    FitModeState (fit_mode.py:29-159)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cut_region = None
        self._pending_cut_click = None
        self.bg_regions = []
        self._pending_bg_click = None

    def add_cut_click(self, x):
        if self._pending_cut_click is None:
            self._pending_cut_click = x
            return False
        lo, hi = sorted((self._pending_cut_click, x))
        self.cut_region = (lo, hi)
        self._pending_cut_click = None
        return True

    def add_bg_click(self, x):
        if self._pending_bg_click is None:
            self._pending_bg_click = x
            return False
        lo, hi = sorted((self._pending_bg_click, x))
        self.bg_regions.append((lo, hi))
        self._pending_bg_click = None
        return True

    def clear_cut(self):
        self.cut_region = None
        self._pending_cut_click = None

    def clear_bg(self):
        self.bg_regions = []
        self._pending_bg_click = None


class MatrixCutController(QObject):
    """Held-key + click marker placement for the matrix panel's
    projection view -- hold C for the cut region, hold G for a
    background region (not B -- reserved for the main window's own
    fit-marking system, which a later task adds to this same window).
    Modeled on FitModeController's eventFilter/_held_key mechanism
    (fit_mode.py:382-428), simplified to two mark types instead of
    three."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.state = MatrixCutState()
        self._held_key = None
        self._artists = []
        panel.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        panel.canvas.installEventFilter(self)
        panel.canvas.mpl_connect("figure_enter_event", lambda event: panel.canvas.setFocus())
        panel.canvas.mpl_connect("button_press_event", self.on_click)

    def eventFilter(self, obj, event):
        if obj is self.panel.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_C:
                    self._held_key = "cut"
                elif event.key() == Qt.Key.Key_G:
                    self._held_key = "gate_bg"
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                self._held_key = None
        return False

    def on_click(self, event):
        if event.inaxes != self.panel.axes or event.xdata is None or event.button != 1:
            return
        channel_x = self.panel.display_to_channel(event.xdata)
        if self._held_key == "cut":
            self.state.add_cut_click(channel_x)
        elif self._held_key == "gate_bg":
            self.state.add_bg_click(channel_x)
        else:
            return
        self._redraw_markers()
        self.panel._update_activate_button()

    def clear(self, redraw=True):
        """Drops every cut/background mark. `redraw=False` skips the
        canvas draw for callers that are about to replot anyway --
        _on_axis_changed swaps the projection underneath us, so the
        default draw here would paint the OLD projection's axes for one
        frame before _plot_data() immediately clears and replaces it.
        The Clear Marks button, by contrast, has nothing following it
        and needs the draw."""
        self.state.reset()
        self._redraw_markers(redraw=redraw)
        self.panel._update_activate_button()

    def _redraw_markers(self, redraw=True):
        for artist in self._artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._artists = []

        axes = self.panel.axes
        to_display = self.panel.channel_to_display
        state = self.state

        if state._pending_cut_click is not None:
            self._artists.append(
                axes.axvline(
                    to_display(state._pending_cut_click),
                    color=CUT_REGION_COLOR, linestyle="--", linewidth=1,
                )
            )
        if state.cut_region is not None:
            lo, hi = state.cut_region
            self._artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=CUT_REGION_COLOR, alpha=CUT_REGION_ALPHA
                )
            )

        if state._pending_bg_click is not None:
            self._artists.append(
                axes.axvline(
                    to_display(state._pending_bg_click),
                    color=BG_REGION_COLOR, linestyle="--", linewidth=1,
                )
            )
        for lo, hi in state.bg_regions:
            self._artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
            )
        if redraw:
            self.panel.canvas.draw()


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

    def __init__(self, main_window, path, matrix=None):
        super().__init__()
        self.main_window = main_window
        self.path = path
        # `matrix` lets a caller hand in an already-decoded matrix
        # instead of paying for the decode here, on the GUI thread.
        # main_window's Open Matrix flow does exactly that, loading in a
        # worker thread so the window keeps painting; passing None keeps
        # the original synchronous behaviour, which every direct
        # construction (including the tests) still relies on.
        self.matrix = load_mtx(path) if matrix is None else matrix
        self.projections = {
            "x": compute_projection(self.matrix, "x"),
            "y": compute_projection(self.matrix, "y"),
        }
        self.working_axis = "x"
        self._rebuild_spectra()
        # A list, not a single attribute -- MatrixHeatmapWindow is a
        # parentless, non-modal top-level window, so under Qt's ownership
        # rules the Python-side reference is what keeps it alive. A
        # single attribute would get overwritten (and the previous
        # window's only reference dropped, making it eligible for
        # garbage collection and liable to vanish) if "Show Heatmap..."
        # is clicked again while an earlier heatmap window is still open.
        self._heatmap_windows = []

        self.setWindowTitle(f"Matrix -- {os.path.basename(path)}")
        self.resize(800, 500)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        from main_window import _TrimmedNavigationToolbar  # local: avoids a circular import with main_window.py

        self.nav_toolbar = _TrimmedNavigationToolbar(self.canvas, self, coordinates=False)
        self.nav_toolbar.addSeparator()

        self.zoom_in_action = QAction("Zoom In (+)", self)
        self.zoom_in_action.setShortcut("Ctrl+=")
        self.zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out (-)", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_out_action)

        self.full_view_action = QAction("Full View", self)
        self.full_view_action.setShortcut("Ctrl+0")
        self.full_view_action.triggered.connect(self._show_full_view)
        self.nav_toolbar.addAction(self.full_view_action)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._pan_last_px = None
        self.canvas.mpl_connect("button_press_event", self._on_pan_press)
        self.canvas.mpl_connect("button_release_event", self._on_pan_release)
        self.canvas.mpl_connect("motion_notify_event", self._pan_to)

        self.axis_selector = QComboBox()
        self.axis_selector.addItem("X projection", "x")
        self.axis_selector.addItem("Y projection", "y")
        self.axis_selector.currentIndexChanged.connect(self._on_axis_changed)

        self.heatmap_button = QPushButton("Show Heatmap...")
        self.heatmap_button.clicked.connect(self._open_heatmap)

        self.activate_cut_button = QPushButton("Activate Cut")
        self.activate_cut_button.setEnabled(False)
        self.activate_cut_button.setShortcut("Ctrl+Alt+C")
        self.activate_cut_button.clicked.connect(self._activate_cut)

        self.clear_marks_button = QPushButton("Clear Marks")
        self.clear_marks_button.clicked.connect(self._clear_marks)

        self.calibrate_button = QPushButton("Calibrate...")
        self.calibrate_button.setShortcut("Ctrl+L")
        self.calibrate_button.clicked.connect(self._open_calibration_dialog)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Working on:"))
        top_bar.addWidget(self.axis_selector)
        top_bar.addStretch()
        top_bar.addWidget(self.clear_marks_button)
        top_bar.addWidget(self.calibrate_button)
        top_bar.addWidget(self.activate_cut_button)
        top_bar.addWidget(self.heatmap_button)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self.cut_controller = MatrixCutController(self)

        self.fit_controller = FitModeController(self)
        self.fit_controller.build_results_panel()
        self.fit_controller.build_parameters_panel()

        self.fit_button = QAction("Fit", self)
        self.fit_button.setShortcut("Ctrl+F")
        self.fit_button.setEnabled(False)
        self.fit_button.triggered.connect(self.fit_controller.run_fit)
        self.addAction(self.fit_button)

        self.clear_fit_button = QAction("Clear", self)
        self.clear_fit_button.setShortcut("Ctrl+C")
        self.clear_fit_button.triggered.connect(self._clear_everything)
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

        # FitModeController.__init__ only wires up the canvas's keyboard
        # (held-key) event filter -- it deliberately leaves connecting its
        # on_click to the canvas's button_press_event to the owning
        # window, exactly as main_window.py itself does
        # (self.canvas.mpl_connect("button_press_event", self._on_canvas_click),
        # where _on_canvas_click just delegates to fit_controller.on_click).
        # Without this, b/r/p marking would silently do nothing: MatrixCutController's
        # own on_click (connected above, in its own __init__) only handles its
        # "cut"/"gate_bg" held-key states and returns immediately for anything
        # else, so nothing else on this canvas would ever route a click to
        # fit_controller.on_click.
        self.canvas.mpl_connect("button_press_event", self.fit_controller.on_click)

        self._plot_data()
        # _plot_data() above already styles self.axes correctly via
        # self.main_window._theme (see style_axes call in _plot_data),
        # but this panel's own nav_toolbar has never had its palette or
        # built-in icons touched at all up to this point -- without this
        # call, a panel opened while the app is ALREADY in dark theme
        # would still start out with light-mode Home/Pan/Save icons
        # (Qt's untouched default palette) until the next theme toggle
        # happened to sweep it up. See _refresh_theme below. (It harmlessly
        # re-styles self.axes and redraws a second time too -- not split
        # into a toolbar-only path to avoid a third code path for what's
        # a cheap, one-time-at-construction cost.)
        self._refresh_theme()

    def _rebuild_spectra(self):
        """Wraps the current working-axis projection as a single
        LoadedSpectrum so FitModeController (duck-typed against this
        window, see the fit-integration task) can operate on it exactly
        as it does on MainWindow's own spectra list. Always exactly one
        entry, always active and visible -- there is no concept of
        multiple or hidden "spectra" in this window.

        color=LIGHT_COLOR_CYCLE[0], not the matplotlib named color
        "tab:blue" this used to read (same rendered color either way --
        "tab:blue" IS "#1f77b4", matplotlib's tab10 palette entry 0,
        verbatim LIGHT_COLOR_CYCLE[0]) -- FitModeController.draw_committed_fits
        unconditionally derives its fit/background-line colors from
        spectrum.color via theme.fit_drawing_colors, which parses it as a
        strict "#RRGGBB" hex string (_hex_to_rgb01) and raises ValueError
        on a named color. That call was never reached before fit-mode
        integration (Task 7) since draw_committed_fits was never invoked
        for this panel until now -- a plain color= tab:blue value was
        harmless as long as it only ever reached matplotlib's own
        axes.plot(color=...), which accepts named colors fine."""
        data = self.projections[self.working_axis]
        # .path must stay path-shaped, not a free-form label: fit_export.auto_log_path
        # and fit_mode.py's "Export Fit Report" default filename both derive a stem via
        # os.path.splitext(os.path.basename(...)) -- a label like "gg.mtx x projection"
        # gets misparsed as stem "gg" with extension ".mtx x projection", losing the
        # axis entirely, and its empty os.path.dirname sends auto-log writes to the
        # process's cwd instead of next to this real matrix file.
        root, ext = os.path.splitext(self.path)
        path = f"{root}_{self.working_axis}_projection{ext}"
        spectrum = LoadedSpectrum(path, data, color=LIGHT_COLOR_CYCLE[0])
        spectrum.active = True
        self.spectra = [spectrum]

    @property
    def _calibration(self):
        return self.main_window._calibration

    @_calibration.setter
    def _calibration(self, value):
        self.main_window._calibration = value

    @property
    def _calibration_active(self):
        return self.main_window._calibration_active

    @_calibration_active.setter
    def _calibration_active(self, value):
        self.main_window._calibration_active = value

    @property
    def _theme(self):
        # FitModeController.draw_committed_fits reads self.main_window._theme
        # via getattr(..., "light") -- without this property, that always
        # silently falls back to "light" when "main_window" is actually this
        # panel (MatrixPanel has no _theme attribute of its own), regardless
        # of the app's real active theme. Delegating, same pattern as
        # _calibration/_calibration_active above, keeps fit/integration
        # overlay colors correctly matched to the real theme in dark mode.
        return self.main_window._theme

    def _refresh_theme(self):
        """Re-applies the current theme to this panel's own plot axes
        and its own nav_toolbar's palette/built-in icons, then redraws.
        Called from main_window._apply_theme's loop over every open
        MatrixPanel, the same way main_window._apply_calibration_change
        already loops over main_window._matrix_panels to push a
        calibration change out to each one -- needed because this panel
        builds its own separate Figure/Axes and its own separate
        NavigationToolbar2QT, entirely independent of MainWindow's own,
        so neither one follows a theme change made anywhere else on its
        own (confirmed empirically: both this toolbar's built-in icon
        bytes and this panel's axes facecolor stayed byte-for-byte
        unchanged across a MainWindow theme toggle before this fix).

        style_axes() + a bare redraw here (rather than a full
        self._plot_data()) mirrors exactly how MainWindow's own
        _apply_theme keeps its own axes in sync -- _plot_data-driven
        extras like recoloring the plotted trace or resetting the
        nav_toolbar's zoom history are handled separately, by
        _apply_theme's caller (_on_theme_toggled), only for
        MainWindow's own canvas/spectra. This panel's single synthetic
        spectrum is deliberately NOT theme-recolored either way (see
        _rebuild_spectra's own docstring above), so there is nothing
        equivalent for this panel to do beyond restyling the axes."""
        style_nav_toolbar_palette(self.nav_toolbar, self._theme)
        refresh_builtin_toolbar_icons(self.nav_toolbar)
        style_axes(self.axes, self._theme)
        self.canvas.draw()

    def channel_to_display(self, channel):
        if not self._calibration_active or self._calibration is None:
            return channel
        return self._calibration.apply(channel)

    def display_to_channel(self, display_x):
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
        # Delegates to main_window's own _apply_calibration_change rather
        # than reimplementing a subset of it. That method already: (a)
        # round-trips ITS OWN view through channel space computed from
        # the OLD calibration before overwriting it, so the same
        # detector region stays in view there under the new calibration;
        # (b) updates its own toolbar/menu calibration indicators and
        # refreshes its Fit Parameters panel; (c) redraws every panel in
        # main_window._matrix_panels -- which includes this one, in real
        # usage (MatrixPanel is only ever constructed via
        # main_window._open_matrix_panel, which registers it there).
        #
        # But that delegated call knows nothing about THIS panel's own
        # zoom state, so this panel's own view is captured and
        # round-tripped here, following the exact same pattern -- and for
        # the exact same reason: channel_bounds MUST be computed from
        # old_xlim BEFORE main_window._apply_calibration_change runs,
        # since that call is what overwrites the shared calibration state
        # (self._calibration/_calibration_active are properties
        # delegating straight to main_window's) that display_to_channel
        # reads. Computing channel_bounds after that call would silently
        # interpret old_xlim's numbers under the NEW calibration instead
        # of the OLD one they actually came from -- the exact bug already
        # found and fixed once in Task 3's code review (see
        # main_window._apply_calibration_change's own docstring/comment).
        # The final self._plot_data(xlim_override=...) is a deliberate,
        # cheap belt-and-suspenders redraw of this panel's own canvas at
        # the round-tripped bounds, not reliant on the
        # main_window._matrix_panels registration loop for correctness.
        old_xlim = self.axes.get_xlim()
        channel_bounds = (self.display_to_channel(old_xlim[0]), self.display_to_channel(old_xlim[1]))
        self.main_window._apply_calibration_change(new_calibration, new_active)
        new_xlim = (self.channel_to_display(channel_bounds[0]), self.channel_to_display(channel_bounds[1]))
        self._plot_data(xlim_override=new_xlim)
        self.fit_controller.refresh_parameters_panel_calibration()

    def _on_scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)

    def _zoom_x(self, factor, center=None):
        spectrum = self.spectra[0]
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = abs(xlim[1] - xlim[0]) / 2 * factor
        max_channel = len(spectrum.data) - 1
        display_lo = self.channel_to_display(0)
        display_hi = self.channel_to_display(max_channel)
        display_lo, display_hi = min(display_lo, display_hi), max(display_lo, display_hi)
        new_lo = max(display_lo, center - half_width)
        new_hi = min(display_hi, center + half_width)
        if new_hi <= new_lo:
            new_hi = min(display_hi, new_lo + 1)
        new_xlim = (new_lo, new_hi) if xlim[0] <= xlim[1] else (new_hi, new_lo)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        # draw_idle for the same reason as main_window._zoom_x: a mouse
        # wheel emits events faster than a full render completes, so
        # drawing synchronously makes the view lag the wheel by one full
        # render per tick. Coalesced into one instead.
        self.canvas.draw_idle()
        self.nav_toolbar.push_current()

    def _on_pan_press(self, event):
        """Right-button drag-pan, mirroring main_window's. Button 1 is
        left alone -- it places cut/background marks."""
        if event.button != 3 or event.inaxes != self.axes or event.x is None:
            return
        self._pan_last_px = event.x

    def _on_pan_release(self, event):
        if self._pan_last_px is None:
            return
        self._pan_last_px = None
        self.nav_toolbar.push_current()

    def _pan_to(self, event):
        """See main_window._pan_to for why this works from the pixel
        delta between motion events rather than from the data coordinate
        the drag started at."""
        if self._pan_last_px is None or event.x is None:
            return
        width_px = self.axes.bbox.width
        if not width_px:
            return
        spectrum = self.spectra[0]
        xlim = self.axes.get_xlim()
        delta = -(event.x - self._pan_last_px) * (xlim[1] - xlim[0]) / width_px
        self._pan_last_px = event.x
        new_xlim = panned_xlim(
            xlim, delta,
            self.channel_to_display(0),
            self.channel_to_display(len(spectrum.data) - 1),
        )
        if new_xlim == xlim:
            return
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw_idle()

    def _show_full_view(self):
        spectrum = self.spectra[0]
        max_channel = len(spectrum.data) - 1
        full_xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _autoscale_y(self, xlim):
        spectrum = self.spectra[0]
        channel_lo = self.display_to_channel(xlim[0])
        channel_hi = self.display_to_channel(xlim[1])
        channel_lo, channel_hi = min(channel_lo, channel_hi), max(channel_lo, channel_hi)
        lo_bound = max(0, int(np.floor(channel_lo)))
        hi_bound = min(int(np.ceil(channel_hi)) + 1, len(spectrum.data))
        if lo_bound >= hi_bound:
            return
        window = spectrum.data[lo_bound:hi_bound]
        y_min = float(np.min(window))
        y_max = float(np.max(window))
        margin = (y_max - y_min) * 0.05 or 1.0
        self.axes.set_ylim(y_min - margin, y_max + margin)

    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self._rebuild_spectra()
        # redraw=False: _plot_data() below clears the axes and repaints
        # from scratch, so a draw here would only render the outgoing
        # projection for one discarded frame.
        self.cut_controller.clear(redraw=False)
        self.fit_controller.reset_marks()
        self._plot_data()

    def _plot_data(self, preserve_view=False, xlim_override=None):
        saved_xlim = self.axes.get_xlim() if preserve_view else None
        self.axes.clear()
        style_axes(self.axes, self.main_window._theme)
        spectrum = self.spectra[0]
        channels = np.arange(len(spectrum.data))
        x = self.channel_to_display(channels)
        self.axes.plot(x, spectrum.data, drawstyle="steps-mid", linewidth=0.8, color=spectrum.color)
        # Resolved before drawing so off-view fits can be skipped, then
        # applied below in its original place -- see main_window's
        # _plot_data for why it cannot simply be read back from the axes
        # inside the draw call.
        if xlim_override is not None:
            xlim = xlim_override
        elif saved_xlim is not None:
            xlim = saved_xlim
        else:
            xlim = (self.channel_to_display(0), self.channel_to_display(len(spectrum.data) - 1))
        self.fit_controller.draw_committed_fits(spectrum, view_xlim=xlim)
        self.fit_controller.update_results_list()
        self.axes.set_xlabel("Energy (keV)" if self._calibration_active else f"{self.working_axis.upper()} channel")
        self.axes.set_ylabel("Counts")
        self.axes.set_xlim(xlim)
        self._autoscale_y(xlim)
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path,
        # matching main_window.py's own _plot_data -- without this the
        # Home button wouldn't know about this view, and after an axis
        # switch or calibration change would reapply a now-nonsensical
        # old view (a different projection's or a different calibration's
        # raw xlim/ylim numbers) instead of resetting the history.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
        self._update_fit_mode_availability()

    def _update_fit_mode_availability(self):
        self.fit_button.setEnabled(self.fit_controller.state.ready_to_fit())
        self.integrate_button.setEnabled(self.fit_controller.state.ready_to_integrate())

    def _clear_marks(self):
        self.cut_controller.clear()

    def _clear_everything(self):
        """Ctrl+C in a matrix panel.

        Does what the Clear Marks button does to the cut and background
        marks, AND what Ctrl+C has always done to fit marks and committed
        fits. Previously it only did the latter, so cut/background marks
        survived it -- and because a replot drops their artists without
        redrawing them, they could vanish from the plot while the state
        still held them and Activate Cut stayed enabled. One key now
        leaves the panel genuinely clear.

        redraw=False on the cut clear because fit_controller.clear()
        replots the whole panel immediately afterwards; drawing twice
        would only paint a frame nobody sees.
        """
        self.cut_controller.clear(redraw=False)
        self.fit_controller.clear()

    def _update_activate_button(self):
        self.activate_cut_button.setEnabled(self.cut_controller.state.cut_region is not None)

    def _activate_cut(self):
        from matrix_cut import compute_cut_with_variance

        state = self.cut_controller.state
        # The variance travels with the cut so a fit on it is weighted by
        # the real uncertainty rather than sqrt(counts) -- see
        # compute_cut_with_variance for why the two differ.
        result, variance = compute_cut_with_variance(
            self.matrix, self.working_axis, state.cut_region, state.bg_regions
        )
        # Path-shaped, not a free-form label -- same rationale and same
        # fix shape as _rebuild_spectra's own path construction above.
        # The region bounds used to be embedded as raw .1f floats right
        # after a literal ".mtx" (e.g. "gpff.mtx y cut [3004.4, 3859.9]"),
        # and fit_export.auto_log_path's os.path.splitext(os.path.basename(...))
        # truncates at the LAST '.' in that string -- which was always
        # one of the region bounds' own decimal points (confirmed by two
        # stray "..._3859_fits.jsonl"/"..._3766_fits.jsonl" files this
        # bug left in the repo root). Merely rounding the bounds to drop
        # their decimal points is not enough by itself: with no dot left
        # in the region numbers, self.path's OWN ".mtx" dot becomes the
        # new last dot, which would make EVERY cut on the same matrix
        # collapse onto the identical auto-log file (verified:
        # splitext("gpff.mtx y cut [3004, 3860]") -> stem "gpff",
        # silently dropping the entire cut descriptor and colliding
        # different cuts' fit logs together). Keeping the real extension
        # at the very end of the constructed path -- exactly like
        # _rebuild_spectra -- avoids both failure modes: the only dot
        # left in the basename is the genuine trailing extension, so
        # splitext parses it correctly no matter what self.path's own
        # stem contains.
        root, ext = os.path.splitext(self.path)
        path = (
            f"{root}_{self.working_axis}_cut_"
            f"{round(state.cut_region[0])}_{round(state.cut_region[1])}{ext}"
        )
        self.main_window._add_combined_spectrum(path, result, variance=variance)

    def _open_heatmap(self):
        from matrix_heatmap import MatrixHeatmapWindow

        window = MatrixHeatmapWindow(self.matrix, self.path, self.main_window._theme, self)
        self._heatmap_windows.append(window)
        window.show()

    def showEvent(self, event):
        super().showEvent(event)
        # Same rationale as MainWindow.showEvent: hold-C/hold-G marking
        # depends on the canvas holding keyboard focus, and the top bar's
        # combo box/buttons are focusable widgets that can claim initial
        # focus on some window managers before the user has ever hovered
        # the canvas (figure_enter_event alone wouldn't cover that case).
        self.canvas.setFocus()

    def _on_activated(self):
        self.main_window.setEnabled(False)
        # Generalizes MainWindow._on_activated's own loop to the
        # multi-panel case: with two or more matrix panels open,
        # activating this one must disable every OTHER panel too, not
        # just MainWindow -- otherwise two panels could end up enabled
        # simultaneously, breaking the "activating one disables the
        # other" guarantee for anything beyond exactly one panel.
        for panel in self.main_window._matrix_panels:
            if panel is not self:
                panel.setEnabled(False)
        self.setEnabled(True)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowActivate:
            self._on_activated()

    def closeEvent(self, event):
        # Same parentless-top-level-window concern as MainWindow.closeEvent:
        # a MatrixHeatmapWindow left open would keep the app from quitting
        # even after this panel (and MainWindow) are both closed.
        for window in list(self._heatmap_windows):
            window.close()
        super().closeEvent(event)
        if self in self.main_window._matrix_panels:
            self.main_window._matrix_panels.remove(self)
        # Only fall back to re-enabling MainWindow if nothing else in the
        # exclusivity group is currently enabled. With a single panel this
        # is always true (MainWindow was the one disabled), but with two
        # or more panels open, closing a PANEL THAT WASN'T THE ACTIVE ONE
        # must not steal focus-exclusivity away from whichever window
        # (MainWindow or another panel) is actually still active.
        others_enabled = self.main_window.isEnabled() or any(
            panel.isEnabled() for panel in self.main_window._matrix_panels
        )
        if not others_enabled:
            self.main_window.setEnabled(True)

        # Release the decoded matrix explicitly.
        #
        # Closing a window does not destroy it. This panel is a
        # parentless top-level widget, so Qt never owned it, and the Qt
        # signal connections made in __init__ capture `self` in closures
        # that outlive the close -- the object stays reachable and its
        # matrix stays resident. Measured before this: opening and
        # closing the same 8192x8192 matrix three times left three live
        # panels holding 1.6 GB between them, growing without bound for
        # as long as the session lasts. The _matrix_panels bookkeeping
        # above was already correct; it is the payload that lingered.
        #
        # Dropping the references here rather than relying on the widget
        # being collected: whether the panel itself is freed depends on
        # Qt/Python ownership details that are easy to regress, while an
        # explicit release frees the memory that actually matters either
        # way. Safe because every reader of self.matrix
        # (_rebuild_spectra, _activate_cut, the heatmap) is a user action
        # on an OPEN panel, and this panel is hidden and de-registered by
        # the time we get here. The heatmap windows closed above hold
        # their own reference to the same array, which is why they are
        # closed first.
        self.matrix = None
        self.projections = {}
