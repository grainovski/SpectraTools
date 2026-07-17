import os
import sys
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QFileDialog, QMenu, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget,
)

import fit_export
from peak_fit import (
    FWHM_FACTOR, FitError, IntegrationResult, fit_peaks, fit_result_values_by_name,
    hypermet_left_tail, integrate_region, parameter_names,
)
from theme import fit_drawing_colors

BG_REGION_CAP = 2

BG_REGION_COLOR = "tab:green"
BG_REGION_ALPHA = 0.3
FIT_REGION_COLOR = "tab:blue"
FIT_REGION_ALPHA = 0.2


class FitModeState:
    """Tracks in-progress background/fit-region/peak marks made via
    independent b/r/p click actions -- order-free, no Qt/matplotlib
    dependency. Each of add_bg_click/add_fit_click counts its own
    pending point independently, so interleaving a different key's
    clicks never disturbs an in-progress pair."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bg_regions = []
        self.pending_bg_click = None
        self.fit_region = None
        self.pending_fit_click = None
        self.peak_positions = []

    def add_bg_click(self, x):
        """Returns the completed (lo, hi) region if this click
        completed a pair, else None (this click becomes the pending
        first point). A 3rd completed pair evicts the oldest region."""
        if self.pending_bg_click is None:
            self.pending_bg_click = x
            return None
        region = (min(self.pending_bg_click, x), max(self.pending_bg_click, x))
        self.pending_bg_click = None
        self.bg_regions.append(region)
        if len(self.bg_regions) > BG_REGION_CAP:
            self.bg_regions.pop(0)
        return region

    def add_fit_click(self, x):
        """Returns the completed (lo, hi) region if this click
        completed a pair, else None. A newly completed pair always
        replaces any existing fit region and drops any already-marked
        peak position that falls outside the new region -- marks persist
        across a committed fit (so re-fitting or tweaking the same
        region doesn't require re-marking peaks), but a peak position
        left over from a now-abandoned region is stale, not intentional,
        and silently breaks the next fit_peaks() call (an out-of-region
        initial guess) if carried along uncleaned."""
        if self.pending_fit_click is None:
            self.pending_fit_click = x
            return None
        region = (min(self.pending_fit_click, x), max(self.pending_fit_click, x))
        self.pending_fit_click = None
        self.fit_region = region
        lo, hi = region
        self.peak_positions = [p for p in self.peak_positions if lo <= p <= hi]
        return region

    def toggle_peak(self, x, proximity):
        """Adds a peak at x, or removes an existing one within
        `proximity` of x. Returns ("added", x), ("removed", old_x), or
        None if there's no fit region yet or x falls outside it."""
        if self.fit_region is None:
            return None
        lo, hi = self.fit_region
        if not (lo <= x <= hi):
            return None
        for i, pos in enumerate(self.peak_positions):
            if abs(pos - x) <= proximity:
                del self.peak_positions[i]
                return ("removed", pos)
        self.peak_positions.append(x)
        return ("added", x)

    def ready_to_fit(self):
        return (
            len(self.bg_regions) == BG_REGION_CAP
            and self.fit_region is not None
            and len(self.peak_positions) > 0
        )

    def ready_to_integrate(self):
        return len(self.bg_regions) == BG_REGION_CAP and self.fit_region is not None

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first. Only valid
        once both regions exist."""
        a, b = self.bg_regions
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)


PEAK_CLICK_PIXEL_PROXIMITY = 8

_KEY_TO_MARK_TYPE = {
    Qt.Key.Key_B: "b",
    Qt.Key.Key_R: "r",
    Qt.Key.Key_P: "p",
}

# Windows "Scan Code Set 1" values for the physical B/R/P key positions.
# A scan code identifies which physical key was pressed, assigned by the
# keyboard hardware before any software layout translation -- unlike
# QKeyEvent.key(), which some non-Latin Windows input languages (e.g.
# Bulgarian) remap away from Qt.Key.Key_B/R/P entirely for those keys,
# breaking b/r/p marking outright for anyone using that layout even
# though their physical keyboard still has "B"/"R"/"P" printed on it.
# Used only as a fallback when the primary Qt.Key lookup misses.
_WINDOWS_SCAN_CODE_TO_MARK_TYPE = {
    0x30: "b",
    0x13: "r",
    0x19: "p",
}

_MARK_TYPE_LABEL = {
    "b": "background region",
    "r": "fit region",
    "p": "peak",
}


def _mark_type_for_key_event(event):
    """Resolves a key event to "b"/"r"/"p" (or None), preferring the
    normal Qt.Key lookup and falling back to a Windows-only physical
    scan-code match when the active keyboard layout has remapped the
    key away from what Qt expects (see _WINDOWS_SCAN_CODE_TO_MARK_TYPE)."""
    mark_type = _KEY_TO_MARK_TYPE.get(event.key())
    if mark_type is not None:
        return mark_type
    if sys.platform.startswith("win"):
        return _WINDOWS_SCAN_CODE_TO_MARK_TYPE.get(event.nativeScanCode())
    return None


def _parameter_label(name):
    """Human-readable row label for a canonical parameter name from
    peak_fit.parameter_names() -- e.g. "amp_0" -> "Peak 1 amplitude".
    Sigma-family names are labeled as FWHM: the panel displays and
    accepts FWHM, converting to/from the fitting engine's internal
    sigma at the UI boundary (see _display_value/_panel_value_to_internal
    below) -- peak_fit.py's own parameter naming and optimizer are
    untouched."""
    if name == "sigma":
        return "Shared FWHM"
    if name == "tail_fraction":
        return "Tail fraction (r)"
    if name == "tail_beta":
        return "Tail beta (β)"
    prefix, index = name.rsplit("_", 1)
    peak_num = int(index) + 1
    kind = {"amp": "amplitude", "pos": "position", "sigma": "FWHM"}[prefix]
    return f"Peak {peak_num} {kind}"


def _integration_tooltip(result):
    """Full gross/background/net breakdown for an IntegrationResult's Fit
    Results row tooltip -- mirrors the level of detail the existing
    left-tail-info tooltip gives for a Gaussian fit."""
    lines = []
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        area = getattr(result, f"{prefix}_area")
        area_err = getattr(result, f"{prefix}_area_err")
        centroid = getattr(result, f"{prefix}_centroid")
        centroid_err = getattr(result, f"{prefix}_centroid_err")
        fwhm = getattr(result, f"{prefix}_fwhm")
        fwhm_err = getattr(result, f"{prefix}_fwhm_err")
        skewness = getattr(result, f"{prefix}_skewness")
        skewness_err = getattr(result, f"{prefix}_skewness_err")
        lines.append(
            f"{label}: area={area:.1f}±{area_err:.1f}, "
            f"centroid={centroid:.2f}±{centroid_err:.2f}, "
            f"FWHM={fwhm:.2f}±{fwhm_err:.2f}, "
            f"skewness={skewness:.3g}±{skewness_err:.3g}"
        )
    return "\n".join(lines)


def _is_sigma_name(name):
    return name == "sigma" or name.startswith("sigma_")


def _display_value(name, value):
    """Converts a canonical parameter's internal value to what the Fit
    Parameters panel displays -- sigma-family values are shown as FWHM,
    everything else unchanged."""
    return value * FWHM_FACTOR if _is_sigma_name(name) else value


def _panel_value_to_internal(name, value):
    """Inverse of _display_value -- converts a value read back from the
    panel (FWHM for sigma-family rows) to the sigma-space value
    fit_peaks() expects."""
    return value / FWHM_FACTOR if _is_sigma_name(name) else value


class FitModeController(QObject):
    """Qt/matplotlib-facing wrapper around FitModeState: tracks which of
    b/r/p is currently held via a Qt event filter on the canvas (not
    matplotlib's own MouseEvent.key, which is documented as unreliable
    if the canvas lacked focus when the key was pressed), owns the
    in-progress marking artists, draws committed fits, and drives
    fit_peaks() when the user clicks "Fit". There is no exclusive
    "fit mode" -- marking is always available alongside normal
    pan/zoom, since b/r/p+click never collides with a plain click-drag."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = FitModeState()
        self._held_key = None
        self._progress_artists = []
        self._status_message_until = 0.0
        self._parameter_names_shown = []
        self._results_row_fit_index = []

        canvas = main_window.canvas
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        canvas.mpl_connect("figure_enter_event", lambda event: canvas.setFocus())

    def eventFilter(self, obj, event):
        if obj is self.main_window.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                mark_type = _mark_type_for_key_event(event)
                if mark_type is not None:
                    self._held_key = mark_type
                    self._show_status_message(
                        f"Marking {_MARK_TYPE_LABEL[mark_type]}: click to place", 60000
                    )
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                mark_type = _mark_type_for_key_event(event)
                if mark_type is not None and self._held_key == mark_type:
                    self._held_key = None
                    # The "Marking ...: click to place" hint above is shown
                    # with a 60s duration so it survives however long the
                    # key is held -- but that also means the mouse-hover
                    # readout stays suppressed for up to 60s after release
                    # unless we explicitly end the suppression window here.
                    self._status_message_until = 0.0
        return False

    def _show_status_message(self, message, duration_ms):
        self.main_window.statusBar().showMessage(message, duration_ms)
        self._status_message_until = time.monotonic() + duration_ms / 1000.0

    def clear(self):
        self._clear_progress()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            for result in active.fits:
                result.visible = False
        self.main_window._plot_data(preserve_view=True)

    def _clear_progress(self):
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                # The artist may already have been invalidated by an
                # unrelated full-axes clear (main_window._plot_data(),
                # e.g. triggered by the results panel's context menu
                # while a fit was still being marked) -- matplotlib's
                # Axes.clear() sets an artist's _remove_method to None
                # for every child it had, making a later .remove() call
                # on that same (now-stale) reference raise this. Since
                # the artist is already gone from the axes either way,
                # there's nothing left to do for it here.
                pass
        self._progress_artists = []
        self.state.reset()
        self.parameters_table.setRowCount(0)
        self._parameter_names_shown = []

    def _redraw_progress(self):
        """Clears and fully rebuilds every in-progress marking artist
        from the current FitModeState fields. Trades a little redundant
        redraw work (never more than a handful of artists) for avoiding
        any incremental per-artist bookkeeping -- no risk of a stale
        artist left behind by an evicted background region or a
        removed peak."""
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._progress_artists = []

        axes = self.main_window.axes
        state = self.state

        if state.pending_bg_click is not None:
            self._progress_artists.append(
                axes.axvline(state.pending_bg_click, color="gray", linestyle="--", linewidth=1)
            )
        for region in state.bg_regions:
            self._progress_artists.append(
                axes.axvspan(*region, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA)
            )

        if state.pending_fit_click is not None:
            self._progress_artists.append(
                axes.axvline(state.pending_fit_click, color="tab:blue", linestyle="--", linewidth=1)
            )
        if state.fit_region is not None:
            self._progress_artists.append(
                axes.axvspan(*state.fit_region, color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA)
            )

        for x in state.peak_positions:
            self._progress_artists.append(
                axes.axvline(x, color="red", linestyle=":", linewidth=1)
            )

        self.main_window.canvas.draw_idle()

    def _pixel_proximity_to_data(self, event):
        """Converts the fixed PEAK_CLICK_PIXEL_PROXIMITY pixel threshold
        into a data-coordinate distance at the current zoom level, so
        the "close enough to hit an existing peak" tolerance stays
        visually consistent regardless of zoom."""
        axes = self.main_window.axes
        inverse = axes.transData.inverted()
        x0 = inverse.transform((event.x, event.y))[0]
        x1 = inverse.transform((event.x + PEAK_CLICK_PIXEL_PROXIMITY, event.y))[0]
        return abs(x1 - x0)

    def _reset_parameters_panel(self):
        """Discards the Fit Parameters panel's rows. Called whenever the
        underlying region/peak marks change, since a row's Value cell is
        read as either a fixed value or an initial-guess override for
        the *next* fit_peaks() call (fixed_params_from_panel /
        initial_guess_overrides_from_panel) -- left in place across a
        change to the marks, it silently carries a stale value from a
        now-abandoned fit configuration into the next one. This was the
        concrete cause of a fit at a newly-marked region failing
        outright: a leftover position override from the previous fit
        placed the optimizer's starting point outside the new region.
        A no-op if the panel is already empty (the common case, since
        it's normally only populated after a fit succeeds)."""
        self.parameters_table.setRowCount(0)
        self._parameter_names_shown = []

    def on_click(self, event):
        if event.inaxes != self.main_window.axes or event.xdata is None:
            return
        if event.button != 1:
            return
        key = self._held_key
        if key == "b":
            self.state.add_bg_click(event.xdata)
            self._redraw_progress()
        elif key == "r":
            completed = self.state.add_fit_click(event.xdata)
            if completed is not None:
                self._reset_parameters_panel()
            self._redraw_progress()
        elif key == "p":
            proximity = self._pixel_proximity_to_data(event)
            result = self.state.toggle_peak(event.xdata, proximity)
            if result is None:
                self._show_status_message(
                    "Mark the fit region (hold R and click twice) before marking peaks", 3000
                )
                return
            self._reset_parameters_panel()
            self._redraw_progress()
        else:
            return
        self.main_window._update_fit_mode_availability()

    def draw_committed_fits(self, spectrum):
        axes = self.main_window.axes
        # x in data coordinates, y in axes-fraction -- keeps peak labels
        # pinned near the top of the visible plot regardless of the
        # current y-axis scale (linear or log) or zoom level.
        label_transform = axes.get_xaxis_transform()
        theme = getattr(self.main_window, "_theme", "light")
        fit_color, bg_line_color = fit_drawing_colors(spectrum.color, theme)
        for result in spectrum.fits:
            if not result.visible:
                continue
            axes.axvspan(*result.left_bg_region, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA)
            axes.axvspan(*result.right_bg_region, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA)
            axes.axvspan(*result.fit_region, color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA)

            if isinstance(result, IntegrationResult):
                lo, hi = result.fit_region
                axes.plot(
                    [lo, hi], [result.background_density, result.background_density],
                    color=bg_line_color, linestyle="--", linewidth=1,
                )
                axes.annotate(
                    f"centroid={result.net_centroid:.1f}\n"
                    f"FWHM={result.net_fwhm:.1f}\n"
                    f"full={result.gross_area:.0f}\nnet={result.net_area:.0f}",
                    xy=(result.net_centroid, 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color=fit_color,
                )
                continue

            lo, hi = result.fit_region
            background_lo = result.background_slope * lo + result.background_intercept
            background_hi = result.background_slope * hi + result.background_intercept
            axes.plot([lo, hi], [background_lo, background_hi], color=bg_line_color,
                       linestyle="--", linewidth=1)

            x_dense = np.linspace(lo, hi, 200)
            total = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                if result.tail_fraction is not None:
                    total = total + peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    total = total + peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
            axes.plot(x_dense, total, color=fit_color, linewidth=1.5)

            # Peak decomposition: each peak's own contribution (background
            # + that single peak), so a multi-peak fit visually shows how
            # the total curve above decomposes into its components.
            background_dense = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                if result.tail_fraction is not None:
                    component = peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    component = peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
                axes.plot(
                    x_dense, background_dense + component,
                    color=fit_color, linewidth=0.75, linestyle="--", alpha=0.6,
                )

            for peak in result.peaks:
                axes.axvline(peak.position, color=fit_color, linestyle=":", linewidth=1)
                axes.annotate(
                    f"{peak.position:.1f}",
                    xy=(peak.position, 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color=fit_color,
                )

    def build_results_panel(self):
        mw = self.main_window
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Fit", "Position", "FWHM", "Volume", "chi^2"]
        )
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._on_results_context_menu)
        self.results_table.itemDoubleClicked.connect(self._on_result_double_clicked)

        self.results_dock = QDockWidget("Fit Results", mw)
        self.results_dock.setWidget(self.results_table)
        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.results_dock)

        self.toggle_results_panel_action = QAction("Fit Results", mw)
        self.toggle_results_panel_action.setCheckable(True)
        self.toggle_results_panel_action.setToolTip("Show/hide fit results")
        self.toggle_results_panel_action.toggled.connect(self.results_dock.setVisible)
        self.results_dock.visibilityChanged.connect(
            self.toggle_results_panel_action.setChecked
        )
        self.results_dock.setVisible(False)

        results_tab_bar = QToolBar("Fit Results Tab", mw)
        results_tab_bar.setMovable(False)
        results_tab_bar.setFloatable(False)
        results_tab_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        results_tab_bar.addAction(self.toggle_results_panel_action)
        mw.addToolBar(Qt.ToolBarArea.RightToolBarArea, results_tab_bar)

    def build_parameters_panel(self):
        mw = self.main_window
        mw.independent_widths_action = QCheckBox("Independent widths")
        mw.independent_widths_action.setToolTip(
            "Fit each peak's width independently instead of sharing one FWHM"
        )
        mw.left_tail_action = QCheckBox("Left tail")
        mw.left_tail_action.setToolTip(
            "Allow a small low-channel tail contribution to each peak's shape"
        )

        self.parameters_table = QTableWidget(0, 3)
        self.parameters_table.setHorizontalHeaderLabels(["Parameter", "Value", "Fix"])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(mw.independent_widths_action)
        layout.addWidget(mw.left_tail_action)
        layout.addWidget(self.parameters_table)

        self.parameters_dock = QDockWidget("Fit Parameters", mw)
        self.parameters_dock.setWidget(container)
        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_dock)

        self.toggle_parameters_panel_action = QAction("Fit Parameters", mw)
        self.toggle_parameters_panel_action.setCheckable(True)
        self.toggle_parameters_panel_action.setToolTip("Show/hide fixable fit parameters")
        self.toggle_parameters_panel_action.toggled.connect(self.parameters_dock.setVisible)
        self.parameters_dock.visibilityChanged.connect(
            self.toggle_parameters_panel_action.setChecked
        )
        # Unlike the Fit Results panel (pure post-fit output, fine to
        # start tucked away), this panel now also holds the
        # Independent-widths/Left-tail checkboxes -- controls needed
        # *before* marking/fitting even starts. Starting it hidden
        # would make them undiscoverable without first finding the
        # tab toggle.
        self.parameters_dock.setVisible(True)

        parameters_tab_bar = QToolBar("Fit Parameters Tab", mw)
        parameters_tab_bar.setMovable(False)
        parameters_tab_bar.setFloatable(False)
        parameters_tab_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        parameters_tab_bar.addAction(self.toggle_parameters_panel_action)
        mw.addToolBar(Qt.ToolBarArea.RightToolBarArea, parameters_tab_bar)

    def update_parameters_panel(self, names, values_by_name):
        """Rebuilds the Fit Parameters table (resetting every Fix
        checkbox) when the set of parameter names has changed since
        the last fit for these marks; otherwise updates displayed
        values in place, preserving Fix checkbox state and any
        user-edited value. Every row's Value cell is always editable:
        a checked row's edited value is read as a fixed value for the
        next fit (fixed_params_from_panel); an unchecked row's edited
        value is read as that parameter's starting guess for the next
        fit (initial_guess_overrides_from_panel) -- the parameter stays
        free, the optimizer can still move it."""
        if names != self._parameter_names_shown:
            self.parameters_table.setRowCount(0)
            self._parameter_names_shown = list(names)
            for name in names:
                row = self.parameters_table.rowCount()
                self.parameters_table.insertRow(row)

                label_item = QTableWidgetItem(_parameter_label(name))
                label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.parameters_table.setItem(row, 0, label_item)

                value_item = QTableWidgetItem(f"{_display_value(name, values_by_name[name]):.6g}")
                value_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
                )
                self.parameters_table.setItem(row, 1, value_item)

                fix_checkbox = QCheckBox()
                self.parameters_table.setCellWidget(row, 2, fix_checkbox)
        else:
            for row, name in enumerate(names):
                fix_checkbox = self.parameters_table.cellWidget(row, 2)
                if not fix_checkbox.isChecked():
                    self.parameters_table.item(row, 1).setText(
                        f"{_display_value(name, values_by_name[name]):.6g}"
                    )

    def fixed_params_from_panel(self):
        """Reads the current Fix checkboxes/values from the Fit
        Parameters panel into a {name: value} dict for the next
        fit_peaks() call. Empty when nothing is fixed (including the
        first fit for a fresh set of marks, before the panel has ever
        been populated). A sigma-family row's Value cell holds FWHM;
        this converts it back to sigma before returning. Raises
        FitError if a checked row's Value cell isn't a valid number."""
        fixed = {}
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and fix_checkbox.isChecked():
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"Fixed value for '{_parameter_label(name)}' is not a valid number: {text!r}"
                    )
                fixed[name] = _panel_value_to_internal(name, value)
        return fixed

    def initial_guess_overrides_from_panel(self):
        """Mirrors fixed_params_from_panel for unchecked rows: reads
        each unfixed row's current Value cell as the starting guess for
        that parameter in the next fit (the parameter stays free -- the
        optimizer can still move it). Converts a sigma-family row's
        displayed FWHM back to sigma, same as fixed_params_from_panel.
        Raises FitError if an unchecked row's Value cell isn't a valid
        number."""
        overrides = {}
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and not fix_checkbox.isChecked():
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"Value for '{_parameter_label(name)}' is not a valid number: {text!r}"
                    )
                overrides[name] = _panel_value_to_internal(name, value)
        return overrides

    def update_results_list(self):
        self.results_table.setRowCount(0)
        self._results_row_fit_index = []
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        for fit_index, result in enumerate(active.fits):
            if isinstance(result, IntegrationResult):
                fit_label = f"{fit_index + 1} [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
                tooltip = _integration_tooltip(result)
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._results_row_fit_index.append(fit_index)
                values = [
                    fit_label,
                    f"{result.net_centroid:.2f} ± {result.net_centroid_err:.2f}",
                    f"{result.net_fwhm:.2f} ± {result.net_fwhm_err:.2f}",
                    f"{result.net_area:.1f} ± {result.net_area_err:.1f}",
                    "—",  # no chi^2 concept for a direct-sum Integration result
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if col == 0:
                        item.setToolTip(tooltip)
                    if not result.visible:
                        item.setForeground(QColor("gray"))
                    self.results_table.setItem(row, col, item)
                continue

            fit_label = f"{fit_index + 1} [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
            shared_tooltip_lines = [
                f"region full (no bg subtracted): {result.gross_area:.1f} ± {result.gross_area_err:.1f}",
                f"region net (bg subtracted): {result.net_area:.1f} ± {result.net_area_err:.1f}",
                f"reduced chi^2: {result.reduced_chi2:.3g}" if result.reduced_chi2 is not None
                else "reduced chi^2: undefined (zero degrees of freedom)",
            ]
            if not result.link_widths:
                shared_tooltip_lines.append("independent widths")
            if result.tail_fraction is not None:
                shared_tooltip_lines.append(
                    f"left tail: r={result.tail_fraction:.2f}"
                    f"±{result.tail_fraction_err:.2f}, "
                    f"β={result.tail_beta:.1f}±{result.tail_beta_err:.1f} "
                    f"(volume excludes tail)"
                )

            for peak in result.peaks:
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._results_row_fit_index.append(fit_index)

                peak_tooltip_lines = [
                    f"peak full (no bg subtracted): {peak.full_area:.1f} ± {peak.full_area_err:.1f}",
                    f"peak net (bg subtracted): {peak.area:.1f} ± {peak.area_err:.1f}",
                    *shared_tooltip_lines,
                ]
                tooltip = "\n".join(peak_tooltip_lines)

                values = [
                    fit_label,
                    f"{peak.position:.2f} ± {peak.position_err:.2f}",
                    f"{peak.fwhm:.2f} ± {peak.fwhm_err:.2f}",
                    f"{peak.area:.1f} ± {peak.area_err:.1f}",
                    f"{result.reduced_chi2:.3g}" if result.reduced_chi2 is not None else "—",
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if col == 0:
                        item.setToolTip(tooltip)
                    if not result.visible:
                        item.setForeground(QColor("gray"))
                    self.results_table.setItem(row, col, item)

    def _on_results_context_menu(self, position):
        mw = self.main_window
        item = self.results_table.itemAt(position)
        active = next((s for s in mw.spectra if s.active), None)
        menu = QMenu(mw)
        remove_action = menu.addAction("Remove Fit") if item is not None else None
        export_one_action = menu.addAction("Export This Fit...") if item is not None else None
        export_all_action = (
            menu.addAction("Export All Fits...") if active is not None and active.fits else None
        )
        clear_action = menu.addAction("Clear All Fits")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(position))
        if active is None:
            return
        if item is not None and chosen == remove_action:
            fit_index = self._results_row_fit_index[item.row()]
            del active.fits[fit_index]
            mw._plot_data(preserve_view=True)
        elif item is not None and chosen == export_one_action:
            fit_index = self._results_row_fit_index[item.row()]
            self._export_fits(active, [fit_index])
        elif export_all_action is not None and chosen == export_all_action:
            self._export_fits(active, list(range(len(active.fits))))
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data(preserve_view=True)

    def _export_fits(self, active, fit_indices):
        """Opens a save-file dialog and writes a plain-text report
        covering the given 0-based indices into active.fits -- used by
        both "Export This Fit..." (a single index) and "Export All
        Fits..." (every index, in Fit Results order)."""
        stem = os.path.splitext(os.path.basename(active.path))[0]
        if len(fit_indices) == 1:
            default_name = f"{stem}_fit{fit_indices[0] + 1}_report.txt"
        else:
            default_name = f"{stem}_fits_report.txt"
        directory = os.path.dirname(active.path)
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Fit Report", os.path.join(directory, default_name),
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        results = [(i + 1, active.fits[i]) for i in fit_indices]
        try:
            fit_export.write_text_report(path, results, active.path)
        except OSError as exc:
            self._show_status_message(f"Could not write export: {exc}", 5000)

    def _on_result_double_clicked(self, item):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        fit_index = self._results_row_fit_index[item.row()]
        result = active.fits[fit_index]

        self._clear_progress()

        # Bypasses the click-pairing API (add_bg_click/add_fit_click)
        # deliberately -- this restores a previously-computed,
        # already-valid state wholesale, not a fresh in-progress click
        # sequence.
        self.state.bg_regions = [result.left_bg_region, result.right_bg_region]
        self.state.fit_region = result.fit_region

        if isinstance(result, IntegrationResult):
            self.state.peak_positions = []
            self._redraw_progress()
            self.main_window._update_fit_mode_availability()
            return

        self.state.peak_positions = [peak.position for peak in result.peaks]

        self.main_window.independent_widths_action.setChecked(not result.link_widths)
        self.main_window.left_tail_action.setChecked(result.tail_fraction is not None)

        names = parameter_names(
            len(result.peaks), result.link_widths, result.tail_fraction is not None
        )
        self._parameter_names_shown = []  # force a full rebuild below
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        for row, name in enumerate(names):
            if name in result.fixed_params:
                self.parameters_table.cellWidget(row, 2).setChecked(True)

        self._redraw_progress()
        self.main_window._update_fit_mode_availability()

    def run_fit(self):
        if not self.state.ready_to_fit():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        link_widths = not self.main_window.independent_widths_action.isChecked()
        enable_left_tail = self.main_window.left_tail_action.isChecked()
        try:
            # A Fix checkbox (or an edited value) from a since-changed
            # row set (e.g. independent widths or left tail toggled
            # since the last fit) names a parameter that no longer
            # exists under the current configuration -- drop it rather
            # than let fit_peaks() reject the whole fit, since
            # update_parameters_panel() would reset that row anyway
            # once this fit succeeds and rebuilds the row set.
            current_names = set(
                parameter_names(len(self.state.peak_positions), link_widths, enable_left_tail)
            )
            fixed_params = {
                name: value for name, value in self.fixed_params_from_panel().items()
                if name in current_names
            }
            initial_guess_overrides = {
                name: value for name, value in self.initial_guess_overrides_from_panel().items()
                if name in current_names
            }
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions),
                link_widths=link_widths, enable_left_tail=enable_left_tail,
                fixed_params=fixed_params, initial_guess_overrides=initial_guess_overrides,
            )
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
        result.timestamp = datetime.now().isoformat(timespec="seconds")
        # Re-fitting the exact same marks (e.g. after toggling a
        # checkbox) is a supported workflow, not a mistake -- but
        # drawing every attempt at the identical region on top of the
        # others is just visual clutter. Only the latest attempt at a
        # given region is drawn; every attempt stays listed in Fit
        # Results.
        for earlier in active.fits:
            if (
                earlier.left_bg_region == result.left_bg_region
                and earlier.right_bg_region == result.right_bg_region
                and earlier.fit_region == result.fit_region
            ):
                earlier.visible = False
        active.fits.append(result)
        try:
            fit_export.append_auto_log(active.path, result)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
        names = parameter_names(len(result.peaks), result.link_widths, result.tail_fraction is not None)
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        self.main_window._plot_data(preserve_view=True)

    def run_integration(self):
        if not self.state.ready_to_integrate():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        try:
            result = integrate_region(x, y, left, right, self.state.fit_region)
        except FitError as exc:
            self._show_status_message(f"Integration failed: {exc}", 5000)
            return
        result.timestamp = datetime.now().isoformat(timespec="seconds")
        for earlier in active.fits:
            if (
                earlier.left_bg_region == result.left_bg_region
                and earlier.right_bg_region == result.right_bg_region
                and earlier.fit_region == result.fit_region
            ):
                earlier.visible = False
        active.fits.append(result)
        try:
            fit_export.append_auto_log(active.path, result)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
        self.main_window._plot_data(preserve_view=True)
