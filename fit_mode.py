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
        return self.fit_blocked_reason() is None

    def ready_to_integrate(self):
        return self.integrate_blocked_reason() is None

    def fit_blocked_reason(self):
        """Human-readable explanation of what's still missing before
        ready_to_fit() would return True, or None if it's already
        ready. Checked in the order a user would naturally complete
        marks in -- toggle_peak() itself refuses to add a peak before
        fit_region exists, so checking fit_region first always points
        at genuinely the next missing thing, never a redundant one.
        Returns None under exactly the condition ready_to_fit()
        returns True (the same three checks, so a caller that already
        confirmed ready_to_fit() is False can call this directly with
        no further guard)."""
        if self.fit_region is None:
            return "Mark the fit region (hold R and click twice) before fitting"
        if len(self.bg_regions) != BG_REGION_CAP:
            return "Mark two background regions (hold B and click twice per region) before fitting"
        if len(self.peak_positions) == 0:
            return "Mark at least one peak (hold P and click) before fitting"
        return None

    def integrate_blocked_reason(self):
        """Same as fit_blocked_reason(), but for ready_to_integrate()
        -- Integration accepts zero or two background regions, never
        exactly one."""
        if self.fit_region is None:
            return "Mark the fit region (hold R and click twice) before integrating"
        if len(self.bg_regions) not in (0, BG_REGION_CAP):
            return "Mark zero or two background regions (not one) before integrating"
        return None

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first. Returns
        (None, None) when no background regions are marked -- a valid,
        deliberate state now that Integration allows a zero-background
        run. Only meaningful when len(bg_regions) is 0 or BG_REGION_CAP;
        any other count (e.g. exactly 1) is not a state
        ready_to_integrate() would ever let a caller reach."""
        if not self.bg_regions:
            return (None, None)
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


def _parameter_label(name, calibrated=False):
    """Human-readable row label for a canonical parameter name from
    peak_fit.parameter_names() -- e.g. "amp_0" -> "Peak 1 amplitude".
    Sigma-family names are labeled as FWHM: the panel displays and
    accepts FWHM, converting to/from the fitting engine's internal
    sigma at the UI boundary (see _display_value/_panel_value_to_internal
    below) -- peak_fit.py's own parameter naming and optimizer are
    untouched. `calibrated=True` appends " (keV)" to position/FWHM
    labels -- the only rows whose displayed/accepted unit changes when
    calibration is active; amplitude and the tail parameters have no
    energy-axis equivalent and are never converted."""
    if name == "sigma":
        return "Shared FWHM (keV)" if calibrated else "Shared FWHM"
    if name == "tail_fraction":
        return "Tail fraction (r)"
    if name == "tail_beta":
        return "Tail beta (β)"
    prefix, index = name.rsplit("_", 1)
    peak_num = int(index) + 1
    kind = {"amp": "amplitude", "pos": "position", "sigma": "FWHM"}[prefix]
    suffix = " (keV)" if calibrated and prefix in ("pos", "sigma") else ""
    return f"Peak {peak_num} {kind}{suffix}"


def _dual_unit_value(main_window, channel_value, channel_err, is_width, reference_position=None):
    """Formats a channel-space value+error as "X.XX ± Y.YY ch (E.EE ±
    F.FF keV)" when calibration is active, or plain "X.XX ± Y.YY"
    otherwise. `is_width=False` (a position, e.g. peak centroid)
    converts through the full calibration (cal.apply, including the
    offset `a`), with its own local derivative used for error
    propagation. `is_width=True` (e.g. FWHM) scales by the calibration's
    local derivative evaluated at `reference_position` -- the peak's own
    position, NOT the width's numeric value, which has no location on
    the calibration curve of its own; required when is_width is True.
    keV uncertainty is first-order error propagation through the
    calibration's local derivative in both cases -- exact for linear
    calibration, a good approximation for quadratic given realistic
    peak-width uncertainties are small relative to the calibration's
    curvature scale."""
    if not main_window._calibration_active or main_window._calibration is None:
        return f"{channel_value:.2f} ± {channel_err:.2f}"
    cal = main_window._calibration
    if is_width:
        slope = abs(cal.derivative(reference_position))
        energy = slope * channel_value
    else:
        slope = abs(cal.derivative(channel_value))
        energy = cal.apply(channel_value)
    energy_err = slope * channel_err
    return f"{channel_value:.2f} ± {channel_err:.2f} ch ({energy:.2f} ± {energy_err:.2f} keV)"


def _unit_switched_value(main_window, channel_value, channel_err, is_width, reference_position=None):
    """Formats a channel-space value+error as plain "X.XX ± Y.YY", in
    keV when calibration is active or channels otherwise. Unlike
    _dual_unit_value (used for the Integration tooltip's supplementary
    detail, where there's room for both units), the Fit Results
    table's columns are too narrow for a combined "ch (keV)" string --
    the caller is responsible for indicating the active unit via the
    column header instead of repeating it in every cell.
    `is_width`/`reference_position` mean the same as in
    _dual_unit_value -- a width converts via the calibration's local
    derivative evaluated at reference_position, never at the width's
    own value."""
    if not main_window._calibration_active or main_window._calibration is None:
        return f"{channel_value:.2f} ± {channel_err:.2f}"
    cal = main_window._calibration
    if is_width:
        slope = abs(cal.derivative(reference_position))
        value = slope * channel_value
    else:
        slope = abs(cal.derivative(channel_value))
        value = cal.apply(channel_value)
    err = slope * channel_err
    return f"{value:.2f} ± {err:.2f}"


def _integration_tooltip(main_window, result):
    """Full gross/background/net breakdown for an IntegrationResult's Fit
    Results row tooltip -- mirrors the level of detail the existing
    left-tail-info tooltip gives for a Gaussian fit. Centroid/FWHM show
    dual units when calibration is active; area has no energy-axis
    equivalent and stays channel-only, matching the results table. With
    no background marked, gross/background/net collapse to one number
    each (net == gross exactly), so a single unlabeled line is shown
    instead of a three-way breakdown -- reading from gross_* directly
    rather than relying on that equality, matching TV's own choice to
    report the total/gross row in this case."""
    if not result.has_background:
        return (
            f"Area={result.gross_area:.1f}±{result.gross_area_err:.1f}, "
            f"centroid={_dual_unit_value(main_window, result.gross_centroid, result.gross_centroid_err, is_width=False)}, "
            f"FWHM={_dual_unit_value(main_window, result.gross_fwhm, result.gross_fwhm_err, is_width=True, reference_position=result.gross_centroid)}, "
            f"skewness={result.gross_skewness:.3g}±{result.gross_skewness_err:.3g}"
        )
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
            f"centroid={_dual_unit_value(main_window, centroid, centroid_err, is_width=False)}, "
            f"FWHM={_dual_unit_value(main_window, fwhm, fwhm_err, is_width=True, reference_position=centroid)}, "
            f"skewness={skewness:.3g}±{skewness_err:.3g}"
        )
    return "\n".join(lines)


def _is_sigma_name(name):
    return name == "sigma" or name.startswith("sigma_")


def _panel_reference_position(name, values_by_name):
    """The peak position (channel-space) a sigma/FWHM row's keV
    conversion should be evaluated at -- that same peak's own "pos_i"
    entry, or "pos_0" for the single shared "sigma" row used when
    widths are linked (a shared width has no one position of its own
    to anchor to). None if the needed position isn't in values_by_name."""
    reference_name = "pos_0" if name == "sigma" else f"pos_{name.split('_', 1)[1]}"
    return values_by_name.get(reference_name)


def _display_value(main_window, name, value, values_by_name):
    """Converts a canonical parameter's internal (always channel-space)
    value to what the Fit Parameters panel displays. Sigma-family
    values are shown as FWHM (fwhm = sigma * FWHM_FACTOR) regardless of
    calibration. When calibration is active, position and FWHM values
    are further converted to keV -- a FWHM row's derivative reference
    point is the corresponding peak's own position (never the FWHM's
    own value, and never a different peak's position -- see
    _panel_reference_position). Amplitude and the tail parameters have
    no energy-axis equivalent and are returned unconverted regardless
    of calibration state."""
    fwhm_value = value * FWHM_FACTOR if _is_sigma_name(name) else value
    if not main_window._calibration_active or main_window._calibration is None:
        return fwhm_value
    cal = main_window._calibration
    if name.startswith("pos_"):
        return cal.apply(fwhm_value)
    if _is_sigma_name(name):
        reference_position = _panel_reference_position(name, values_by_name)
        if reference_position is None:
            return fwhm_value
        return abs(cal.derivative(reference_position)) * fwhm_value
    return fwhm_value


def _panel_value_to_internal(main_window, name, value, values_by_name):
    """Inverse of _display_value -- converts a value read back from the
    panel (in whichever unit is currently displayed: keV if calibration
    is active, channels/FWHM otherwise) to the sigma-space channel
    value fit_peaks() expects. Uses the same reference-position lookup
    as _display_value for FWHM rows, from values_by_name (the panel's
    last-known channel-space values from the last fit or refresh, not
    a freshly re-parsed position cell -- editing a peak's position and
    its own FWHM in keV in the same batch before re-fitting uses the
    pre-edit position as the reference; a narrow, accepted
    simplification, not a general live-recompute)."""
    fwhm_value = value
    if main_window._calibration_active and main_window._calibration is not None:
        cal = main_window._calibration
        if name.startswith("pos_"):
            fwhm_value = cal.invert(value)
        elif _is_sigma_name(name):
            reference_position = _panel_reference_position(name, values_by_name)
            if reference_position is not None:
                slope = abs(cal.derivative(reference_position))
                if slope != 0:
                    fwhm_value = value / slope
    return fwhm_value / FWHM_FACTOR if _is_sigma_name(name) else fwhm_value


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
        self._parameters_panel_values_shown = {}
        self._parameters_panel_calibrated = False
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
        self.reset_marks()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            for result in active.fits:
                result.visible = False
        self.main_window._plot_data(preserve_view=True)

    def reset_marks(self):
        """Public entry point for resetting in-progress marking state
        (in-progress background/fit-region/peak clicks and their drawn
        artists) without touching committed fits or replotting -- used
        by clear() (Ctrl+C) and by Multiply/Rebin/Normalize
        (main_window.py) after they change a spectrum's data, per the
        design spec's "Fits/Marks Clearing Semantics"."""
        self._clear_progress()

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
        removed peak. FitModeState itself always stays channel-based;
        to_display converts to keV for drawing only, when calibrated."""
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._progress_artists = []

        axes = self.main_window.axes
        state = self.state
        to_display = self.main_window.channel_to_display

        if state.pending_bg_click is not None:
            self._progress_artists.append(
                axes.axvline(
                    to_display(state.pending_bg_click), color="gray", linestyle="--", linewidth=1
                )
            )
        for lo, hi in state.bg_regions:
            self._progress_artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
            )

        if state.pending_fit_click is not None:
            self._progress_artists.append(
                axes.axvline(
                    to_display(state.pending_fit_click), color="tab:blue", linestyle="--", linewidth=1
                )
            )
        if state.fit_region is not None:
            lo, hi = state.fit_region
            self._progress_artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA
                )
            )

        for x in state.peak_positions:
            self._progress_artists.append(
                axes.axvline(to_display(x), color="red", linestyle=":", linewidth=1)
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
        channel_x = self.main_window.display_to_channel(event.xdata)
        if key == "b":
            self.state.add_bg_click(channel_x)
            self._redraw_progress()
        elif key == "r":
            completed = self.state.add_fit_click(channel_x)
            if completed is not None:
                self._reset_parameters_panel()
            self._redraw_progress()
        elif key == "p":
            proximity = self._pixel_proximity_to_data(event)
            # Convert the proximity *distance* to channel units by
            # inverting both endpoints and taking their difference,
            # rather than scaling by the calibration's derivative --
            # reuses the exact same, already-tested invert() with no
            # extra approximation math. Exact for a linear calibration;
            # for quadratic, this is a one-sided estimate (only the
            # rightward endpoint is inverted), so the resulting
            # hit-test tolerance is very slightly asymmetric -- fine
            # for a fuzzy "close enough" click radius.
            proximity_channels = abs(
                self.main_window.display_to_channel(event.xdata + proximity) - channel_x
            )
            result = self.state.toggle_peak(channel_x, proximity_channels)
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
        to_display = self.main_window.channel_to_display
        for result in spectrum.fits:
            if not result.visible:
                continue
            if result.left_bg_region is not None:
                left_lo, left_hi = result.left_bg_region
                axes.axvspan(
                    to_display(left_lo), to_display(left_hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
                right_lo, right_hi = result.right_bg_region
                axes.axvspan(
                    to_display(right_lo), to_display(right_hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
            fit_lo, fit_hi = result.fit_region
            axes.axvspan(
                to_display(fit_lo), to_display(fit_hi), color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA
            )

            if isinstance(result, IntegrationResult):
                if result.has_background:
                    lo, hi = result.fit_region
                    axes.plot(
                        [to_display(lo), to_display(hi)],
                        [result.background_density, result.background_density],
                        color=bg_line_color, linestyle="--", linewidth=1,
                    )
                    axes.annotate(
                        f"centroid={result.net_centroid:.1f}\n"
                        f"FWHM={result.net_fwhm:.1f}\n"
                        f"full={result.gross_area:.0f}\nnet={result.net_area:.0f}",
                        xy=(to_display(result.net_centroid), 0.95),
                        xycoords=label_transform,
                        ha="center", va="top",
                        fontsize=7, color=fit_color,
                    )
                else:
                    axes.annotate(
                        f"centroid={result.gross_centroid:.1f}\n"
                        f"FWHM={result.gross_fwhm:.1f}\n"
                        f"area={result.gross_area:.0f}",
                        xy=(to_display(result.gross_centroid), 0.95),
                        xycoords=label_transform,
                        ha="center", va="top",
                        fontsize=7, color=fit_color,
                    )
                continue

            lo, hi = result.fit_region
            background_lo = result.background_slope * lo + result.background_intercept
            background_hi = result.background_slope * hi + result.background_intercept
            axes.plot(
                [to_display(lo), to_display(hi)], [background_lo, background_hi],
                color=bg_line_color, linestyle="--", linewidth=1,
            )

            # x_dense stays in channel space -- the model below (linear
            # background + Gaussian/hypermet peaks) is defined in terms
            # of the fitted channel-space parameters (peak.position,
            # peak.sigma, background_slope). Only the final plotted
            # x-coordinates are converted, via to_display(x_dense),
            # never the values used in the model math itself.
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
            axes.plot(to_display(x_dense), total, color=fit_color, linewidth=1.5)

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
                    to_display(x_dense), background_dense + component,
                    color=fit_color, linewidth=0.75, linestyle="--", alpha=0.6,
                )

            for peak in result.peaks:
                label_x = to_display(peak.position)
                axes.axvline(label_x, color=fit_color, linestyle=":", linewidth=1)
                axes.annotate(
                    f"{label_x:.1f}",
                    xy=(label_x, 0.95),
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

        self.clear_all_fits_action = QAction("Clear All Fits", mw)
        self.clear_all_fits_action.setShortcut("Ctrl+Shift+C")
        self.clear_all_fits_action.triggered.connect(self._clear_all_fits)
        mw.addAction(self.clear_all_fits_action)

        self.export_all_fits_action = QAction("Export All Fits...", mw)
        self.export_all_fits_action.setShortcut("Ctrl+E")
        self.export_all_fits_action.triggered.connect(self._export_all_fits)
        mw.addAction(self.export_all_fits_action)

        self.results_dock = QDockWidget("Fit Results", mw)
        self.results_dock.setWidget(self.results_table)
        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.results_dock)

        self.toggle_results_panel_action = QAction("Fit Results", mw)
        self.toggle_results_panel_action.setShortcut("Ctrl+2")
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
        self.toggle_parameters_panel_action.setShortcut("Ctrl+3")
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
        checkbox) when the set of parameter names OR the calibration
        active/inactive state has changed since the last call;
        otherwise updates displayed values in place, preserving Fix
        checkbox state and any user-edited value. Every row's Value
        cell is always editable: a checked row's edited value is read
        as a fixed value for the next fit (fixed_params_from_panel);
        an unchecked row's edited value is read as that parameter's
        starting guess for the next fit
        (initial_guess_overrides_from_panel) -- the parameter stays
        free, the optimizer can still move it. Position/FWHM rows
        display and accept keV when calibration is active (their row
        label gains a "(keV)" suffix to say so), channels otherwise --
        a value read back through this boundary is always converted to
        channels before reaching fit_peaks(), which never sees
        calibration. Toggling calibration is treated like a names
        change (a full rebuild, clearing Fix checkboxes) rather than
        trying to re-interpret already-displayed text in the new unit
        -- simpler and safer than risking a silently wrong
        reinterpretation of a hand-edited value, at the minor cost of
        needing to re-fix a row if calibration happens to be toggled
        between fixing it and the next fit."""
        calibrated = self.main_window._calibration_active and self.main_window._calibration is not None
        if names != self._parameter_names_shown or calibrated != self._parameters_panel_calibrated:
            self.parameters_table.setRowCount(0)
            self._parameter_names_shown = list(names)
            self._parameters_panel_calibrated = calibrated
            for name in names:
                row = self.parameters_table.rowCount()
                self.parameters_table.insertRow(row)

                label_item = QTableWidgetItem(_parameter_label(name, calibrated))
                label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.parameters_table.setItem(row, 0, label_item)

                value_item = QTableWidgetItem(
                    f"{_display_value(self.main_window, name, values_by_name[name], values_by_name):.6g}"
                )
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
                        f"{_display_value(self.main_window, name, values_by_name[name], values_by_name):.6g}"
                    )
        self._parameters_panel_values_shown = dict(values_by_name)

    def refresh_parameters_panel_calibration(self):
        """Re-renders the Fit Parameters panel for the current
        calibration state, reusing whatever values were last shown --
        called when calibration is toggled so an already-populated
        panel doesn't lag behind the plot and Fit Results table, which
        both already refresh immediately via
        MainWindow._apply_calibration_change's own _plot_data() call.
        Since update_parameters_panel treats a calibration-state
        change as a full rebuild, this also resets any Fix checkboxes
        -- see its docstring for why. A no-op if nothing has been
        shown yet."""
        if self._parameter_names_shown:
            self.update_parameters_panel(self._parameter_names_shown, self._parameters_panel_values_shown)

    def fixed_params_from_panel(self):
        """Reads the current Fix checkboxes/values from the Fit
        Parameters panel into a {name: value} dict for the next
        fit_peaks() call. Empty when nothing is fixed (including the
        first fit for a fresh set of marks, before the panel has ever
        been populated). A sigma-family row's Value cell holds FWHM;
        a position/FWHM row's Value cell holds keV when calibration is
        active -- _panel_value_to_internal converts either back to the
        sigma-space channel value fit_peaks() expects. Raises FitError
        if a checked row's Value cell isn't a valid number."""
        fixed = {}
        calibrated = self.main_window._calibration_active and self.main_window._calibration is not None
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and fix_checkbox.isChecked():
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"Fixed value for '{_parameter_label(name, calibrated)}' is not a valid number: {text!r}"
                    )
                fixed[name] = _panel_value_to_internal(
                    self.main_window, name, value, self._parameters_panel_values_shown
                )
        return fixed

    def initial_guess_overrides_from_panel(self):
        """Mirrors fixed_params_from_panel for unchecked rows: reads
        each unfixed row's current Value cell as the starting guess for
        that parameter in the next fit (the parameter stays free -- the
        optimizer can still move it). Converts back to a channel-space
        sigma value the same way fixed_params_from_panel does. Raises
        FitError if an unchecked row's Value cell isn't a valid
        number."""
        overrides = {}
        calibrated = self.main_window._calibration_active and self.main_window._calibration is not None
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and not fix_checkbox.isChecked():
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"Value for '{_parameter_label(name, calibrated)}' is not a valid number: {text!r}"
                    )
                overrides[name] = _panel_value_to_internal(
                    self.main_window, name, value, self._parameters_panel_values_shown
                )
        return overrides

    def update_results_list(self):
        calibrated = self.main_window._calibration_active and self.main_window._calibration is not None
        self.results_table.setHorizontalHeaderLabels([
            "Fit",
            "Position (keV)" if calibrated else "Position",
            "FWHM (keV)" if calibrated else "FWHM",
            "Volume",
            "chi^2",
        ])
        self.results_table.setRowCount(0)
        self._results_row_fit_index = []
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        for fit_index, result in enumerate(active.fits):
            if isinstance(result, IntegrationResult):
                fit_label = f"{fit_index + 1} [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
                tooltip = _integration_tooltip(self.main_window, result)
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._results_row_fit_index.append(fit_index)
                values = [
                    fit_label,
                    _unit_switched_value(
                        self.main_window, result.net_centroid, result.net_centroid_err, is_width=False
                    ),
                    _unit_switched_value(
                        self.main_window, result.net_fwhm, result.net_fwhm_err,
                        is_width=True, reference_position=result.net_centroid,
                    ),
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
                    _unit_switched_value(
                        self.main_window, peak.position, peak.position_err, is_width=False
                    ),
                    _unit_switched_value(
                        self.main_window, peak.fwhm, peak.fwhm_err,
                        is_width=True, reference_position=peak.position,
                    ),
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
            self._export_all_fits()
        elif chosen == clear_action:
            self._clear_all_fits()

    def _clear_all_fits(self):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        active.fits.clear()
        self.main_window._plot_data(preserve_view=True)

    def _export_all_fits(self):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None or not active.fits:
            return
        self._export_fits(active, list(range(len(active.fits))))

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
        calibration = self.main_window._calibration if self.main_window._calibration_active else None
        try:
            fit_export.write_text_report(path, results, active.path, calibration)
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
        # sequence. An empty list (not [None, None]) is bg_regions'
        # own canonical "no background" representation everywhere else
        # in FitModeState (see ordered_bg_regions()) -- restoring a
        # zero-background result must produce [], or _redraw_progress()'s
        # `for lo, hi in state.bg_regions:` unpack crashes on None.
        self.state.bg_regions = (
            [] if result.left_bg_region is None
            else [result.left_bg_region, result.right_bg_region]
        )
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
            self._show_status_message(self.state.fit_blocked_reason(), 5000)
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
        calibration = self.main_window._calibration if self.main_window._calibration_active else None
        try:
            fit_export.append_auto_log(active.path, result, calibration)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
        names = parameter_names(len(result.peaks), result.link_widths, result.tail_fraction is not None)
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        self.main_window._plot_data(preserve_view=True)

    def run_integration(self):
        if not self.state.ready_to_integrate():
            self._show_status_message(self.state.integrate_blocked_reason(), 5000)
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
        calibration = self.main_window._calibration if self.main_window._calibration_active else None
        try:
            fit_export.append_auto_log(active.path, result, calibration)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
        self.main_window._plot_data(preserve_view=True)
