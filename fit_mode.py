import time

import numpy as np
from matplotlib.backend_bases import _Mode
from PySide6.QtGui import QAction

from peak_fit import FitError, fit_peaks

STEP_LEFT_BG = "left_bg"
STEP_RIGHT_BG = "right_bg"
STEP_FIT_REGION = "fit_region"
STEP_MARKING_PEAKS = "marking_peaks"


class FitModeState:
    """Tracks the in-progress region/peak marking sequence for one
    fit-mode session. Pure state -- no Qt/matplotlib dependency."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step = STEP_LEFT_BG
        self.left_bg_region = None
        self.right_bg_region = None
        self.fit_region = None
        self.peak_positions = []

    def add_region(self, lo, hi):
        region = (min(lo, hi), max(lo, hi))
        if self.step == STEP_LEFT_BG:
            self.left_bg_region = region
            self.step = STEP_RIGHT_BG
        elif self.step == STEP_RIGHT_BG:
            self.right_bg_region = region
            self.step = STEP_FIT_REGION
        elif self.step == STEP_FIT_REGION:
            self.fit_region = region
            self.step = STEP_MARKING_PEAKS
        else:
            raise ValueError("Not currently awaiting a region selection")

    def add_peak(self, position):
        if self.step != STEP_MARKING_PEAKS:
            raise ValueError("Not currently marking peaks")
        lo, hi = self.fit_region
        if not (lo <= position <= hi):
            return False
        self.peak_positions.append(position)
        return True

    def ready_to_fit(self):
        return (
            self.left_bg_region is not None
            and self.right_bg_region is not None
            and self.fit_region is not None
            and len(self.peak_positions) > 0
        )

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first."""
        a, b = self.left_bg_region, self.right_bg_region
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)


class FitModeController:
    """Qt/matplotlib-facing wrapper around FitModeState: owns the plot
    artists for in-progress marking (committed-fit drawing and the
    results panel are added on top of this in later tasks), and drives
    fit_peaks() when the user clicks "Fit"."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.enabled = False
        self.state = FitModeState()
        self._drag_start = None
        self._progress_artists = []
        self._status_message_until = 0.0

    def _show_status_message(self, message, duration_ms):
        self.main_window.statusBar().showMessage(message, duration_ms)
        self._status_message_until = time.monotonic() + duration_ms / 1000.0

    def toggle(self, enabled):
        self.enabled = enabled
        self._clear_progress()
        mw = self.main_window
        mw.nav_toolbar.setEnabled(not enabled)
        if enabled:
            if mw.nav_toolbar.mode == _Mode.PAN:
                mw.nav_toolbar.pan()
            elif mw.nav_toolbar.mode == _Mode.ZOOM:
                mw.nav_toolbar.zoom()
        mw.fit_button.setEnabled(False)
        mw.clear_fit_button.setEnabled(enabled)
        mw.canvas.draw_idle()

    def clear(self):
        self._clear_progress()
        self.main_window.fit_button.setEnabled(False)
        self.main_window.canvas.draw_idle()

    def _clear_progress(self):
        for artist in self._progress_artists:
            artist.remove()
        self._progress_artists = []
        self.state.reset()

    def on_press(self, event):
        if not self.enabled or event.inaxes != self.main_window.axes or event.xdata is None:
            return
        if event.button != 1:
            return
        self._drag_start = event.xdata

    def on_release(self, event):
        if not self.enabled or self._drag_start is None:
            return
        if event.inaxes != self.main_window.axes or event.xdata is None:
            self._drag_start = None
            return
        start = self._drag_start
        end = event.xdata
        self._drag_start = None

        if self.state.step == STEP_MARKING_PEAKS:
            added = self.state.add_peak(end)
            if added:
                artist = self.main_window.axes.axvline(
                    end, color="red", linestyle=":", linewidth=1
                )
                self._progress_artists.append(artist)
                self.main_window.canvas.draw_idle()
            else:
                self._show_status_message("Peak position must be inside the fit region", 3000)
        else:
            lo, hi = min(start, end), max(start, end)
            if hi <= lo:
                self._show_status_message(
                    "Drag to select a region (a click alone is not enough)", 3000
                )
                return
            color = "tab:blue" if self.state.step == STEP_FIT_REGION else "gray"
            artist = self.main_window.axes.axvspan(lo, hi, color=color, alpha=0.15)
            self._progress_artists.append(artist)
            self.state.add_region(lo, hi)
            self.main_window.canvas.draw_idle()

        self.main_window.fit_button.setEnabled(self.state.ready_to_fit())

    def run_fit(self):
        if not self.state.ready_to_fit():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        try:
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions)
            )
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
        active.fits.append(result)
        self._clear_progress()
        self.main_window.fit_button.setEnabled(False)
        self.main_window._plot_data()
