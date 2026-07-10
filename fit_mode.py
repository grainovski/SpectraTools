import time

import numpy as np
from matplotlib.backend_bases import _Mode
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QToolBar

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

    def draw_committed_fits(self, spectrum):
        axes = self.main_window.axes
        for result in spectrum.fits:
            axes.axvspan(*result.left_bg_region, color="gray", alpha=0.15)
            axes.axvspan(*result.right_bg_region, color="gray", alpha=0.15)
            axes.axvspan(*result.fit_region, color="tab:blue", alpha=0.1)

            lo, hi = result.fit_region
            background_lo = result.background_slope * lo + result.background_intercept
            background_hi = result.background_slope * hi + result.background_intercept
            axes.plot([lo, hi], [background_lo, background_hi], color="black",
                       linestyle="--", linewidth=1)

            x_dense = np.linspace(lo, hi, 200)
            total = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                total = total + peak.amplitude * np.exp(
                    -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                )
            axes.plot(x_dense, total, color="red", linewidth=1.5)

            for peak in result.peaks:
                axes.axvline(peak.position, color="red", linestyle=":", linewidth=1)

    def build_results_panel(self):
        mw = self.main_window
        self.results_list = QListWidget()
        self.results_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_list.customContextMenuRequested.connect(self._on_results_context_menu)

        self.results_dock = QDockWidget("Fit Results", mw)
        self.results_dock.setWidget(self.results_list)
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

    def update_results_list(self):
        self.results_list.clear()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        for result in active.fits:
            lines = [f"Fit region [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"]
            for i, peak in enumerate(result.peaks, start=1):
                lines.append(
                    f"  Peak {i}: pos={peak.position:.2f}±{peak.position_err:.2f}  "
                    f"FWHM={peak.fwhm:.2f}±{peak.fwhm_err:.2f}  "
                    f"area={peak.area:.1f}±{peak.area_err:.1f}"
                )
            self.results_list.addItem(QListWidgetItem("\n".join(lines)))

    def _on_results_context_menu(self, position):
        mw = self.main_window
        item = self.results_list.itemAt(position)
        menu = QMenu(mw)
        remove_action = menu.addAction("Remove Fit") if item is not None else None
        clear_action = menu.addAction("Clear All Fits")
        chosen = menu.exec(self.results_list.viewport().mapToGlobal(position))
        active = next((s for s in mw.spectra if s.active), None)
        if active is None:
            return
        if item is not None and chosen == remove_action:
            index = self.results_list.row(item)
            del active.fits[index]
            mw._plot_data()
            self.update_results_list()
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data()
            self.update_results_list()

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
        self.update_results_list()
