import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QToolBar

from peak_fit import FitError, fit_peaks, hypermet_left_tail

BG_REGION_CAP = 2


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
        replaces any existing fit region."""
        if self.pending_fit_click is None:
            self.pending_fit_click = x
            return None
        region = (min(self.pending_fit_click, x), max(self.pending_fit_click, x))
        self.pending_fit_click = None
        self.fit_region = region
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

_MARK_TYPE_LABEL = {
    "b": "background region",
    "r": "fit region",
    "p": "peak",
}


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

        canvas = main_window.canvas
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        canvas.mpl_connect("figure_enter_event", lambda event: canvas.setFocus())

    def eventFilter(self, obj, event):
        if obj is self.main_window.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                mark_type = _KEY_TO_MARK_TYPE.get(event.key())
                if mark_type is not None:
                    self._held_key = mark_type
                    self._show_status_message(
                        f"Marking {_MARK_TYPE_LABEL[mark_type]}: click to place", 60000
                    )
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                mark_type = _KEY_TO_MARK_TYPE.get(event.key())
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
        self.main_window._update_fit_mode_availability()
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
            self._progress_artists.append(axes.axvspan(*region, color="gray", alpha=0.15))

        if state.pending_fit_click is not None:
            self._progress_artists.append(
                axes.axvline(state.pending_fit_click, color="tab:blue", linestyle="--", linewidth=1)
            )
        if state.fit_region is not None:
            self._progress_artists.append(
                axes.axvspan(*state.fit_region, color="tab:blue", alpha=0.1)
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
            self.state.add_fit_click(event.xdata)
            self._redraw_progress()
        elif key == "p":
            proximity = self._pixel_proximity_to_data(event)
            result = self.state.toggle_peak(event.xdata, proximity)
            if result is None:
                self._show_status_message(
                    "Mark the fit region (hold R and click twice) before marking peaks", 3000
                )
                return
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
                if result.tail_fraction is not None:
                    total = total + peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    total = total + peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
            axes.plot(x_dense, total, color="red", linewidth=1.5)

            for peak in result.peaks:
                axes.axvline(peak.position, color="red", linestyle=":", linewidth=1)
                axes.annotate(
                    f"pos={peak.position:.1f}\nFWHM={peak.fwhm:.1f}\nvol={peak.area:.0f}",
                    xy=(peak.position, 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color="red",
                )

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
            header = f"Fit region [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
            if not result.link_widths:
                header += "  (independent widths)"
            if result.tail_fraction is not None:
                header += (
                    f"  (left tail: r={result.tail_fraction:.2f}"
                    f"±{result.tail_fraction_err:.2f}, "
                    f"β={result.tail_beta:.1f}±{result.tail_beta_err:.1f}, "
                    f"volume excludes tail)"
                )
            lines = [header]
            for i, peak in enumerate(result.peaks, start=1):
                lines.append(
                    f"  Peak {i}: pos={peak.position:.2f}±{peak.position_err:.2f}  "
                    f"FWHM={peak.fwhm:.2f}±{peak.fwhm_err:.2f}  "
                    f"volume={peak.area:.1f}±{peak.area_err:.1f}"
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
            mw._plot_data(preserve_view=True)
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data(preserve_view=True)

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
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions),
                link_widths=link_widths, enable_left_tail=enable_left_tail,
            )
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
        active.fits.append(result)
        self.main_window._plot_data(preserve_view=True)
