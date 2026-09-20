"""Shows an energy calibration as a fit, not as a status line.

A calibration's coefficients look equally plausible whether or not one
of its points was misidentified. The residual strip is where that shows
up, which is why this window exists at all rather than a message box
reporting a and b.
"""

import math
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
)

from caleneff_export import ExportError, build_rows, write_caleneff
from calibration_quality import reduced_chi_squared
from value_format import compact

#: Points drawn along the fitted curve. Enough that a quadratic reads as
#: a curve rather than a polyline at any zoom the dialog offers.
_CURVE_SAMPLES = 400

#: How near a click has to land, in screen pixels, to count as picking a
#: point. Measured on screen and not in data units because the axes are
#: channels against keV: a distance in data units would be meaningless
#: in one direction or the other. Generous enough to hit a marker
#: without aiming, small enough that a click on empty space clears the
#: pick instead of grabbing whatever was nearest.
_PICK_RADIUS_PX = 12.0

#: How close a plotted point's energy has to be to a source line's for
#: that line's stated uncertainty to be used as the point's own. Both
#: numbers came from the same file and differ only by a round trip
#: through the table's text, so this only has to absorb that -- it is
#: the same tolerance caleneff_export matches on, for the same reason.
_ENERGY_MATCH_KEV = 1e-6


class EfficiencyWorker(QThread):
    """Runs the efficiency Monte Carlo off the UI thread.

    Ten thousand iterations across two models takes about a minute. Doing
    that in the click handler would freeze the window for the whole of it,
    with no way to stop, and a frozen window is indistinguishable from a
    crashed one.

    The signals are progressed, succeeded and failed rather than the more
    obvious "finished", because QThread already defines a signal of that
    name and shadowing it sends the connection somewhere else with no error.
    """

    progressed = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent, rows):
        super().__init__(parent)
        self._rows = rows
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def compute(self, iterations=None):
        """The fit and the Monte Carlo, synchronously.

        Separate from run() so it can be tested without a thread: what the
        thread does and the fact that it is a thread are two claims, and
        they are worth checking apart.
        """
        from efficiency import (N_MC_EFFICIENCY, fit_efficiency,
                                run_monte_carlo)

        rows = self._rows
        E = np.array([r.energy for r in rows], dtype=float)
        N = np.array([r.area for r in rows], dtype=float)
        dN = np.array([r.area_err for r in rows], dtype=float)
        I = np.array([r.intensity_pct for r in rows], dtype=float)
        dI = np.array([r.intensity_pct_err for r in rows], dtype=float)
        fit = fit_efficiency(E, N, dN, I, dI)
        mc = run_monte_carlo(
            fit, N, dN, I, dI,
            iterations=N_MC_EFFICIENCY if iterations is None else iterations,
            progress=self._progress)
        return fit, mc

    def run(self):
        try:
            fit, mc = self.compute()
        except Exception as exc:          # noqa: BLE001 - reported, not raised
            # A raise here would cross a thread boundary and vanish; in a
            # --windowed build there is no stderr to carry it either.
            self.failed.emit(str(exc))
            return
        if self._cancelled:
            return
        self.succeeded.emit((fit, mc))

    def _progress(self, done, total):
        self.progressed.emit(done, total)
        return not self._cancelled


class CalibrationPlotDialog(QDialog):
    """`points` is (channel, channel_err, area, area_err, energy) per
    assigned peak. `source_lines` are the .sou lines they were assigned
    from, used only by the export."""

    #: Index into the `points` this window was last drawn with, when the
    #: user clicks one of them on either axes; -1 when a click lands on
    #: none. The Calibrate dialog uses it to select the matching row, so
    #: a point read off the residual strip can be found in the table
    #: without counting rows.
    pointPicked = Signal(int)

    #: Radware has five free parameters, so ndf = n - 5 must exceed zero.
    #: Six is the smallest n leaving any redundancy at all.
    MIN_EFFICIENCY_POINTS = 6

    def __init__(self, parent, calibration, points, source_lines,
                 max_channel, default_path, excluded=(), reason=None):
        super().__init__(parent)
        self.setWindowTitle("Energy Calibration")
        self._default_path = default_path
        self._max_channel = max_channel
        # Remembered by channel, not by position in the list: the list
        # is rebuilt on every keystroke in the Calibrate dialog's table,
        # and clearing one row's energy shifts every index after it. A
        # ring that quietly moved to a different peak would be worse than
        # no ring at all.
        self._picked_channel = None
        self._picked_artists = []

        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(6.5, 5.0))
        self.canvas = FigureCanvasQTAgg(self._figure)
        # Residuals share the x axis and get a third of the height: they
        # are read against the main plot, not on their own.
        self.axes, self.residual_axes = self._figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        layout.addWidget(self.canvas)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        # Selectable, so the fitted coefficients can be copied out of
        # the window. The previous call handed the label its own
        # current flags straight back and therefore changed nothing.
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.summary_label)

        # Which point was clicked, in words. The ring on the plot says
        # where it is; this says what it is, and is the only feedback
        # visible when the Calibrate dialog's table is behind this
        # window.
        self.picked_label = QLabel()
        self.picked_label.setWordWrap(True)
        layout.addWidget(self.picked_label)

        self.canvas.mpl_connect("button_press_event", self._on_click)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.finish_button = buttons.addButton(
            "Finish and save for CalEnEff...",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.finish_button.clicked.connect(self._on_finish)
        self.efficiency_button = buttons.addButton(
            "Auto MC Efficiency Calibration...",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.efficiency_button.clicked.connect(self._on_efficiency_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.set_data(calibration, points, source_lines, excluded=excluded,
                      reason=reason)

    def set_data(self, calibration, points, source_lines, max_channel=None,
                 excluded=(), reason=None):
        """Redraw for a new calibration without rebuilding the window.

        The Calibrate dialog's live preview refreshes after every change
        to the assignments. Closing and recreating the window each time
        would raise it to the front and take focus away from the table
        the user is still typing into, so the same window is redrawn
        instead.

        `calibration` may be None -- too few points for the chosen kind,
        or a typo mid-edit. The points are still drawn, without a curve
        or residuals, and `reason` replaces the coefficients in the
        summary. The window stays: closing it whenever the fit was
        momentarily impossible made switching linear to quadratic with
        two points look like the plot had crashed.
        """
        self._calibration = calibration
        self._points = list(points)
        self._source_lines = list(source_lines or [])
        self._excluded = tuple(excluded or ())
        if max_channel is not None:
            self._max_channel = max_channel

        # An excluded point is still DRAWN, hollow, and its residual is
        # still shown. That is the whole reason to exclude one: you want
        # to see how far the point you rejected sits from the fit made
        # without it. Hiding it would remove the evidence.
        def is_out(energy):
            return any(abs(energy - e) < 1e-9 for e in self._excluded)

        used = [p for p in self._points if not is_out(p[4])]
        left = [p for p in self._points if is_out(p[4])]

        channels = [p[0] for p in used]
        errors = [p[1] for p in used]
        energies = [p[4] for p in used]

        self.axes.clear()
        self.residual_axes.clear()
        self._picked_artists = []       # the clear() above took them with it

        self.axes.errorbar(
            channels, energies, xerr=errors, fmt="o", capsize=3,
            label="assigned peaks",
        )
        if left:
            self.axes.errorbar(
                [p[0] for p in left], [p[4] for p in left],
                xerr=[p[1] for p in left], fmt="o", capsize=3,
                markerfacecolor="none", label="excluded from the fit",
            )
        self.axes.set_ylabel("Energy (keV)")
        if calibration is not None:
            upper = max(self._max_channel, max(channels) if channels else 0)
            grid = np.linspace(0.0, float(upper), _CURVE_SAMPLES)
            self.axes.plot(grid, calibration.apply(grid), "-", label="calibration")
            residuals = [e - calibration.apply(c) for c, e in zip(channels, energies)]
            self.residual_axes.axhline(0.0, linewidth=0.8)
            self.residual_axes.errorbar(
                channels, residuals,
                yerr=self._residual_errors(channels, errors, energies),
                fmt="o", capsize=3,
            )
            if left:
                self.residual_axes.errorbar(
                    [p[0] for p in left],
                    [p[4] - calibration.apply(p[0]) for p in left],
                    yerr=self._residual_errors(
                        [p[0] for p in left], [p[1] for p in left],
                        [p[4] for p in left],
                    ),
                    fmt="o", capsize=3, markerfacecolor="none",
                )
            # Scaled to the points the fit was made through, not to the
            # ones left out of it. An excluded point is usually excluded
            # for being far off the line, and letting it set the scale
            # squashed every remaining residual into the middle of the
            # strip -- which is the one thing this strip exists to show.
            # The excluded point is still drawn; it is simply allowed to
            # fall outside the view, and the plot still names it when
            # clicked on the curve above.
            self._scale_residuals(residuals)
        else:
            self.residual_axes.text(
                0.5, 0.5, reason or "no calibration to compare against",
                ha="center", va="center", transform=self.residual_axes.transAxes,
            )
            self.residual_axes.set_yticks([])
        if used or left:
            self.axes.legend(loc="best")
        self.residual_axes.set_xlabel("Channel")
        self.residual_axes.set_ylabel("Residual (keV)")
        self._figure.tight_layout()
        self._draw_pick()
        self.canvas.draw_idle()

        if calibration is not None:
            self.summary_label.setText(self._summary_text(channels, energies, errors))
        else:
            self.summary_label.setText(
                f"no calibration: {reason}" if reason else "no calibration yet"
            )

        self._refresh_efficiency_button()

    def _scale_residuals(self, residuals):
        """Fit the residual strip's y axis around `residuals` alone."""
        if not residuals:
            return
        low, high = min(residuals), max(residuals)
        margin = 0.1 * (high - low)
        if not (margin > 0.0):
            # One point, or a fit passing exactly through all of them.
            margin = max(0.1 * abs(high), 1e-3)
        self.residual_axes.set_ylim(low - margin, high + margin)

    # --- picking a point --------------------------------------------------

    def _pickable(self, axes):
        """[(index, x, y)] of every point drawn in `axes`, excluded ones
        included -- an excluded point is exactly the one a user wants to
        find in the table."""
        if axes is self.axes:
            return [(i, p[0], p[4]) for i, p in enumerate(self._points)]
        if axes is self.residual_axes and self._calibration is not None:
            return [(i, p[0], p[4] - float(self._calibration.apply(p[0])))
                    for i, p in enumerate(self._points)]
        return []

    def _on_click(self, event):
        """Pick the point nearest the click, on either axes."""
        if event.button != 1 or event.inaxes is None:
            return
        if self._figure.canvas.widgetlock.locked():
            return          # a pan or zoom is in progress; not a pick
        nearest, distance = None, _PICK_RADIUS_PX
        for index, x, y in self._pickable(event.inaxes):
            px, py = event.inaxes.transData.transform((x, y))
            gap = math.hypot(px - event.x, py - event.y)
            if gap <= distance:
                nearest, distance = index, gap
        self._picked_channel = (None if nearest is None
                                else self._points[nearest][0])
        self._draw_pick()
        self.canvas.draw_idle()
        self.pointPicked.emit(-1 if nearest is None else nearest)

    def picked_index(self):
        """Index of the picked point among the ones now drawn, or None
        when nothing is picked or the point it was has gone."""
        if self._picked_channel is None:
            return None
        for index, point in enumerate(self._points):
            if abs(point[0] - self._picked_channel) < 1e-9:
                return index
        return None

    def _draw_pick(self):
        """Ring the picked point on both axes and describe it below."""
        for artist in self._picked_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._picked_artists = []
        picked = self.picked_index()
        if picked is None:
            self.picked_label.setText("")
            return
        for axes in (self.axes, self.residual_axes):
            for index, x, y in self._pickable(axes):
                if index == picked:
                    self._picked_artists.extend(axes.plot(
                        [x], [y], "o", markersize=13, markerfacecolor="none",
                        markeredgewidth=2.0, color="tab:red", zorder=5,
                    ))
        self.picked_label.setText(self._picked_text(picked))

    def _picked_text(self, picked):
        channel, channel_err, _area, _area_err, energy = self._points[picked]
        text = f"picked: channel {compact(channel, channel_err)} = {energy:g} keV"
        if self._calibration is not None:
            text += (f", residual "
                     f"{energy - float(self._calibration.apply(channel)):+.4g} keV")
        if any(abs(energy - e) < 1e-9 for e in self._excluded):
            text += " (excluded from the fit)"
        return text

    def _residual_errors(self, channels, channel_errors, energies):
        """Each residual's own sigma, in keV.

        The strip exists to show whether a point sits off the line by
        more than it could. Without these bars a 2-sigma outlier and a
        0.2-sigma one are drawn identically, which is the one
        distinction the panel is for.

        Same combination calibration_quality.reduced_chi_squared
        weights with: the centroid uncertainty carried into keV through
        the calibration's own slope, added in quadrature to what the
        literature states about the line.

        NaN where that combination is not defined, which draws the point
        with no bar at all. It is the same condition reduced_chi_squared
        refuses on, and the two have to agree: at a quadratic's turning
        point dE/dch is zero, so a channel uncertainty maps to no energy
        uncertainty, and if the line carries no stated dE either then
        nothing is left. Drawing that as a zero-length bar would say the
        point is exact -- the opposite of undefined, on precisely the
        point a reader should trust least.
        """
        literature = self._literature_errors(energies)
        out = []
        for channel, sigma_ch, sigma_e in zip(channels, channel_errors, literature):
            try:
                slope = float(self._calibration.derivative(channel))
                variance = (float(sigma_ch) * slope) ** 2 + float(sigma_e) ** 2
            except (TypeError, ValueError):
                variance = float("nan")
            # reduced_chi_squared's own test, so the plot and the number
            # never disagree about which points are usable.
            out.append(math.sqrt(variance)
                       if math.isfinite(variance) and variance > 0.0
                       else float("nan"))
        return out

    def _literature_errors(self, energies):
        """Each point's stated energy uncertainty, from the source lines
        it was assigned from, or zero where it came from nowhere -- a
        hand-typed energy has no dE to quote."""
        out = []
        for energy in energies:
            match = 0.0
            for line in self._source_lines:
                if abs(line.energy - energy) <= _ENERGY_MATCH_KEV:
                    match = float(line.energy_err or 0.0)
                    break
            out.append(match)
        return out

    def _summary_text(self, channels, energies, errors):
        cal = self._calibration
        coefficient_errors = cal.coefficient_errors or ()

        def coefficient(index, value, name):
            error = coefficient_errors[index] if index < len(coefficient_errors) else None
            return f"{name} = {compact(value, error)}"

        parts = [coefficient(0, cal.a, "a"), coefficient(1, cal.b, "b")]
        if cal.kind == "quadratic":
            parts.append(coefficient(2, cal.c, "c"))

        value, reason = reduced_chi_squared(
            cal, channels, energies, errors, self._literature_errors(energies))
        parts.append(
            f"reduced chi^2 = {value:.4g}" if reason is None
            else f"reduced chi^2 {reason}"
        )
        parts.append(f"{len(channels)} points")
        return f"{cal.kind} calibration:  " + ",   ".join(parts)

    def export_to(self, path):
        """Write the CalEnEff file. Returns a summary; raises ExportError."""
        rows, skipped = build_rows(self._points, self._source_lines)
        write_caleneff(path, rows)
        summary = f"Wrote {len(rows)} rows to {os.path.basename(path)}"
        if skipped:
            summary += (
                f" — skipped {skipped} peak(s): no matching source line, "
                f"or no usable uncertainty on either the area or the "
                f"intensity"
            )
        return summary

    def _on_finish(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save for CalEnEff", self._default_path,
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            summary = self.export_to(path)
        except ExportError as exc:
            QMessageBox.warning(self, "Could not export", str(exc))
            return
        QMessageBox.information(self, "Saved", summary)

    # --- Auto MC efficiency calibration ------------------------------------

    def _efficiency_rows(self):
        """The peaks usable for an efficiency fit, as CalEnEff's own seven
        columns.

        Built on caleneff_export.build_rows rather than pairing points to
        source lines again here. That is what makes the in-app fit and the
        exported file agree by construction: a user who exports these peaks
        and runs CalEnEff on them gets the numbers this dialog showed.

        Excluded points are dropped FIRST. One the user rejected from the
        energy calibration must not silently steer the efficiency curve.
        """
        def is_out(energy):
            return any(abs(energy - e) < 1e-9 for e in self._excluded)

        used = [p for p in self._points if not is_out(p[4])]
        try:
            rows, _skipped = build_rows(used, self._source_lines)
        except ExportError:
            return []
        return rows

    def _efficiency_blocked_reason(self):
        """Why the efficiency calibration cannot run, or None.

        Checked here rather than left to fail inside scipy: a fitter's error
        points at the fit, not at the row of data that caused it.
        """
        rows = self._efficiency_rows()
        if len(rows) < self.MIN_EFFICIENCY_POINTS:
            return (
                "Needs at least %d peaks matched to source lines with a "
                "usable uncertainty; this calibration has %d. The Radware "
                "model has 5 free parameters."
                % (self.MIN_EFFICIENCY_POINTS, len(rows)))
        for row in rows:
            if not row.area > 0:
                return ("Every peak needs a positive area; the one at "
                        "%.2f keV does not." % row.energy)
            if not row.energy > 0:
                return ("Every energy must be above 0 keV; both models "
                        "evaluate ln(E) and b/E.")
            if not row.intensity_pct > 0:
                return ("Every source line needs a positive intensity; the "
                        "one at %.2f keV does not." % row.energy)
        return None

    def _refresh_efficiency_button(self):
        reason = self._efficiency_blocked_reason()
        self.efficiency_button.setEnabled(reason is None)
        self.efficiency_button.setToolTip(
            reason or "Fit a relative efficiency curve from these peaks.")

    def _on_efficiency_clicked(self):
        """Fit the efficiency curves, then show them.

        Re-checks the gate rather than trusting the button: the handler is
        reachable by keyboard and by code even when the button is disabled,
        and refusing here is cheaper than a fitter error later.
        """
        reason = self._efficiency_blocked_reason()
        if reason is not None:
            QMessageBox.warning(self, "Cannot fit an efficiency", reason)
            return

        rows = self._efficiency_rows()
        progress = QProgressDialog(
            "Fitting efficiency curves (Monte Carlo)...", "Cancel",
            0, 100, self)
        progress.setWindowTitle("Efficiency calibration")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = EfficiencyWorker(self, rows)
        # Held on self: a QThread that goes out of scope while running is
        # destroyed mid-run, which Qt reports as a crash rather than an
        # error.
        self._efficiency_worker = worker

        def on_progress(done, total):
            progress.setValue(int(100 * done / max(total, 1)))

        def on_succeeded(payload):
            progress.close()
            self._show_efficiency(*payload)

        def on_failed(message):
            progress.close()
            QMessageBox.warning(self, "Efficiency fit failed", message)

        worker.progressed.connect(on_progress)
        worker.succeeded.connect(on_succeeded)
        worker.failed.connect(on_failed)
        progress.canceled.connect(worker.cancel)
        worker.start()

    def _show_efficiency(self, fit, mc):
        from efficiency import EfficiencyResult
        from efficiency_dialog import EfficiencyDialog

        result = EfficiencyResult(
            fit=fit, mc=mc, model="kfr",
            calibration=self._calibration,
            source=os.path.basename(self._default_path or ""))
        energy_errors = self._literature_errors(result.fit.E)
        previous = getattr(self, "_efficiency_dialog", None)
        if previous is not None:
            previous.close()
            previous.deleteLater()
        self._efficiency_dialog = EfficiencyDialog(
            self, result, energy_errors,
            theme=getattr(self.parent(), "_theme", "light"),
            default_path=self._default_path,
            channels=(self._max_channel or 4095) + 1)
        self._efficiency_dialog.setAttribute(
            Qt.WidgetAttribute.WA_DeleteOnClose)
        self._efficiency_dialog.show()
