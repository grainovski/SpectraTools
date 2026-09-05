"""Shows an energy calibration as a fit, not as a status line.

A calibration's coefficients look equally plausible whether or not one
of its points was misidentified. The residual strip is where that shows
up, which is why this window exists at all rather than a message box
reporting a and b.
"""

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from caleneff_export import ExportError, build_rows, write_caleneff
from calibration_quality import reduced_chi_squared
from value_format import compact

#: Points drawn along the fitted curve. Enough that a quadratic reads as
#: a curve rather than a polyline at any zoom the dialog offers.
_CURVE_SAMPLES = 400


class CalibrationPlotDialog(QDialog):
    """`points` is (channel, channel_err, area, area_err, energy) per
    assigned peak. `source_lines` are the .sou lines they were assigned
    from, used only by the export."""

    def __init__(self, parent, calibration, points, source_lines,
                 max_channel, default_path, excluded=()):
        super().__init__(parent)
        self.setWindowTitle("Energy Calibration")
        self._default_path = default_path
        self._max_channel = max_channel

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
        self.summary_label.setTextInteractionFlags(
            self.summary_label.textInteractionFlags()
        )
        layout.addWidget(self.summary_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.finish_button = buttons.addButton(
            "Finish and save for CalEnEff...",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.finish_button.clicked.connect(self._on_finish)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.set_data(calibration, points, source_lines, excluded=excluded)

    def set_data(self, calibration, points, source_lines, max_channel=None,
                 excluded=()):
        """Redraw for a new calibration without rebuilding the window.

        The Calibrate dialog's live preview refreshes after every change
        to the assignments. Closing and recreating the window each time
        would raise it to the front and take focus away from the table
        the user is still typing into, so the same window is redrawn
        instead.
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
        upper = max(self._max_channel, max(channels) if channels else 0)
        grid = np.linspace(0.0, float(upper), _CURVE_SAMPLES)
        self.axes.plot(grid, calibration.apply(grid), "-", label="calibration")
        self.axes.set_ylabel("Energy (keV)")
        self.axes.legend(loc="best")

        residuals = [e - calibration.apply(c) for c, e in zip(channels, energies)]
        self.residual_axes.axhline(0.0, linewidth=0.8)
        self.residual_axes.errorbar(channels, residuals, fmt="o", capsize=3)
        if left:
            self.residual_axes.errorbar(
                [p[0] for p in left],
                [p[4] - calibration.apply(p[0]) for p in left],
                fmt="o", capsize=3, markerfacecolor="none",
            )
        self.residual_axes.set_xlabel("Channel")
        self.residual_axes.set_ylabel("Residual (keV)")
        self._figure.tight_layout()
        self.canvas.draw_idle()

        self.summary_label.setText(self._summary_text(channels, energies, errors))

    def _summary_text(self, channels, energies, errors):
        cal = self._calibration
        coefficient_errors = cal.coefficient_errors or ()

        def coefficient(index, value, name):
            error = coefficient_errors[index] if index < len(coefficient_errors) else None
            return f"{name} = {compact(value, error)}"

        parts = [coefficient(0, cal.a, "a"), coefficient(1, cal.b, "b")]
        if cal.kind == "quadratic":
            parts.append(coefficient(2, cal.c, "c"))

        value, reason = reduced_chi_squared(cal, channels, energies, errors)
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
                f" — skipped {skipped} peak(s) with no source line, so no "
                f"intensity was available for them"
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
