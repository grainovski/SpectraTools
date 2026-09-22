"""The efficiency calibration results window.

Shows what CalEnEff shows -- both curves with their Monte Carlo bands, the
residuals, and a readout for any energy -- in this application's own themes
rather than CalEnEff's palette.

The curve drawn is the Monte Carlo MEAN, which is what gets applied and
saved. The best fit is reported beside it when examining a single energy,
because the two differ and the difference is the point of showing both.
"""

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
    QVBoxLayout,
)

from efficiency_io import write_per_bin, write_per_peak
from theme import style_axes

#: Mid-tone on white and on #1e1e1e alike, and clear of the red-to-blue
#: spectrum ramp so a curve is never mistaken for a trace.
KRF_COLOR = "#2AA198"      # teal
RADWARE_COLOR = "#D9822B"  # amber

MODEL_LABELS = (("krf", "KRF"), ("rw", "Radware"))


class EfficiencyDialog(QDialog):
    """A finished efficiency calibration, drawn and queryable.

    Takes an EfficiencyResult that has already been fitted; it runs no
    Monte Carlo of its own.
    """

    def __init__(self, parent, result, energy_errors, theme="light",
                 default_path=None, channels=4096, main_window=None):
        super().__init__(parent)
        #: Where Apply sends its corrected spectrum. Named apart from the
        #: _main_window() method below, which it feeds: an attribute of the
        #: same name silently shadows the method and every call site turns
        #: into "object is not callable".
        self._owner_window = main_window
        self.setWindowTitle("Relative Efficiency")
        self.result = result
        self._energy_errors = np.asarray(energy_errors, dtype=float)
        self._theme = theme
        self._default_path = default_path or "efficiency.txt"
        self._channels = int(channels)

        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(6.5, 5.0))
        self.canvas = FigureCanvasQTAgg(self._figure)
        # Same split as the energy-calibration window: residuals share the
        # x axis and take a third of the height, because they are read
        # against the curve above rather than on their own.
        self.axes, self.residual_axes = self._figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        layout.addWidget(self.canvas)

        row = QHBoxLayout()
        row.addWidget(QLabel("Apply:"))
        self._model_buttons = QButtonGroup(self)
        for key, label in MODEL_LABELS:
            button = QRadioButton(label)
            button.setChecked(key == result.model)
            # Radware may simply not have converged for this data.
            button.setEnabled(key == "krf" or result.fit.rw_params is not None)
            button.toggled.connect(
                lambda checked, k=key: checked and self.select_model(k))
            self._model_buttons.addButton(button)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)

        examine_row = QHBoxLayout()
        examine_row.addWidget(QLabel("Efficiency at (keV):"))
        self.energy_edit = QLineEdit()
        self.energy_edit.setMaximumWidth(120)
        self.energy_edit.returnPressed.connect(self._on_examine)
        examine_row.addWidget(self.energy_edit)
        examine_button = QPushButton("Examine")
        examine_button.clicked.connect(self._on_examine)
        examine_row.addWidget(examine_button)
        examine_row.addStretch(1)
        layout.addLayout(examine_row)

        self.examine_label = QLabel()
        self.examine_label.setWordWrap(True)
        layout.addWidget(self.examine_label)

        # Aim at any loaded spectrum, not only the active one: the spectrum
        # you want corrected is often not the one you were looking at when
        # the calibration finished.
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Apply to:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(220)
        target_row.addWidget(self.target_combo)
        # The only Apply. A second button reading "Apply to active
        # spectrum" used to sit in the bottom row, left over from before
        # this target list existed: it called the same slot, so it corrected
        # whatever the dropdown had selected rather than the active
        # spectrum, and its label said otherwise. It also rebound
        # self.apply_button, leaving THIS button with no surviving
        # reference -- harmless only because nothing enabled or disabled it.
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        target_row.addWidget(self.apply_button)
        self.apply_all_button = QPushButton("Apply to all")
        self.apply_all_button.clicked.connect(self._on_apply_all)
        target_row.addWidget(self.apply_all_button)
        target_row.addStretch(1)
        layout.addLayout(target_row)
        self.refresh_targets()

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.save_button = buttons.addButton(
            "Save efficiency...", QDialogButtonBox.ButtonRole.ActionRole)
        self.save_button.clicked.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._draw()

    # --- the three things the tests drive ----------------------------

    def select_model(self, name):
        """Choose which curve is applied and saved.

        Changing it changes the normalisation too, because that is the
        selected curve's own peak -- so every number on screen moves.
        """
        if name == self.result.model:
            return
        self.result.model = name
        self._draw()

    def examine(self, energy):
        """The readout for one energy, both models, as displayed.

        Reports the Monte Carlo mean and the best fit side by side. They are
        different quantities and CalEnEff shows both; collapsing them would
        hide the disagreement the second number exists to expose.
        """
        lines = ["At %.2f keV" % float(energy)]
        for key, label in MODEL_LABELS:
            if key == "rw" and self.result.fit.rw_params is None:
                lines.append("  %-8s did not converge" % label)
                continue
            mean, sigma, best = self.result.predict(energy, key)
            lines.append(
                "  %-8s %s +- %s   (best fit %s)"
                % (label, _fmt(mean), _fmt(sigma), _fmt(best)))
        return "\n".join(lines)

    def summary_text(self):
        """Fit quality per model, plus how much Monte Carlo survived.

        The accepted counts are on screen rather than implied: a band built
        on 200 surviving samples means something different from one built on
        9,900, and nothing else would reveal which this is.
        """
        fit = self.result.fit
        out = ["KRF      chi2/ndf %.2f/%d   Birge %.3f   RMS %.4g"
               % (fit.krf_chi2, fit.krf_ndf, fit.krf_birge, fit.krf_rms)]
        if fit.rw_params is None:
            out.append("Radware  did not converge")
        else:
            out.append("Radware  chi2/ndf %.2f/%d   Birge %.3f   RMS %.4g"
                       % (fit.rw_chi2, fit.rw_ndf, fit.rw_birge, fit.rw_rms))
        mc = self.result.mc
        if mc is not None:
            out.append(
                "Monte Carlo: KRF %d accepted, Radware %d accepted, "
                "%d rejected" % (mc.krf_accepted, mc.rw_accepted, mc.rejected))
        out.append("Fitted range %.2f - %.2f keV   applied: %s"
                   % (fit.E.min(), fit.E.max(),
                      dict(MODEL_LABELS)[self.result.model]))
        return "\n".join(out)

    # --- drawing ------------------------------------------------------

    def _draw(self):
        result = self.result
        fit = result.fit
        E = fit.E
        grid = np.linspace(float(E.min()) * 0.98, float(E.max()) * 1.02, 400)

        self.axes.clear()
        self.residual_axes.clear()
        style_axes(self.axes, self._theme)
        style_axes(self.residual_axes, self._theme)

        scale = result.normalisation
        self.axes.errorbar(E, fit.eff * scale, yerr=fit.deff * scale,
                           fmt="o", ms=4, capsize=3, elinewidth=1,
                           color=_fg(self._theme), ecolor=_fg(self._theme),
                           label="data", zorder=5)

        for key, label in MODEL_LABELS:
            if key == "rw" and fit.rw_params is None:
                continue
            colour = KRF_COLOR if key == "krf" else RADWARE_COLOR
            style = "-" if key == "krf" else "--"
            with np.errstate(over="ignore", invalid="ignore",
                             divide="ignore"):
                # One MC-mean evaluation, reused as the band's centre.
                # band() recomputes it otherwise, doubling the cost of every
                # redraw for nothing.
                centre = result._mc_mean_raw(grid, key)
                curve = centre * result.normalisation
                lo, hi = result.band(grid, key, centre=centre)
                residual = fit.eff * scale - result.curve(E, key)
            self.axes.plot(grid, curve, style, color=colour, lw=1.8,
                           label=label)
            good = np.isfinite(lo) & np.isfinite(hi)
            if good.any():
                self.axes.fill_between(grid[good], lo[good], hi[good],
                                       color=colour, alpha=0.18, lw=0)
            self.residual_axes.errorbar(
                E, residual, yerr=fit.deff * scale, fmt="o", ms=3,
                capsize=2, elinewidth=0.8, color=colour, ecolor=colour,
                label=label)

        self.axes.set_ylabel("relative efficiency")
        self.axes.legend(fontsize=8)
        self.residual_axes.axhline(0.0, lw=0.8, color=_fg(self._theme))
        self.residual_axes.set_ylabel("residual")
        self.residual_axes.set_xlabel("Energy (keV)")

        self.summary_label.setText(self.summary_text())
        self._figure.tight_layout()
        self.canvas.draw()

    # --- handlers -----------------------------------------------------

    def _on_examine(self):
        text = self.energy_edit.text().strip()
        try:
            energy = float(text)
        except ValueError:
            self.examine_label.setText("Enter an energy in keV.")
            return
        self.examine_label.setText(self.examine(energy))

    def _on_save(self):
        """Writes BOTH files from one dialog, since they are two views of
        one calibration and saving only half of it is never what is
        wanted."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save efficiency", self._default_path,
            "Text files (*.txt);;All files (*)")
        if not path:
            return
        stem, ext = os.path.splitext(path)
        ext = ext or ".txt"
        try:
            write_per_peak(stem + "_peaks" + ext, self.result,
                           self._energy_errors)
            if self.result.calibration is not None:
                write_per_bin(stem + "_bins" + ext, self.result,
                              self.result.calibration, self._channels)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        QMessageBox.information(
            self, "Efficiency saved",
            "Wrote %s_peaks%s%s" % (os.path.basename(stem), ext,
                                    "" if self.result.calibration is None
                                    else " and _bins" + ext))

    def refresh_targets(self):
        """Rebuild the target list from the window's current spectra.

        Called on open and after every Apply, because applying ADDS a
        spectrum -- the list the user is looking at goes stale the moment
        they use it.
        """
        window = self._main_window()
        self.target_combo.clear()
        if window is None:
            return
        active_index = 0
        for index, spectrum in enumerate(window.spectra):
            self.target_combo.addItem(spectrum.path)
            if getattr(spectrum, "active", False):
                active_index = index
        if self.target_combo.count():
            self.target_combo.setCurrentIndex(active_index)

    def _selected_spectrum(self):
        window = self._main_window()
        index = self.target_combo.currentIndex()
        if window is None or index < 0 or index >= len(window.spectra):
            return None
        return window.spectra[index]

    def _ready(self):
        """The window and its calibration, or None after explaining why."""
        window = self._main_window()
        if window is None:
            QMessageBox.warning(
                self, "Nothing to apply to",
                "This efficiency window is not attached to a main window.")
            return None
        if not window.can_apply_efficiency():
            QMessageBox.warning(
                self, "No active calibration",
                "Applying an efficiency needs an energy for every bin, "
                "which only an active energy calibration provides.")
            return None
        return window

    def _drift_accepted(self, window):
        """Ask before applying through a calibration this was not fitted
        under. The same eps(E) through a different channel-to-energy map is
        a different correction, and nothing else would reveal it: the
        numbers look reasonable either way."""
        if (self.result.calibration is None
                or window._calibration == self.result.calibration):
            return True
        answer = QMessageBox.question(
            self, "Calibration has changed",
            "This efficiency was derived under a different energy "
            "calibration than the one now active. The same curve applied "
            "through a different channel-to-energy map is a different "
            "correction.\n\nApply it anyway?")
        return answer == QMessageBox.StandardButton.Yes

    def _main_window(self):
        """Walk up to whatever can apply an efficiency.

        Given explicitly when this window is parentless, which it now is:
        a widget parent would make it Win32-owned and pin it above the
        window that opened it. The walk is kept for callers that still
        pass a parent -- it finds the target by capability rather than by
        counting levels, so it survives re-parenting either way.
        """
        if self._owner_window is not None:
            return self._owner_window
        widget = self.parent()
        while widget is not None and not hasattr(widget, "apply_efficiency"):
            widget = widget.parent()
        return widget

    def _on_apply(self):
        """Divide the PICKED spectrum by this curve, into a new spectrum."""
        window = self._ready()
        if window is None:
            return
        spectrum = self._selected_spectrum()
        if spectrum is None:
            QMessageBox.warning(self, "No spectrum selected",
                                "Choose a spectrum to correct first.")
            return
        if not self._drift_accepted(window):
            return
        window.apply_efficiency(spectrum, self.result)
        self.refresh_targets()

    def _on_apply_all(self):
        """Correct every loaded spectrum that is not already a correction.

        Dividing an already-corrected spectrum a second time is physically
        meaningless, and pressing this twice is an easy thing to do, so
        those are skipped and counted rather than silently doubled.
        """
        window = self._ready()
        if window is None:
            return
        if not self._drift_accepted(window):
            return
        # Two reasons to skip, and they are different mistakes. A
        # correction must never be corrected again -- that is a double
        # correction and physically meaningless. And a spectrum whose
        # correction under this model already exists is skipped so that
        # pressing this button twice is a no-op rather than a way to fill
        # the plot with identical copies.
        existing = {s.path for s in window.spectra}
        targets = [
            s for s in list(window.spectra)
            if not getattr(s, "efficiency_corrected", False)
            and window.efficiency_label(s, self.result) not in existing
        ]
        skipped = len(window.spectra) - len(targets)
        if not targets:
            QMessageBox.information(
                self, "Nothing to correct",
                "Every loaded spectrum is already an efficiency correction.")
            return
        for spectrum in targets:
            window.apply_efficiency(spectrum, self.result)
        self.refresh_targets()
        if skipped:
            QMessageBox.information(
                self, "Efficiency applied",
                "Corrected %d spectrum(s). Skipped %d that were already "
                "efficiency corrections." % (len(targets), skipped))


def _fg(theme):
    return "#e0e0e0" if theme == "dark" else "black"


def _fmt(value):
    """Six significant figures, or a plain marker when there is no number.

    nan reaches here whenever fewer than MIN_MC_SAMPLES of the Monte Carlo
    family survived at that energy, which is a real answer and not a bug.
    """
    return "n/a" if value is None or not np.isfinite(value) else "%.6g" % value
