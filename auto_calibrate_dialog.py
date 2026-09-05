"""Automatic calibration: find the peaks, fit them, and identify them
against a source file with no calibration to start from.

A thin UI layer, as goto_dialog.py is. The search, the fitting and the
matching live in peak_search and auto_calibrate, which are Qt-free and
tested against spectra whose answer is known. This dialog collects a
source file and a sensitivity, shows how many peaks that sensitivity
finds while the user adjusts it, and on Run hands an
auto_calibrate.AutoCalibration back to main_window -- which commits the
fits and opens the ordinary Calibrate from Fitted Peaks dialog on them,
prefilled. That dialog already has the live plot, the include/exclude
ticks and Suggest, so nothing here duplicates them.
"""

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

import auto_calibrate
import peak_search
from histogram_io import ParseError
from peak_fit import channel_indices
from sou_io import load_sou

#: The spin box's range, in sigma above the local continuum. Below two
#: nearly every noise ripple is a peak and the fitting pass grows long on
#: a busy spectrum; above thirty only the strongest few lines survive,
#: too few to calibrate from. The default sits where a person would draw
#: the line -- see peak_search.DEFAULT_SENSITIVITY.
MIN_SENSITIVITY = 2.0
MAX_SENSITIVITY = 30.0
SENSITIVITY_STEP = 0.5


class AutoCalibrateDialog(QDialog):
    """`outcome` is an auto_calibrate.AutoCalibration, set only after a
    successful Run; the caller commits its fits and reads its match.

    `counts` are the active spectrum's counts and `variance` its
    propagated per-channel variance, or None for counts read from a
    file. `settings`, when given, is the app's Settings: the source
    picker starts in and remembers the same last folder as every other
    file dialog. `source_path` is a .sou the user already chose for this
    spectrum; it is loaded up front so it need not be picked twice.
    """

    def __init__(self, parent, counts, settings=None, variance=None,
                 source_path=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Calibration")
        self.outcome = None
        self._counts = np.asarray(counts, dtype=float)
        self._variance = variance
        self._settings = settings
        #: Lines of the loaded .sou file, or None before one is loaded.
        self.source_lines = None
        self._source_path = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Finds the peaks of the active spectrum, fits them, and works "
            "out which line of the source each one is -- no calibration is "
            "needed to start. The fits are added to the spectrum; anything "
            "already fitted is kept. The result opens in the Calibrate from "
            "Fitted Peaks dialog for review."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        source_row = QHBoxLayout()
        self.load_source_button = QPushButton("Load source...")
        self.load_source_button.clicked.connect(self._on_load_source_clicked)
        source_row.addWidget(self.load_source_button)
        self.source_label = QLabel("No source loaded")
        source_row.addWidget(self.source_label)
        source_row.addStretch(1)
        layout.addLayout(source_row)

        sensitivity_row = QHBoxLayout()
        sensitivity_row.addWidget(QLabel("Sensitivity:"))
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(MIN_SENSITIVITY, MAX_SENSITIVITY)
        self.sensitivity.setSingleStep(SENSITIVITY_STEP)
        self.sensitivity.setDecimals(1)
        self.sensitivity.setSuffix(" sigma")
        self.sensitivity.setValue(peak_search.DEFAULT_SENSITIVITY)
        self.sensitivity.setToolTip(
            "How many standard deviations a peak must stand above the "
            "continuum around it. Lower finds more, including noise; higher "
            "finds only the strongest lines."
        )
        self.sensitivity.valueChanged.connect(self._on_sensitivity_changed)
        sensitivity_row.addWidget(self.sensitivity)
        # Live, so the number is seen while choosing rather than after a
        # run that fitted two hundred noise ripples.
        self.peaks_label = QLabel("")
        sensitivity_row.addWidget(self.peaks_label)
        sensitivity_row.addStretch(1)
        layout.addLayout(sensitivity_row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.run_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.run_button.setText("Run")
        buttons.accepted.connect(self._on_run)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_sensitivity_changed(self.sensitivity.value())
        self._update_run_availability()
        if source_path and os.path.exists(source_path):
            self.load_source(source_path)

    # --- what the user can read off the dialog -----------------------------

    def found_count(self):
        """How many peaks the current sensitivity finds."""
        return self._found_count

    def source_path(self):
        """The loaded .sou path, or None."""
        return self._source_path

    def source_name(self):
        """The loaded .sou file's name, or None."""
        return os.path.basename(self._source_path) if self._source_path else None

    # --- sensitivity ------------------------------------------------------

    def _on_sensitivity_changed(self, value):
        self._found_count = len(peak_search.search(self._counts, float(value)))
        noun = "peak" if self._found_count == 1 else "peaks"
        self.peaks_label.setText(f"{self._found_count} {noun} found")

    # --- source loading ---------------------------------------------------

    def _on_load_source_clicked(self):
        directory = self._settings.last_folder() if self._settings else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Source", directory, "Source files (*.sou);;All files (*)"
        )
        if path:
            self.load_source(path)

    def load_source(self, path):
        """Load a .sou file as the source to identify against. Returns
        True on success; on failure the message goes to the status line
        and any previously loaded source is kept."""
        try:
            lines = load_sou(path)
        except ParseError as exc:
            self.status.setText(str(exc))
            return False
        self.source_lines = lines
        self._source_path = path
        self.source_label.setText(f"{os.path.basename(path)}: {len(lines)} lines")
        if self._settings is not None:
            self._settings.set_last_folder(os.path.dirname(path))
        self.status.setText("")
        self._update_run_availability()
        return True

    def _update_run_availability(self):
        ready = self.source_lines is not None
        self.run_button.setEnabled(ready)
        self.run_button.setToolTip("" if ready else "Needs a loaded source file")

    # --- running ----------------------------------------------------------

    def _on_run(self):
        if self.source_lines is None:
            self.status.setText("Load a source file first.")
            return
        counts = self._counts
        # Fitting a busy spectrum takes a second or two; the cursor says
        # so, since nothing else on screen changes until it is done.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            outcome = auto_calibrate.calibrate(
                channel_indices(len(counts)), counts, self.source_lines,
                sensitivity=self.sensitivity.value(), variance=self._variance,
            )
        finally:
            QApplication.restoreOverrideCursor()
        if not outcome.results:
            # Nothing to commit -- no peaks at this sensitivity, or none
            # that would fit. Say which and stay open, so the sensitivity
            # can be changed and Run pressed again.
            reason = outcome.match.reason
            self.status.setText(reason[0].upper() + reason[1:] + ".")
            return
        # A refused match still comes back with its fits: they are real
        # fits of real peaks, and the caller opens them for assignment
        # by hand with the reason on show.
        self.outcome = outcome
        self.accept()
