"""Assign known energies to fitted peaks, then calibrate from them.

The calibration this produces is anchored on FITTED peak centroids, so it
inherits the fits' precision rather than the user's aim with a cursor --
which is the difference between a calibration that is a physics
measurement and one that is a clerical exercise.

Modelled on HDTV's "fit position assign" followed by "calibration
position recalibrate", which is the same loop: attach literature energies
to peaks you have already fitted, refit the calibration from every
assignment, and repeat as peaks are added or corrected.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import math

import calibration as calibration_module
from calibration import CalibrationError


def _format_uncertainty(value):
    """A centroid uncertainty for display, or an em dash where there is
    none to show.

    A position held fixed reports exactly 0.0 and one the fit could not
    determine reports NaN. Neither is a measurement, and printing "0.000"
    for the first would read as a perfectly known channel -- the most
    misleading thing this column could say.
    """
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number) or number <= 0.0:
        return "—"
    return f"{number:.3f}"


class EnergyAssignDialog(QDialog):
    """`result_calibration` is set only after a successful OK.

    `peaks` is a list of (label, channel) for every committed peak on the
    active spectrum, in the order the Fit Results panel shows them.
    """

    # Shifted right by one when the centroid-uncertainty column was added.
    # Every caller reads the constant rather than the literal, which is why
    # that column could be inserted at all.
    ENERGY_COLUMN = 3

    def __init__(self, parent, peaks, quadratic=False):
        super().__init__(parent)
        self.setWindowTitle("Calibrate from Fitted Peaks")
        self.result_calibration = None
        # Entries are (label, channel) or (label, channel, channel_err).
        # Both accepted: the uncertainty is extra information about a peak,
        # not part of what identifies it, and a caller that has none should
        # not have to invent one.
        self._peaks = [
            (entry[0], entry[1], entry[2] if len(entry) > 2 else None)
            for entry in peaks
        ]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Type the known energy for the peaks you can identify and leave "
            "the rest blank."
        ))

        self.table = QTableWidget(len(self._peaks), 4)
        self.table.setHorizontalHeaderLabels(
            ["Peak", "Channel", "± ch", "Energy (keV)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for row, (label, channel, channel_err) in enumerate(self._peaks):
            name = QTableWidgetItem(label)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, name)

            position = QTableWidgetItem(f"{channel:.3f}")
            position.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, position)

            # Shown, not just used: it is the reason one row deserves more
            # weight than another, and seeing a peak with a large
            # uncertainty explains a calibration that leans away from it.
            uncertainty = QTableWidgetItem(_format_uncertainty(channel_err))
            uncertainty.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 2, uncertainty)

            self.table.setItem(row, self.ENERGY_COLUMN, QTableWidgetItem(""))
        layout.addWidget(self.table)

        self.quadratic_checkbox = QCheckBox("Quadratic (needs 3 or more assignments)")
        self.quadratic_checkbox.setChecked(quadratic)
        layout.addWidget(self.quadratic_checkbox)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def assignments(self):
        """(channels, energies, channel_errors) for the rows that carry an
        energy.

        Raises ValueError naming the row for anything unparseable, rather
        than skipping it: a typo silently dropped would produce a
        calibration quietly fitted to fewer points than the user thinks.
        """
        channels, energies, errors = [], [], []
        for row, (label, channel, channel_err) in enumerate(self._peaks):
            item = self.table.item(row, self.ENERGY_COLUMN)
            text = (item.text() if item else "").strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a number") from None
            # float() alone accepts "nan"/"inf", which from_points would
            # only reject later with a message about the whole fit's
            # coefficients -- name the offending row here instead, exactly
            # like an unparseable entry.
            if not math.isfinite(value):
                raise ValueError(f"{label}: {text!r} is not a finite energy") from None
            energies.append(value)
            channels.append(channel)
            errors.append(channel_err)
        return channels, energies, errors

    def _on_accept(self):
        try:
            channels, energies, errors = self.assignments()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        try:
            self.result_calibration = calibration_module.from_points(
                channels, energies, self.quadratic_checkbox.isChecked(),
                # from_points falls back to an unweighted fit if any of
                # these is missing or unusable, so passing them
                # unconditionally is safe.
                channel_errors=errors,
            )
        except CalibrationError as exc:
            self.status.setText(str(exc))
            return

        # Residuals are reported rather than only the coefficients: one
        # mistyped energy moves the whole fit and is obvious here while
        # being invisible in a and b.
        worst = max(
            abs(r) for r in calibration_module.residuals(
                self.result_calibration, channels, energies
            )
        ) if channels else 0.0
        self._worst_residual = worst
        self.accept()
