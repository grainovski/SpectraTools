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

import calibration as calibration_module
from calibration import CalibrationError


class EnergyAssignDialog(QDialog):
    """`result_calibration` is set only after a successful OK.

    `peaks` is a list of (label, channel) for every committed peak on the
    active spectrum, in the order the Fit Results panel shows them.
    """

    ENERGY_COLUMN = 2

    def __init__(self, parent, peaks, quadratic=False):
        super().__init__(parent)
        self.setWindowTitle("Calibrate from Fitted Peaks")
        self.result_calibration = None
        self._peaks = list(peaks)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Type the known energy for the peaks you can identify and leave "
            "the rest blank."
        ))

        self.table = QTableWidget(len(self._peaks), 3)
        self.table.setHorizontalHeaderLabels(["Peak", "Channel", "Energy (keV)"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for row, (label, channel) in enumerate(self._peaks):
            name = QTableWidgetItem(label)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, name)

            position = QTableWidgetItem(f"{channel:.3f}")
            position.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, position)

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
        """(channels, energies) for the rows that carry an energy.

        Raises ValueError naming the row for anything unparseable, rather
        than skipping it: a typo silently dropped would produce a
        calibration quietly fitted to fewer points than the user thinks.
        """
        channels, energies = [], []
        for row, (label, channel) in enumerate(self._peaks):
            item = self.table.item(row, self.ENERGY_COLUMN)
            text = (item.text() if item else "").strip()
            if not text:
                continue
            try:
                energies.append(float(text))
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a number") from None
            channels.append(channel)
        return channels, energies

    def _on_accept(self):
        try:
            channels, energies = self.assignments()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        try:
            self.result_calibration = calibration_module.from_points(
                channels, energies, self.quadratic_checkbox.isChecked()
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
