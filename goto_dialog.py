"""Qt dialog for Go To -- jump the view to one energy or channel.

A thin UI layer with no pure-logic counterpart of its own, as
factor_dialog.py is: the only logic here is parsing and range-checking a
single field. The view arithmetic lives in spectrum.goto_channel_window,
which is Qt-free and shared by the main window and the matrix panel.
"""

import math

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class GoToDialog(QDialog):
    """`result_value` is the number the user entered, in whatever unit the
    x-axis is currently showing, and is set only after a successful OK.

    `calibrated` decides the unit the dialog asks for, and it is the
    caller's view state that decides it, not whether a calibration merely
    exists -- a calibration that is loaded but switched off leaves the axis
    in channels, and asking for keV then would be asking for a number the
    view cannot use.

    `lo`/`hi` bound what the axis can currently show, in that same unit, and
    are used both to range-check the entry and to say what the valid range
    IS when it fails. They arrive already ordered: a negative calibration
    slope makes the display axis descend, so the caller sorts them rather
    than every reader of this class having to remember that.
    """

    def __init__(self, parent, calibrated, lo, hi, initial=None):
        super().__init__(parent)
        self.setWindowTitle("Go To")
        self.result_value = None
        self._calibrated = calibrated
        self._lo = float(lo)
        self._hi = float(hi)

        self._unit = "keV" if calibrated else "channel"
        label = "Energy (keV):" if calibrated else "Channel:"

        self._field = QLineEdit()
        if initial is not None:
            self._field.setText(f"{float(initial):g}")
        self._field.selectAll()

        form = QFormLayout()
        form.addRow(label, self._field)

        # The range is shown up front rather than only in an error, so the
        # user knows what the spectrum covers before typing -- the useful
        # moment for it is while deciding, not after being refused.
        hint = QLabel(f"Range: {self._lo:g} to {self._hi:g} {self._unit}")
        hint.setWordWrap(True)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(button_box)

        self._field.setFocus()

    def _on_accept(self):
        text = self._field.text().strip()
        try:
            value = float(text)
        except ValueError:
            QMessageBox.warning(
                self, "Go To",
                f"Please enter a number ({'energy in keV' if self._calibrated else 'channel'}).",
            )
            return
        if not math.isfinite(value):
            # float() accepts "nan"/"inf" happily, and either would send the
            # view somewhere it cannot come back from.
            QMessageBox.warning(self, "Go To", "Please enter a finite number.")
            return
        if not (self._lo <= value <= self._hi):
            QMessageBox.warning(
                self, "Go To",
                f"{value:g} {self._unit} is outside this spectrum, which covers "
                f"{self._lo:g} to {self._hi:g} {self._unit}.",
            )
            return
        self.result_value = value
        self.accept()
