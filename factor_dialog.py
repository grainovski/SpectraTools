"""Qt dialog for a single validated numeric factor -- shared by
Multiply by Factor and Rebin by Factor, the thin UI layer with no
pure-logic counterpart of its own (there's no logic here beyond
parsing/validating one field), mirroring calibration_dialog.py's
QMessageBox-on-error style."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class FactorDialog(QDialog):
    """`result_factor` is set only after a successful OK; stays `None`
    if cancelled or if every OK attempt failed to parse/validate.
    `parse(text) -> value` should raise ValueError on unparseable text.
    `validate(value) -> str | None` returns an error message for an
    unacceptable value, or None if it's fine."""

    def __init__(self, parent, title, label, parse, validate):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result_factor = None
        self._parse = parse
        self._validate = validate

        self._field = QLineEdit()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow(label, self._field)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(button_box)

    def _on_accept(self):
        try:
            value = self._parse(self._field.text())
        except ValueError:
            QMessageBox.warning(self, self.windowTitle(), "Please enter a valid number.")
            return
        error = self._validate(value)
        if error is not None:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self.result_factor = value
        self.accept()
