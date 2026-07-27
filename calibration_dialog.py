"""Qt dialog for setting a channel/energy Calibration -- the thin UI
layer on top of calibration.py's pure logic."""

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from calibration import Calibration, CalibrationError, CalibrationFileError, read_coefficients_file


class CalibrationDialog(QDialog):
    """`result_calibration`/`result_active` are set only after a
    successful OK (`_on_accept` calls `self.accept()`); both stay at
    their initial `None`/`False` if the dialog is cancelled or closed
    without a valid OK."""

    def __init__(self, parent, initial=None, initially_active=False):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.result_calibration = None
        self.result_active = False

        self._linear_radio = QRadioButton("Linear")
        self._quadratic_radio = QRadioButton("Quadratic")
        self._linear_radio.setChecked(True)
        self._quadratic_radio.toggled.connect(self._update_c_field_visibility)
        self._linear_radio.toggled.connect(self._update_c_field_visibility)

        self._a_field = QLineEdit()
        self._b_field = QLineEdit()
        self._c_field = QLineEdit()
        self._c_label = QLabel("c:")

        self._load_button = QPushButton("Load from file...")
        self._load_button.clicked.connect(self._on_load_file)

        self._active_checkbox = QCheckBox("Active")

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        type_row = QHBoxLayout()
        type_row.addWidget(self._linear_radio)
        type_row.addWidget(self._quadratic_radio)

        form = QFormLayout()
        form.addRow("a:", self._a_field)
        form.addRow("b:", self._b_field)
        form.addRow(self._c_label, self._c_field)

        layout = QVBoxLayout(self)
        layout.addLayout(type_row)
        layout.addLayout(form)
        layout.addWidget(self._load_button)
        layout.addWidget(self._active_checkbox)
        layout.addWidget(button_box)

        if initial is not None:
            self._quadratic_radio.setChecked(initial.kind == "quadratic")
            self._linear_radio.setChecked(initial.kind == "linear")
            self._a_field.setText(str(initial.a))
            self._b_field.setText(str(initial.b))
            self._c_field.setText(str(initial.c))
        self._active_checkbox.setChecked(initially_active)
        self._update_c_field_visibility()

    def _update_c_field_visibility(self):
        is_quadratic = self._quadratic_radio.isChecked()
        self._c_field.setVisible(is_quadratic)
        self._c_label.setVisible(is_quadratic)

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Coefficients", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            values = read_coefficients_file(path, quadratic=self._quadratic_radio.isChecked())
        except CalibrationFileError as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        self._a_field.setText(str(values[0]))
        self._b_field.setText(str(values[1]))
        if len(values) > 2:
            self._c_field.setText(str(values[2]))

    def _on_accept(self):
        try:
            a = float(self._a_field.text())
            b = float(self._b_field.text())
            c = float(self._c_field.text()) if self._quadratic_radio.isChecked() else 0.0
        except ValueError:
            QMessageBox.warning(self, "Calibration", "Coefficients must be valid numbers.")
            return
        kind = "quadratic" if self._quadratic_radio.isChecked() else "linear"
        try:
            calibration = Calibration(kind=kind, a=a, b=b, c=c)
        except CalibrationError as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        self.result_calibration = calibration
        self.result_active = self._active_checkbox.isChecked()
        self.accept()
