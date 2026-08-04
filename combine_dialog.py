"""Qt dialog for combining two loaded spectra (Add or Subtract), with
the second scaled by a user-provided factor first -- the two-spectrum-
picker analog of factor_dialog.py's single generic field. Not an
extension of FactorDialog, which is deliberately kept simple and shared
verbatim by Multiply/Rebin; bolting spectrum-pickers onto it would
compromise that reusability."""

import os

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class CombineDialog(QDialog):
    """`result_spectrum_a`, `result_spectrum_b`, `result_factor` are set
    only after a successful OK; stay `None` if cancelled or if every OK
    attempt failed to parse/validate. `spectra` is the full list of
    currently loaded LoadedSpectrum objects -- both dropdowns show all
    of them, unfiltered. `active_spectrum` (may be None) is used only to
    pick Spectrum A's default selection."""

    def __init__(self, parent, title, spectra, active_spectrum):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result_spectrum_a = None
        self.result_spectrum_b = None
        self.result_factor = None
        self._spectra = spectra

        self._combo_a = QComboBox()
        self._combo_b = QComboBox()
        for spectrum in spectra:
            label = os.path.basename(spectrum.path)
            self._combo_a.addItem(label)
            self._combo_b.addItem(label)

        default_a_index = spectra.index(active_spectrum) if active_spectrum in spectra else 0
        self._combo_a.setCurrentIndex(default_a_index)
        # B defaults to the first spectrum that ISN'T A's default, so the
        # two dropdowns never start pointing at the same spectrum.
        default_b_index = 0 if default_a_index != 0 else min(1, len(spectra) - 1)
        self._combo_b.setCurrentIndex(default_b_index)

        self._factor_field = QLineEdit("1")

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Spectrum A:", self._combo_a)
        form.addRow("Spectrum B:", self._combo_b)
        form.addRow("Factor:", self._factor_field)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(button_box)

    def _on_accept(self):
        try:
            factor = float(self._factor_field.text())
        except ValueError:
            QMessageBox.warning(self, self.windowTitle(), "Please enter a valid number.")
            return
        if factor <= 0:
            QMessageBox.warning(self, self.windowTitle(), "Factor must be greater than zero.")
            return
        spectrum_a = self._spectra[self._combo_a.currentIndex()]
        spectrum_b = self._spectra[self._combo_b.currentIndex()]
        if len(spectrum_a.data) != len(spectrum_b.data):
            QMessageBox.warning(
                self, self.windowTitle(),
                f"Spectrum A has {len(spectrum_a.data)} channels; "
                f"Spectrum B has {len(spectrum_b.data)} channels. "
                "Add/Subtract requires both to have the same length.",
            )
            return
        self.result_spectrum_a = spectrum_a
        self.result_spectrum_b = spectrum_b
        self.result_factor = factor
        self.accept()
