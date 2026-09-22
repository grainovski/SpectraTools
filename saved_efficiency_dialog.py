"""The window for an efficiency read back from a saved file.

A saved efficiency can be applied exactly like a freshly fitted one, and
through exactly the same rules -- see efficiency_dialog.ApplyTargetsMixin.
What it cannot do is show a fit: the files hold the finished curve, not the
measured points or the Monte Carlo samples behind it. So this window is the
fitted one's Apply half, plus the one thing only a saved file offers -- the
energy calibration the efficiency was made under, which is otherwise not
saved anywhere.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QVBoxLayout,
)

from efficiency_dialog import MODEL_LABELS, ApplyTargetsMixin


def describe_calibration(calibration):
    """One line a reader can check against their own notes."""
    if calibration is None:
        return "none recorded"
    if calibration.kind == "linear":
        return "linear, E = %.10g + %.10g*ch" % (calibration.a, calibration.b)
    return ("quadratic, E = %.10g + %.10g*ch + %.6g*ch^2"
            % (calibration.a, calibration.b, calibration.c))


class SavedEfficiencyDialog(ApplyTargetsMixin, QDialog):
    """Apply a saved efficiency, optionally with its energy calibration.

    Created parentless, like every other tool window here: a widget parent
    makes a window Win32-owned, and Windows then keeps it above its owner no
    matter what is clicked. `main_window` is given explicitly instead.
    """

    def __init__(self, parent, saved, main_window=None):
        super().__init__(parent)
        self.result = saved
        self._owner_window = main_window
        self.setWindowTitle("Saved efficiency -- %s"
                            % os.path.basename(saved.path))

        layout = QVBoxLayout(self)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Apply:"))
        self.model_buttons = {}
        for key, label in MODEL_LABELS:
            button = QRadioButton(label)
            button.setEnabled(key in saved.available_models)
            button.setChecked(key == saved.model)
            button.toggled.connect(
                lambda checked, k=key: checked and self.select_model(k))
            self.model_buttons[key] = button
            model_row.addWidget(button)
        model_row.addStretch(1)
        layout.addLayout(model_row)

        self.calibration_button = QPushButton("Use its energy calibration")
        self.calibration_button.setEnabled(saved.calibration is not None)
        self.calibration_button.setToolTip(
            "Make the energy calibration this efficiency was made under the "
            "active one: %s" % describe_calibration(saved.calibration))
        self.calibration_button.clicked.connect(self.use_its_calibration)
        layout.addWidget(self.calibration_button)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Apply to:"))
        self.target_combo = QComboBox()
        target_row.addWidget(self.target_combo, 1)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        target_row.addWidget(self.apply_button)
        self.apply_all_button = QPushButton("Apply to all")
        self.apply_all_button.clicked.connect(self._on_apply_all)
        target_row.addWidget(self.apply_all_button)
        layout.addLayout(target_row)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

        self.refresh_targets()
        self._refresh_summary()

    def select_model(self, name):
        self.result.model = name
        self._refresh_summary()

    def summary_text(self):
        saved = self.result
        lines = ["File: %s" % saved.path]
        if saved.source:
            lines.append("Made from: %s" % saved.source)
        fitted = saved.field("fitted energy range")
        if fitted:
            lines.append("Fitted over: %s" % fitted)
        for key, label in MODEL_LABELS:
            if key not in saved.available_models:
                lines.append("%s: not in this file" % label)
                continue
            stats = saved.statistic(key, "chi2/ndf")
            lines.append("%s: chi2/ndf %s" % (label, stats) if stats
                         else "%s: available" % label)
        lines.append("Energy calibration it was made with: %s"
                     % describe_calibration(saved.calibration))
        lo, hi = saved.energy_range
        lines.append(
            "The %s curve is saved from %.4g to %.4g keV. Bins outside that "
            "range are zeroed when it is applied, since nothing was saved "
            "for them." % (dict(MODEL_LABELS)[saved.model], lo, hi))
        return "\n".join(lines)

    def _refresh_summary(self):
        self.summary_label.setText(self.summary_text())

    def use_its_calibration(self):
        """Make the saved energy calibration the active one.

        Goes through the main window's ordinary calibration path, so the
        view keeps its place, open matrix panels follow, and the warning for
        a quadratic that folds inside the spectrum still fires.
        """
        window = self._main_window()
        if window is None or self.result.calibration is None:
            return
        window.use_calibration(self.result.calibration)
        self.calibration_button.setText("Its energy calibration is active")
        self.calibration_button.setEnabled(False)
