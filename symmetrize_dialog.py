"""The Symmetrize... dialog: how Y is put on X's scale before folding.

Only collects and checks the map x = a + b*y; the matrix work itself is
matrix_symmetrize's. Kept free of exec()-time logic so a test can set the
fields and read `mapping` without running a modal loop.
"""

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QGridLayout, QLabel, QRadioButton, QVBoxLayout,
)

from channel_axis import to_channel, to_index
from matrix_symmetrize import SymmetrizeError, map_from_lines, refine_centroid


class SymmetrizeDialog(QDialog):
    def __init__(self, parent, projection_x, projection_y, axes=None):
        super().__init__(parent)
        self.setWindowTitle("Symmetrize Matrix")
        self._px, self._py = projection_x, projection_y
        # Positions are typed in detector CHANNELS -- what the projections
        # show (see channel_axis) -- and turned into bins for the fold,
        # which works on the matrix's own bins.
        self._x_axis, self._y_axis = axes if axes is not None else (None, None)
        #: (a, b) of x = a + b*y once the inputs are valid, else None.
        self.mapping = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Builds a symmetric matrix S = M + Mᵀ in a new window; this one "
            "is left as it is.\nA matrix from two detectors has a different "
            "gain on each axis, so Y must first be put on X's scale, or every "
            "line would appear twice.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.match_radio = QRadioButton(
            "Match Y to X using two lines (two detectors)")
        self.same_radio = QRadioButton(
            "Both axes already share one scale (single array)")
        group = QButtonGroup(self)
        group.addButton(self.match_radio)
        group.addButton(self.same_radio)
        self.match_radio.setChecked(True)
        layout.addWidget(self.match_radio)

        grid = QGridLayout()
        grid.addWidget(QLabel("Same line seen on"), 0, 0)
        grid.addWidget(QLabel("X projection (channel)"), 0, 1)
        grid.addWidget(QLabel("Y projection (channel)"), 0, 2)
        self.x_boxes, self.y_boxes = [], []
        for row in (1, 2):
            grid.addWidget(QLabel(f"Line {row}"), row, 0)
            for col, (proj, boxes, axis) in enumerate(
                    ((self._px, self.x_boxes, self._x_axis),
                     (self._py, self.y_boxes, self._y_axis)), start=1):
                box = QDoubleSpinBox()
                box.setDecimals(1)
                box.setRange(max(0.0, float(to_channel(axis, 0.0))),
                             float(to_channel(axis, len(proj) - 1.0)))
                box.valueChanged.connect(self._recompute)
                grid.addWidget(box, row, col)
                boxes.append(box)
        layout.addLayout(grid)

        self.refine_check = QCheckBox(
            "Refine each position to the nearby peak centroid")
        self.refine_check.setChecked(True)
        self.refine_check.toggled.connect(self._recompute)
        layout.addWidget(self.refine_check)

        layout.addWidget(self.same_radio)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.match_radio.toggled.connect(self._recompute)
        self._recompute()

    def _lines(self):
        """(xs, ys) of the two lines in BINS of each projection."""
        xs = [to_index(self._x_axis, b.value()) for b in self.x_boxes]
        ys = [to_index(self._y_axis, b.value()) for b in self.y_boxes]
        if self.refine_check.isChecked():
            xs = [refine_centroid(self._px, v) for v in xs]
            ys = [refine_centroid(self._py, v) for v in ys]
        return xs, ys

    def _recompute(self, *_):
        matching = self.match_radio.isChecked()
        for box in self.x_boxes + self.y_boxes:
            box.setEnabled(matching)
        self.refine_check.setEnabled(matching)

        if not matching:
            self.mapping = (0.0, 1.0)
            self.summary.setText("No matching: S = M + Mᵀ as it stands.")
        elif all(b.value() == 0 for b in self.x_boxes + self.y_boxes):
            self.mapping = None
            self.summary.setText(
                "Enter the approximate positions of two lines that appear on "
                "both projections -- for Co-60, 1173 and 1332 keV.")
        else:
            try:
                xs, ys = self._lines()
                a, b = map_from_lines(xs, ys)
            except SymmetrizeError as exc:
                self.mapping = None
                self.summary.setText(str(exc))
            else:
                self.mapping = (a, b)
                # Shown in channels, as typed; the mapping applied is the
                # same line expressed in bins.
                xc = [float(to_channel(self._x_axis, v)) for v in xs]
                yc = [float(to_channel(self._y_axis, v)) for v in ys]
                a_ch, b_ch = map_from_lines(xc, yc)
                self.summary.setText(
                    f"Line 1: Y {yc[0]:.2f} → X {xc[0]:.2f}     "
                    f"Line 2: Y {yc[1]:.2f} → X {xc[1]:.2f}\n"
                    f"Y is put on X's scale by  x = {a_ch:.3f} + {b_ch:.5f} · y")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.mapping is not None)
