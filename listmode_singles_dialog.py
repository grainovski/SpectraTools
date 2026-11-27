"""Step 2 of Sort List-Mode gamma-gamma Data: the singles spectra.

Nothing is chosen here. The step exists so the user sees, before fitting
peaks for gain matching, that the file was read the way they meant: both
detectors' singles at full ADC resolution, what the flag exclusion threw
away, and how many pairs the pairing range found. A wrong detector
assignment or an unexpected flag selection is obvious in these numbers
and costs nothing to fix with Back now.

"Send to main window" hands the two spectra over as ordinary spectra, so
they can be fitted, calibrated or saved with the existing tools; it does
not end the operation.
"""

import re
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
)

import theme
from listmode_input_dialog import format_ps
from listmode_session import StepDialog

#: Line colours by channel. Fixed per channel rather than per position so
#: CH0 is the same colour here and in the later steps' plots.
CHANNEL_COLORS = {0: "#1f77b4", 1: "#ff7f0e"}


def run_label(session):
    """A short name for the run: the info.txt's Run ID, else the run ID
    in CoMPASS's SData file name, else the file's stem."""
    if session.run_info is not None and session.run_info.run_id:
        return session.run_info.run_id
    if session.path:
        name = Path(session.path).name
        m = re.match(r"^SData[RF]?_(.+)\.root$", name)
        return m.group(1) if m else Path(name).stem
    return "list-mode"


def singles_names(session):
    """[(name, array)] of the session's singles, by channel number."""
    run = run_label(session)
    return [(f"{run} CH{c} singles", np.asarray(session.singles[c]))
            for c in sorted(session.singles)]


class ListModeSinglesDialog(StepDialog):
    def __init__(self, session, parent=None):
        super().__init__(session, f"Sort List-Mode γγ Data — "
                         f"2 of 5: Singles", parent)
        self._theme = getattr(parent, "_theme", "light")
        self.resize(760, 560)

        self.figure = Figure(figsize=(7.0, 4.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.body.addWidget(self.canvas, 1)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.body.addWidget(self.summary_label)

        row = QHBoxLayout()
        self.log_check = QCheckBox("Logarithmic counts")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(self._plot)
        row.addWidget(self.log_check)
        row.addStretch(1)
        self.sent_label = QLabel()
        row.addWidget(self.sent_label)
        self.send_button = QPushButton("Send to main window")
        self.send_button.setToolTip(
            "Add both singles spectra to the main window as ordinary "
            "spectra (the operation stays open).")
        self.send_button.clicked.connect(self._send)
        row.addWidget(self.send_button)
        self.body.addLayout(row)

    # ------------------------------------------------------------ session
    def load_from_session(self):
        s = self.session
        self.send_button.setEnabled(s.send_spectra is not None
                                    and s.singles is not None)
        self.sent_label.setText("")
        self._plot()
        self._summarise()

    def apply_to_session(self):
        return None  # nothing to choose

    # ------------------------------------------------------------ display
    def _plot(self, *_):
        ax = self.axes
        ax.clear()
        singles = self.session.singles
        nchan = int(self.session.nchan)
        if singles:
            x = np.arange(nchan)
            for c in sorted(singles):
                role = (" (X)" if c == self.session.x_channel else
                        " (Y)" if c == self.session.y_channel else "")
                ax.plot(x, np.asarray(singles[c]), drawstyle="steps-mid",
                        linewidth=0.7, color=CHANNEL_COLORS.get(c),
                        label=f"CH{c}{role}")
            ax.legend(loc="upper right")
        ax.set_xlim(0, nchan - 1)
        ax.set_xlabel(f"Energy [ADC channel, 0–{nchan - 1}]")
        ax.set_ylabel("Counts")
        if self.log_check.isChecked():
            ax.set_yscale("log", nonpositive="clip")
            ax.set_ylim(bottom=0.5)
        else:
            ax.set_yscale("linear")
            ax.set_ylim(bottom=0)
        theme.style_axes(ax, self._theme)
        legend = ax.get_legend()
        if legend is not None:
            fg = ax.xaxis.label.get_color()
            for text in legend.get_texts():
                text.set_color(fg)
            legend.get_frame().set_facecolor(ax.get_facecolor())
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _summarise(self):
        s = self.session
        singles = s.singles
        if singles is None:
            self.summary_label.setText("The singles have not been sorted.")
            return
        lines = []
        for c in sorted(singles):
            over = singles.overflow.get(c, 0)
            lines.append(f"CH{c}: {int(np.sum(singles[c])):,} counts"
                         + (f", {over:,} outside 0–{s.nchan - 1} "
                            f"(not shown)" if over else
                            f", none outside 0–{s.nchan - 1}"))
        lines.append(f"Events excluded by their Flags value: "
                     f"{singles.excluded:,}")
        if s.pairs is not None:
            lines.append(
                f"Pairs (every CH{s.x_channel} × CH{s.y_channel} "
                f"combination) within |Δt| ≤ "
                f"{format_ps(s.max_dt_ps)}: {len(s.pairs.dt):,}")
        self.summary_label.setText("\n".join(lines))

    # ------------------------------------------------------------- action
    def _send(self):
        s = self.session
        if s.send_spectra is None or s.singles is None:
            return
        spectra = singles_names(s)
        try:
            s.send_spectra(spectra)
        except Exception as exc:  # a slot must never swallow it
            QMessageBox.warning(self, self.windowTitle(),
                                f"Could not add the spectra:\n\n"
                                f"{type(exc).__name__}: {exc}")
            return
        self.sent_label.setText(
            "Sent: " + ", ".join(name for name, _ in spectra))
