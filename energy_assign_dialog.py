"""Assign known energies to fitted peaks, then calibrate from them.

The calibration this produces is anchored on FITTED peak centroids, so it
inherits the fits' precision rather than the user's aim with a cursor --
which is the difference between a calibration that is a physics
measurement and one that is a clerical exercise.

Modelled on HDTV's "fit position assign" followed by "calibration
position recalibrate", which is the same loop: attach literature energies
to peaks you have already fitted, refit the calibration from every
assignment, and repeat as peaks are added or corrected.

Energies can be typed, or taken from a `.sou` source file (see sou_io):
load the nuclide, anchor two peaks by hand, and "Suggest remaining" fills
every other row whose nearest source line is unambiguous. Suggestions
are ordinary editable cells, tinted and tooltipped so they read as
guesses until the user has looked at them -- OK uses exactly what the
table shows, never a hidden list.
"""

import math
import os
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import calibration as calibration_module
from calibration_plot_dialog import CalibrationPlotDialog
from calibration import CalibrationError
from energy_assignments import restore
from histogram_io import ParseError
from sou_io import load_sou, match_line

#: Background for a suggested energy cell. Translucent amber: it blends
#: over whatever the table's own base colour is, so it reads on the dark
#: theme as well as the light one -- an opaque pastel would vanish into
#: a light base and glare on a dark one.
_SUGGESTED_TINT = QColor(255, 170, 0, 70)


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


def _parse_energy(text):
    """The finite float `text` holds, or None. Lenient on purpose: this
    decides whether a row COUNTS as an anchor, and a half-typed value
    should simply not count rather than raise."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class _PeakRow:
    """One dialog row's peak. A record rather than a widening tuple:
    callers pass anything from (label, channel) to the full six fields,
    and every field beyond the first two is optional, so positional
    unpacking at each use site would be a standing bug."""
    label: str
    channel: float
    channel_err: float = None
    fwhm: float = None
    area: float = 0.0
    area_err: float = 0.0


class EnergyAssignDialog(QDialog):
    """`result_calibration` is set only after a successful OK.

    `peaks` is a list of (label, channel) for every committed peak on the
    active spectrum, in the order the Fit Results panel shows them.

    `settings`, when given, is the app's Settings: the source picker
    starts in and remembers the same last folder as the spectrum dialogs.
    """

    # Shifted right by one when the centroid-uncertainty column was added.
    # Every caller reads the constant rather than the literal, which is why
    # that column could be inserted at all.
    ENERGY_COLUMN = 3

    def __init__(self, parent, peaks, quadratic=False, settings=None,
                 assignments=None, max_channel=None, export_default_path=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate from Fitted Peaks")
        self.result_calibration = None
        #: The live plot needs the channel range to draw the curve over
        #: and a filename to offer its export under. Both are properties
        #: of the spectrum, which this dialog otherwise knows nothing
        #: about; when they are absent no live plot is shown, which is
        #: what keeps every existing caller and test unaffected.
        self._max_channel = max_channel
        self._export_default_path = export_default_path
        self._live_plot = None
        self._settings = settings
        # Accepts (label, channel) through the full six fields. Older
        # callers and every existing test pass the short form.
        self._peaks = [_PeakRow(*entry[:6]) for entry in peaks]
        #: Lines of the loaded .sou file, or None before one is loaded.
        self.source_lines = None
        self._source_name = None
        self._source_path = None
        #: row -> the text this dialog put there. A row stays a
        #: suggestion only while its cell still reads exactly that; the
        #: moment the user edits it, it is theirs (see _on_item_changed).
        self._suggested = {}
        #: True while this dialog itself is writing cells, so the
        #: itemChanged handler does not mistake its own writes for edits.
        self._writing = False
        #: Set by Clear so the caller knows to erase the spectrum's
        #: stored record, not merely to skip writing a new one.
        self.cleared = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Type the known energy for the peaks you can identify and leave "
            "the rest blank -- or load a source file, type two energies, and "
            "let it suggest the rest."
        ))

        source_row = QHBoxLayout()
        self.load_source_button = QPushButton("Load source...")
        self.load_source_button.clicked.connect(self._on_load_source_clicked)
        source_row.addWidget(self.load_source_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip(
            "Forget every assigned energy and the loaded source for this "
            "spectrum. Does not change the active calibration."
        )
        self.clear_button.clicked.connect(self._on_clear)
        source_row.addWidget(self.clear_button)
        self.source_label = QLabel("No source loaded")
        source_row.addWidget(self.source_label)
        source_row.addStretch(1)
        self.suggest_button = QPushButton("Suggest remaining")
        self.suggest_button.setEnabled(False)
        self.suggest_button.setToolTip(
            "Needs a loaded source and at least two typed energies"
        )
        self.suggest_button.clicked.connect(self._on_suggest)
        source_row.addWidget(self.suggest_button)
        layout.addLayout(source_row)

        self.table = QTableWidget(len(self._peaks), 4)
        self.table.setHorizontalHeaderLabels(
            ["Peak", "Channel", "± ch", "Energy (keV)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for row, peak in enumerate(self._peaks):
            name = QTableWidgetItem(peak.label)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, name)

            position = QTableWidgetItem(f"{peak.channel:.3f}")
            position.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, position)

            # Shown, not just used: it is the reason one row deserves more
            # weight than another, and seeing a peak with a large
            # uncertainty explains a calibration that leans away from it.
            uncertainty = QTableWidgetItem(_format_uncertainty(peak.channel_err))
            uncertainty.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 2, uncertainty)

            self.table.setItem(row, self.ENERGY_COLUMN, QTableWidgetItem(""))
        self.table.itemChanged.connect(self._on_item_changed)
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

        # Deferred to the end of construction: restoring calls load_source(),
        # which reports through self.status -- a widget that does not exist
        # until every row above it has run.
        if assignments is not None:
            self._restore_assignments(assignments)

    def _restore_assignments(self, assignments):
        """Refill the table from a previous visit.

        Matching is by channel within each peak's own FWHM, so a refit
        that nudged a centroid keeps its energy while a peak that moved
        further than its own width opens blank rather than inheriting
        an identification that belonged to something else.
        """
        peaks = [(peak.channel, peak.fwhm) for peak in self._peaks]
        self._writing = True
        try:
            for row, energy in restore(assignments, peaks).items():
                self.table.item(row, self.ENERGY_COLUMN).setText(repr(energy))
        finally:
            self._writing = False
        if assignments.source_path and os.path.exists(assignments.source_path):
            self.load_source(assignments.source_path)
        self._assignments_changed()

    def _on_clear(self):
        """Empty every energy, drop the source, and mark the stored
        record for deletion. Deliberately does NOT touch the active
        calibration: discarding the identifications and un-calibrating
        the spectrum are different actions."""
        self._writing = True
        try:
            for row in range(len(self._peaks)):
                item = self.table.item(row, self.ENERGY_COLUMN)
                item.setText("")
                item.setBackground(QBrush())
                item.setToolTip("")
        finally:
            self._writing = False
        self._suggested.clear()
        self.source_lines = None
        self._source_name = None
        # Without this, the discarded file survives in _source_path: OK
        # after a Clear then hand-typed energies still stores that path,
        # and reopening the dialog silently reloads the rejected source.
        self._source_path = None
        self.source_label.setText("No source loaded")
        self.cleared = True
        self.status.setText("Cleared all assignments.")
        self._assignments_changed()

    # --- reading the table ------------------------------------------------

    def _energy_text(self, row):
        item = self.table.item(row, self.ENERGY_COLUMN)
        return item.text() if item else ""

    def assignments(self):
        """(channels, energies, channel_errors) for the rows that carry an
        energy.

        Raises ValueError naming the row for anything unparseable, rather
        than skipping it: a typo silently dropped would produce a
        calibration quietly fitted to fewer points than the user thinks.
        """
        channels, energies, errors = [], [], []
        for row, peak in enumerate(self._peaks):
            text = self._energy_text(row).strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{peak.label}: {text!r} is not a number") from None
            # float() alone accepts "nan"/"inf", which from_points would
            # only reject later with a message about the whole fit's
            # coefficients -- name the offending row here instead, exactly
            # like an unparseable entry.
            if not math.isfinite(value):
                raise ValueError(f"{peak.label}: {text!r} is not a finite energy") from None
            energies.append(value)
            channels.append(peak.channel)
            errors.append(peak.channel_err)
        return channels, energies, errors

    def source_path(self):
        """The loaded .sou path, or None. Stored so reopening the dialog
        can load the same source again."""
        return getattr(self, "_source_path", None)

    def export_points(self):
        """(channel, channel_err, area, area_err, energy) per assigned
        row, for the plot window and the CalEnEff export."""
        points = []
        for row, peak in enumerate(self._peaks):
            text = self._energy_text(row).strip()
            if not text:
                continue
            points.append((
                peak.channel, peak.channel_err or 0.0,
                peak.area, peak.area_err, float(text),
            ))
        return points

    def suggested_rows(self):
        """Rows currently holding an untouched suggestion."""
        return set(self._suggested)

    def _anchor_count(self):
        return sum(
            1 for row in range(len(self._peaks))
            if _parse_energy(self._energy_text(row)) is not None
        )

    # --- source loading ---------------------------------------------------

    def _on_load_source_clicked(self):
        directory = self._settings.last_folder() if self._settings else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Source", directory, "Source files (*.sou);;All files (*)"
        )
        if path:
            self.load_source(path)

    def load_source(self, path):
        """Load a .sou file as the source to suggest from. Returns True on
        success; on failure the message goes to the status line and any
        previously loaded source is kept."""
        try:
            lines = load_sou(path)
        except ParseError as exc:
            self.status.setText(str(exc))
            return False
        # Guesses made from the PREVIOUS nuclide must not survive into a
        # new one. Loading a different source is exactly what a user does
        # after realising they had the wrong nuclide, and leaving those
        # rows filled would let OK calibrate against the source they just
        # rejected -- with a tooltip still naming the old file. Only
        # untouched suggestions go; anything typed or edited is the user's.
        replaced = self._source_name is not None and self.source_lines is not None
        dropped = len(self._suggested) if replaced else 0
        if replaced:
            self._clear_suggestions()

        self.source_lines = lines
        self._source_name = os.path.basename(path)
        self._source_path = path
        self.source_label.setText(f"{self._source_name}: {len(lines)} lines")
        if self._settings is not None:
            self._settings.set_last_folder(os.path.dirname(path))
        self.status.setText(
            f"Cleared {dropped} suggestion(s) from the previous source."
            if dropped else ""
        )
        self._assignments_changed()
        return True

    # --- suggestions ------------------------------------------------------

    def _update_suggest_availability(self):
        self.suggest_button.setEnabled(
            self.source_lines is not None and self._anchor_count() >= 2
        )

    def _on_item_changed(self, item):
        if self._writing or item.column() != self.ENERGY_COLUMN:
            return
        row = item.row()
        # An edited suggestion is the user's own value now: it loses the
        # tint so nothing in the table claims to be a guess when it is not,
        # and it counts as an anchor on the next Suggest.
        if row in self._suggested and item.text() != self._suggested[row]:
            self._unmark(row)
        self._assignments_changed()

    def _mark(self, row, text):
        item = self.table.item(row, self.ENERGY_COLUMN)
        self._suggested[row] = text
        self._writing = True
        try:
            item.setText(text)
            item.setBackground(QBrush(_SUGGESTED_TINT))
            item.setToolTip(
                f"Suggested from {self._source_name} -- check before OK"
            )
        finally:
            self._writing = False

    def _unmark(self, row):
        self._suggested.pop(row, None)
        item = self.table.item(row, self.ENERGY_COLUMN)
        self._writing = True
        try:
            item.setBackground(QBrush())
            item.setToolTip("")
        finally:
            self._writing = False

    def _clear_suggestions(self):
        """Blank every suggestion the user has not touched, so a fresh
        round is computed from their values alone and never from the
        previous round's guesses."""
        for row, text in list(self._suggested.items()):
            item = self.table.item(row, self.ENERGY_COLUMN)
            self._unmark(row)
            if item.text() == text:
                self._writing = True
                try:
                    item.setText("")
                finally:
                    self._writing = False

    def _on_suggest(self):
        if self.source_lines is None:
            self.status.setText("Load a source file first.")
            return
        self._clear_suggestions()
        try:
            channels, energies, errors = self.assignments()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if len(channels) < 2:
            self.status.setText(
                "Type the energies of at least two peaks first; they anchor "
                "the line the suggestions are read from."
            )
            return
        # A straight line through the anchors is all that is needed to
        # predict where the other peaks fall. Always linear here, whatever
        # the checkbox says: two anchors cannot define a quadratic, and the
        # final fit is a separate step with its own choice.
        try:
            provisional = calibration_module.from_points(
                channels, energies, False, channel_errors=errors
            )
        except CalibrationError as exc:
            self.status.setText(str(exc))
            return

        # Energies the anchors already claim. A source line matched to a
        # blank row must not be one of these -- two peaks cannot be the
        # same line, and if the nearest line is taken, the honest answer is
        # no suggestion, not the runner-up.
        taken = {
            line.energy for line in self.source_lines
            if any(math.isclose(line.energy, e, rel_tol=1e-9, abs_tol=1e-9)
                   for e in energies)
        }
        proposals = {}
        for row, peak in enumerate(self._peaks):
            if self._energy_text(row).strip():
                continue
            line = match_line(provisional.apply(peak.channel), self.source_lines)
            if line is None or line.energy in taken:
                continue
            proposals[row] = line.energy

        # Two blank rows both nearest to one line is a conflict, not a
        # tie to break: neither gets it.
        wanted = {}
        for energy in proposals.values():
            wanted[energy] = wanted.get(energy, 0) + 1
        accepted = {
            row: energy for row, energy in proposals.items() if wanted[energy] == 1
        }
        for row, energy in accepted.items():
            self._mark(row, repr(energy))
        self._assignments_changed()

        # Counted BEFORE marking would be simpler, but the marks are what
        # make a row non-blank, so the two are added back together here.
        still_blank = sum(
            1 for row in range(len(self._peaks)) if not self._energy_text(row).strip()
        )
        candidates = still_blank + len(accepted)
        if not candidates:
            self.status.setText("Every peak already has an energy.")
        elif not accepted:
            self.status.setText(
                f"No unambiguous match in {self._source_name} for any of the "
                f"{candidates} unassigned peaks."
            )
        elif still_blank:
            self.status.setText(
                f"{len(accepted)} of {candidates} unassigned peaks suggested "
                f"from {self._source_name} -- review before OK. The other "
                f"{still_blank} had no unambiguous match."
            )
        else:
            self.status.setText(
                f"All {len(accepted)} unassigned peaks suggested from "
                f"{self._source_name} -- review before OK."
            )

    # --- accepting --------------------------------------------------------

    def _compute_calibration(self):
        """(calibration, None) or (None, reason).

        Shared by OK and by the live preview so the plot on screen can
        never be the result of a different fit from the one OK applies.
        """
        try:
            channels, energies, errors = self.assignments()
        except ValueError as exc:
            return None, str(exc)
        try:
            return calibration_module.from_points(
                channels, energies, self.quadratic_checkbox.isChecked(),
                # from_points falls back to an unweighted fit if any of
                # these is missing or unusable, so passing them
                # unconditionally is safe.
                channel_errors=errors,
            ), None
        except CalibrationError as exc:
            return None, str(exc)

    def _assignments_changed(self):
        """Called by every path that can change the assignment set."""
        self._update_suggest_availability()
        self._refresh_live_plot()

    def _close_live_plot(self):
        previous = self._live_plot
        self._live_plot = None
        if previous is None:
            return
        try:
            previous.close()
            previous.deleteLater()
        except RuntimeError:
            pass  # Qt destroyed it already; nothing left to close

    def _refresh_live_plot(self):
        """Show the calibration as it currently stands.

        Parented to this dialog, so this dialog's own modality does not
        block it and it is destroyed along with it -- the window
        main_window opens on OK is a separate, longer-lived one.

        An assignment set that does not yet make a calibration (too few
        points, or a typo mid-edit) closes the preview rather than
        leaving a stale curve on screen claiming to describe the table.
        """
        if self._max_channel is None:
            return
        calibration, _reason = self._compute_calibration()
        if calibration is None:
            self._close_live_plot()
            return
        points = self.export_points()
        if self._live_plot is not None:
            try:
                self._live_plot.set_data(calibration, points, self.source_lines)
                return
            except RuntimeError:
                # Closed by the user; fall through and build a new one.
                self._live_plot = None
        self._live_plot = CalibrationPlotDialog(
            self, calibration, points, self.source_lines,
            self._max_channel, self._export_default_path,
        )
        self._live_plot.show()

    def _on_accept(self):
        calibration, reason = self._compute_calibration()
        if calibration is None:
            self.status.setText(reason)
            return
        self.result_calibration = calibration
        channels, energies, _errors = self.assignments()

        # Residuals are reported rather than only the coefficients: one
        # mistyped energy moves the whole fit and is obvious here while
        # being invisible in a and b.
        worst = max(
            abs(r) for r in calibration_module.residuals(
                self.result_calibration, channels, energies
            )
        ) if channels else 0.0
        self._worst_residual = worst
        # The preview belongs to this dialog; main_window opens its own
        # window once the calibration is actually applied. Closing here
        # rather than relying on the parent being hidden keeps exactly
        # one calibration plot on screen at every moment.
        self._close_live_plot()
        self.accept()
