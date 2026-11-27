"""Step 1 of Sort List-Mode gamma-gamma Data: choose and describe the input.

Everything the later steps rest on is decided here: which file, which
events are thrown away by their Flags value, which detector is X and
which is Y, and how far apart in time two events may be and still form
a pair. The dialog shows what the file actually holds (stream, counts
per channel, timestamp tick, time order, the run's info.txt) before any
of that is chosen, because each choice is only sensible against those
facts -- a pairing range that is not a whole number of timestamp ticks,
for instance, leaves the outermost time-difference bin half empty.

Two kinds of file are called out instead of silently accepted:

* FILTERED: CoMPASS's software coincidence has already removed the pairs
  a random window needs. Warned about, not refused, because the user may
  want to look at it anyway.
* Not time-sorted: the pairing refuses such input, so Next is refused
  here, where the reason can be explained, rather than after a sort.
"""

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem,
)

import listmode_io
from listmode_session import StepDialog, run_with_progress

FILE_FILTER = "CoMPASS list-mode data (*.root);;All files (*)"

#: Largest pairing range offered, in ns. The pair count grows linearly
#: with the range, and 100 us is already far beyond any prompt peak.
MAX_DT_LIMIT_NS = 100_000.0

FILTERED_WARNING = (
    "This is CoMPASS's FILTERED stream. CoMPASS's software coincidence has "
    "already removed essentially all pairs away from the prompt peak, so a "
    "random window has nothing to measure and the random subtraction "
    "cannot work. Use the UNFILTERED file (SData_<run>.root in the "
    "UNFILTERED folder) instead.")

UNSORTED_ERROR = (
    "The events in this file are not in time order. Pairing needs "
    "time-sorted events (CoMPASS's time sorting of the list output), so "
    "this file cannot be used as it is.")

NO_TIMESTAMPS_ERROR = (
    "Every timestamp in this file is zero, so no time differences can be "
    "formed.")

# Flags table columns.
COL_EXCLUDE, COL_VALUE, COL_COUNT, COL_MEANING = range(4)


def snap_max_dt_ps(value_ps, granularity_ps):
    """`value_ps` rounded to the nearest whole number of timestamp ticks,
    and at least one tick.

    The time-difference spectrum is binned at the tick, so a range that
    ends part-way through a tick makes the outermost bin only partly
    populated, and a random window reaching it under-measures (v7 plan,
    carried over from Task 2's review). With no known tick the value is
    only rounded to a whole picosecond.
    """
    value_ps = int(round(float(value_ps)))
    if not granularity_ps:
        return max(1, value_ps)
    g = int(granularity_ps)
    return max(g, int(round(value_ps / g)) * g)


def format_ps(ps):
    """A time in ps as ns (or us for whole microseconds), exact."""
    ps = int(ps)
    if ps % 1_000_000 == 0 and ps >= 1_000_000:
        return f"{ps // 1_000_000:,} µs"
    if ps % 1000 == 0:
        return f"{ps // 1000:,} ns"
    return f"{ps / 1000:,.3f} ns"


def _format_seconds(s):
    return "—" if s is None else f"{s:,.3f} s"


class ListModeInputDialog(StepDialog):
    """Step 1: file, flag exclusion, detector assignment, pairing range.

    The file being looked at is held on the dialog (``self.listmode``)
    and reaches the session only through ``apply_to_session``, so a file
    opened and then abandoned with Cancel leaves the session untouched.
    """

    def __init__(self, session, parent=None, initial_path=None):
        super().__init__(session, f"Sort List-Mode γγ Data — "
                         f"1 of 5: Input", parent, first=True)
        self._theme = getattr(parent, "_theme", "light")
        self._initial_path = initial_path
        self.listmode = None
        self.run_info = None
        self.path = None
        self.setMinimumWidth(640)

        # File row.
        row = QHBoxLayout()
        row.addWidget(QLabel("List-mode file:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("SData_<run>.root")
        row.addWidget(self.path_edit, 1)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.browse_button)
        self.body.addLayout(row)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        colour = "#ff6b6b" if self._theme == "dark" else "#b00020"
        for label in (self.error_label, self.warning_label):
            label.setStyleSheet(f"color: {colour}; font-weight: bold;")
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            label.hide()
        self.body.addWidget(self.error_label)
        self.body.addWidget(self.warning_label)

        # What the file holds.
        info_box = QGroupBox("File")
        form = QFormLayout(info_box)
        self.info_labels = {}
        for key, text in (("tree", "Tree / stream:"),
                          ("entries", "Entries:"),
                          ("channels", "Events per channel:"),
                          ("granularity", "Timestamp granularity:"),
                          ("sorted", "Time order:"),
                          ("run", "Run:"),
                          ("start", "Start:"),
                          ("stop", "Stop:"),
                          ("live", "Live time:")):
            label = QLabel("—")
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(text, label)
            self.info_labels[key] = label
        self.body.addWidget(info_box)

        # Flags.
        flags_box = QGroupBox("Flags (tick a value to exclude its events)")
        flags_layout = QHBoxLayout(flags_box)
        self.flags_table = QTableWidget(0, 4)
        self.flags_table.setHorizontalHeaderLabels(
            ["Exclude", "Flags", "Events", "Meaning"])
        self.flags_table.verticalHeader().setVisible(False)
        self.flags_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self.flags_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.flags_table.horizontalHeader()
        for col in (COL_EXCLUDE, COL_VALUE, COL_COUNT):
            header.setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_MEANING, QHeaderView.ResizeMode.Stretch)
        self.flags_table.setMinimumHeight(110)
        flags_layout.addWidget(self.flags_table)
        self.body.addWidget(flags_box)

        # Detectors and pairing range.
        pair_box = QGroupBox("Pairing")
        pair_form = QFormLayout(pair_box)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.x_combo.currentIndexChanged.connect(self._update_channel_check)
        self.y_combo.currentIndexChanged.connect(self._update_channel_check)
        pair_form.addRow("X detector (matrix x axis):", self.x_combo)
        pair_form.addRow("Y detector (matrix y axis, gain reference):",
                         self.y_combo)
        self.channel_label = QLabel()
        self.channel_label.setStyleSheet(f"color: {colour};")
        self.channel_label.hide()
        pair_form.addRow("", self.channel_label)
        dt_row = QHBoxLayout()
        self.max_dt_spin = QDoubleSpinBox()
        self.max_dt_spin.setDecimals(3)
        self.max_dt_spin.setRange(0.001, MAX_DT_LIMIT_NS)
        self.max_dt_spin.setSuffix(" ns")
        self.max_dt_spin.setValue(session.max_dt_ps / 1000.0)
        self.max_dt_spin.editingFinished.connect(self._snap_spin)
        dt_row.addWidget(self.max_dt_spin)
        self.dt_note = QLabel()
        dt_row.addWidget(self.dt_note, 1)
        pair_form.addRow("Maximum |Δt| = |t(X) − t(Y)|:", dt_row)
        self.body.addWidget(pair_box)

        self._show_file()

    # ------------------------------------------------------------ session
    def load_from_session(self):
        s = self.session
        self.path = s.path
        self.listmode = s.listmode
        self.run_info = s.run_info
        self.max_dt_spin.setValue(s.max_dt_ps / 1000.0)
        self._show_file(exclude=s.exclude_flags if s.listmode else None)
        if s.listmode is not None:
            self._select_channel(self.x_combo, s.x_channel)
            self._select_channel(self.y_combo, s.y_channel)
        self._snap_spin()

    def apply_to_session(self):
        if self.listmode is None:
            return "Choose a list-mode file (SData_<run>.root) first."
        problem = self._file_problem()
        if problem:
            return problem
        x, y = self.x_combo.currentData(), self.y_combo.currentData()
        if x is None or y is None:
            return ("The file needs events on two channels to form pairs; "
                    "it has events on "
                    f"{len(self.listmode.events_per_channel)}.")
        if x == y:
            return "The X and Y detectors must be different channels."
        s = self.session
        s.path = str(self.path)
        s.listmode = self.listmode
        s.run_info = self.run_info
        s.exclude_flags = self.excluded_flag_values()
        s.x_channel = int(x)
        s.y_channel = int(y)
        s.max_dt_ps = self.max_dt_ps()
        return None

    # ------------------------------------------------------------- values
    def excluded_flag_values(self):
        """The Flags VALUES ticked for exclusion, as a set of ints."""
        out = set()
        for row in range(self.flags_table.rowCount()):
            item = self.flags_table.item(row, COL_EXCLUDE)
            if item.checkState() == Qt.CheckState.Checked:
                out.add(int(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def set_flag_excluded(self, value, excluded):
        for row in range(self.flags_table.rowCount()):
            item = self.flags_table.item(row, COL_EXCLUDE)
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(value):
                item.setCheckState(Qt.CheckState.Checked if excluded
                                   else Qt.CheckState.Unchecked)
                return
        raise KeyError(f"No flags value 0x{int(value):X} in the table")

    def granularity_ps(self):
        return None if self.listmode is None else self.listmode.granularity_ps

    def max_dt_ps(self):
        """The spin box's value in ps, snapped to the timestamp tick."""
        return snap_max_dt_ps(self.max_dt_spin.value() * 1000.0,
                              self.granularity_ps())

    # --------------------------------------------------------------- file
    def _start_folder(self):
        for candidate in (self.path, self._initial_path):
            if candidate:
                p = Path(candidate)
                return str(p if p.is_dir() else p.parent)
        try:
            from settings import Settings
            return Settings().last_folder()
        except Exception:
            return ""

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open List-Mode Data", self._start_folder(), FILE_FILTER)
        if not path:
            return
        try:
            from settings import Settings
            Settings().set_last_folder(os.path.dirname(path))
        except Exception:
            pass
        self.open_path(path)

    def open_path(self, path):
        """Read and describe `path`. A file that cannot be read is
        reported in the dialog, never raised; the previous file is then
        forgotten so Next cannot go ahead with something not shown."""

        def work(progress):
            return (listmode_io.open_listmode(path),
                    listmode_io.read_run_info(path))

        result, error = run_with_progress(
            self, "Reading list-mode file",
            f"Reading {os.path.basename(str(path))}…", work)
        self.path = str(path)
        if error is not None:
            self.listmode = None
            self.run_info = None
            self._show_file(error=error)
            return False
        self.listmode, self.run_info = result
        self._show_file()
        self._snap_spin()
        return True

    def _file_problem(self):
        lm = self.listmode
        if lm is None:
            return None
        if not lm.is_sorted:
            return UNSORTED_ERROR
        if not lm.granularity_ps:
            return NO_TIMESTAMPS_ERROR
        return None

    # ------------------------------------------------------------ display
    def _show_file(self, error=None, exclude=None):
        """Fill every field from self.listmode / self.run_info. `exclude`
        is the set of flag values to tick; None means the defaults."""
        lm = self.listmode
        self.path_edit.setText("" if self.path is None else str(self.path))
        self.error_label.setVisible(bool(error))
        self.error_label.setText(error or "")

        warnings = []
        if lm is not None and lm.stream == "FILTERED":
            warnings.append(FILTERED_WARNING)
        problem = self._file_problem()
        if problem:
            warnings.append(problem)
        self.warning_label.setText("\n\n".join(warnings))
        self.warning_label.setVisible(bool(warnings))

        info = self.info_labels
        if lm is None:
            for label in info.values():
                label.setText("—")
        else:
            info["tree"].setText(f"{lm.tree_name} ({lm.stream})")
            info["entries"].setText(f"{lm.n_entries:,}")
            info["channels"].setText(", ".join(
                f"CH{c}: {n:,}" for c, n in sorted(
                    lm.events_per_channel.items())) or "none")
            info["granularity"].setText(
                "unknown (all timestamps are zero)" if not lm.granularity_ps
                else f"{format_ps(lm.granularity_ps)} "
                     f"({lm.granularity_ps:,} ps)")
            info["sorted"].setText("time-sorted" if lm.is_sorted
                                   else "NOT time-sorted")
            ri = self.run_info
            if ri is None:
                for key in ("run", "start", "stop", "live"):
                    info[key].setText("no <run>_info.txt found beside the "
                                      "stream folders" if key == "run"
                                      else "—")
            else:
                info["run"].setText(ri.run_id or "—")
                info["start"].setText(ri.start or "—")
                info["stop"].setText(ri.stop or "—")
                info["live"].setText(", ".join(
                    f"CH{c}: {_format_seconds(ch.live_time_s)} live, "
                    f"{_format_seconds(ch.real_time_s)} real"
                    for c, ch in sorted(ri.channels.items())) or "—")

        self._fill_flags(exclude)
        # Next stays enabled: apply_to_session says why it refuses,
        # which a greyed-out button would not.
        self._fill_channels()

    def _fill_flags(self, exclude):
        lm = self.listmode
        counts = {} if lm is None else lm.flag_counts
        if exclude is None:
            exclude = listmode_io.default_excluded_flags(counts)
        exclude = {int(v) for v in exclude}
        table = self.flags_table
        table.setRowCount(0)
        for value, count in sorted(counts.items()):
            row = table.rowCount()
            table.insertRow(row)
            box = QTableWidgetItem()
            box.setFlags(Qt.ItemFlag.ItemIsUserCheckable
                         | Qt.ItemFlag.ItemIsEnabled)
            box.setCheckState(Qt.CheckState.Checked if int(value) in exclude
                              else Qt.CheckState.Unchecked)
            box.setData(Qt.ItemDataRole.UserRole, int(value))
            table.setItem(row, COL_EXCLUDE, box)
            for col, text in ((COL_VALUE, f"0x{int(value):X}"),
                              (COL_COUNT, f"{int(count):,}"),
                              (COL_MEANING, "; ".join(
                                  listmode_io.describe_flags(value))
                               or "no flag set")):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if col == COL_COUNT:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, col, item)

    def _fill_channels(self):
        lm = self.listmode
        channels = [] if lm is None else sorted(lm.events_per_channel)
        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(True)
            combo.clear()
            for c in channels:
                combo.addItem(f"CH{c} ({lm.events_per_channel[c]:,} events)",
                              int(c))
            combo.blockSignals(False)
        # Defaults: X = CH1, Y = CH0 (CoMPASS's orientation), else the
        # first two channels present.
        if len(channels) >= 2:
            x = 1 if 1 in channels else channels[1]
            y = 0 if 0 in channels and x != 0 else next(
                c for c in channels if c != x)
            self._select_channel(self.x_combo, x)
            self._select_channel(self.y_combo, y)
        self._update_channel_check()

    @staticmethod
    def _select_channel(combo, channel):
        i = combo.findData(int(channel))
        if i >= 0:
            combo.setCurrentIndex(i)

    def _update_channel_check(self, *_):
        x, y = self.x_combo.currentData(), self.y_combo.currentData()
        text = ""
        if self.listmode is not None:
            if x is None or y is None or len(
                    self.listmode.events_per_channel) < 2:
                text = "Pairing needs events on two channels."
            elif x == y:
                text = "X and Y must be different detectors."
        self.channel_label.setText(text)
        self.channel_label.setVisible(bool(text))

    def _snap_spin(self):
        g = self.granularity_ps()
        snapped = self.max_dt_ps()
        if g:
            self.max_dt_spin.setSingleStep(g / 1000.0)
            self.max_dt_spin.setMinimum(g / 1000.0)
            self.dt_note.setText(
                f"= ±{format_ps(snapped)}, a whole number of "
                f"{format_ps(g)} timestamp ticks")
        else:
            self.dt_note.setText("")
        self.max_dt_spin.setValue(snapped / 1000.0)
