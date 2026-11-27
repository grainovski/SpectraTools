"""Step 3 of Sort List-Mode gamma-gamma Data: gain matching CH1 onto CH0.

A UI layer over gain_match, which is Qt-free and tested against spectra
whose answer is known. gain_match.gain_match() would do the whole step
in one call; this dialog calls the four steps one by one instead
(find_peaks, pair_peaks, fit_gain, GainMatch.map) so that the user can
see and override each of them: which spectra the peaks come from, which
pairs are used, the polynomial order, and whether to gain match at all.

Three guards are required by Task 3's review of the real Na-22 run
(v7 plan, Task 5). There the automatic choice -- order 2 at four or more
pairs -- takes its curvature from a feature near CH0 channel 150 that is
probably the threshold edge, not a line both detectors see. Through the
three real lines (511, 1275 and 1461 keV) a straight line fits to a few
channels, while the quadratic differs from it by ~380 channels near ADC
16000, far beyond the last pair, where nothing constrains it. A gain
fitted that way passes through every point it was fitted to and looks
entirely reasonable, which is why each guard is a visible part of the
dialog rather than a check that fires once:

(a) the map is drawn over the whole ADC range with the part outside the
    pair-covered range dashed and shaded, and the range is stated;
(b) with order 2 the same pairs are also fitted at order 1, and a label
    warns when the two differ by more than one FWHM anywhere outside the
    pair range (see extrapolation_check);
(c) pairs whose CH0 centroid lies below an editable low-channel cutoff
    start unticked (see DEFAULT_LOW_CUTOFF).

The order is the user's choice; gain_match's automatic choice is only
the starting value.
"""

import dataclasses
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import gain_match
import peak_search
from bundled_sources import bundled_source_files, bundled_sources_dir
from histogram_io import ParseError
from listmode_session import StepDialog, run_with_progress
from sou_io import load_sou
from spectrum_files import SPECTRUM_FILTER, load_spectrum_file
from theme import style_axes

#: Pairs whose CH0 centroid is below this channel start unticked (guard
#: (c)). Below a few hundred channels of a 16k digitizer spectrum sit the
#: trigger threshold edge, noise and, for an HPGe, X-rays -- features
#: whose position depends on each detector's own threshold and noise, not
#: on the gain, so they are not "a line both detectors see" even when
#: the matcher pairs them. On the real Na-22 run the matcher pairs such a
#: feature near CH0 channel 150 while the lowest real line, 511 keV, is at
#: channel ~2170 (0.24 keV per channel): 300 channels (~70 keV there)
#: lies safely between the two. For an HPGe at 16k (0.1-0.2 keV per
#: channel) 300 channels is 30-60 keV, below the lowest lines normally
#: used (Eu-152's 121.8 keV, Ba-133's 81 keV at 0.2 keV per channel is
#: channel ~400). It only sets the starting ticks; the user can tick any
#: pair back or move the cutoff, which re-applies the default ticks.
DEFAULT_LOW_CUTOFF = 300

#: Order labels in the combo, index = order - 1.
ORDER_LABELS = ("1 (linear)", "2 (quadratic)")

#: What the user is told CH1 does when gain matching is skipped.
SKIP_EXPLANATION = (
    "Gain matching is skipped: CH1 is sorted on its own ADC scale (the "
    "identity map). The two matrix axes then have different gains, so "
    "the matrix must not be symmetrized as it is.")


# ------------------------------------------------------------- pure helpers


def _load_1d(path):
    """Counts of a 1-D spectrum file as floats, read by the same
    spectrum_files.load_spectrum_file as File > Open. An embedded energy
    calibration is ignored: gain matching works in ADC channels. Raises
    ParseError or OSError.
    """
    data, _calibration = load_spectrum_file(path)
    return np.asarray(data, dtype=float)


def _padded(counts, nchan):
    """`counts` as float, zero-padded up to `nchan` channels. Never cut:
    a longer spectrum keeps its top, so no peak is silently dropped."""
    counts = np.asarray(counts, dtype=float).ravel()
    if counts.size < nchan:
        counts = np.concatenate([counts, np.zeros(nchan - counts.size)])
    return counts


def pair_range(pairs):
    """(lowest, highest) CH1 centroid of the included pairs, or None.

    In CH1 channels, because that is the map's domain: the map is
    constrained between these two CH1 values and extrapolated outside.
    """
    xs = [p.ch1 for p in pairs if p.included]
    if not xs:
        return None
    return float(min(xs)), float(max(xs))


def _largest_fwhm(gain, pairs):
    """The largest FWHM of the included pairs, in CH0 channels, or None.

    CH0's own fitted width when known, otherwise CH1's width carried
    through the slope of the map (a width maps by P').
    """
    widths = []
    for p in pairs:
        if not p.included:
            continue
        if np.isfinite(p.fwhm0) and p.fwhm0 > 0:
            widths.append(float(p.fwhm0))
        elif np.isfinite(p.fwhm1) and p.fwhm1 > 0:
            widths.append(float(p.fwhm1 * gain.derivative(p.ch1)))
    return max(widths) if widths else None


@dataclasses.dataclass
class Extrapolation:
    """How far an order-2 map leaves the order-1 fit of the same pairs
    outside the pair-covered range. `deviation` (CH0 channels) is the
    largest |P2 - P1| there, at CH1 channel `channel`; `fwhm` is the
    threshold it was compared with (None when no width is known)."""

    deviation: float
    channel: int
    fwhm: float
    warn: bool


def extrapolation_check(gain, linear, pairs, nchan):
    """Guard (b): compare the order-2 `gain` with the order-1 `linear`
    outside the pair range, over CH1 channels 0 ... nchan-1.

    The threshold is ONE FWHM, and the FWHM used is the LARGEST of the
    included pairs. Extrapolating the width instead would need a
    resolution model the dialog does not know -- a scintillator's grows
    as sqrt(E), an HPGe's barely at all -- and a width extrapolated by
    the wrong model would make the warning depend on an assumption. The
    largest measured width is a measured number; it is generous below the
    range (where real widths are smaller) and, for a scintillator, a
    little strict above it, which is the side where Na-22's quadratic
    goes wrong: there the deviation is ~380 channels against a largest
    width of ~110.

    Returns an Extrapolation, or None when there is nothing to compare
    (an order-1 map, no linear fit, or pairs covering the whole range).
    """
    if gain is None or linear is None or gain.order != 2:
        return None
    covered = pair_range(pairs)
    if covered is None:
        return None
    lo, hi = covered
    x = np.arange(int(nchan), dtype=float)
    outside = x[(x < lo) | (x > hi)]
    if outside.size == 0:
        return None
    diff = np.abs(gain.map(outside) - linear.map(outside))
    k = int(np.argmax(diff))
    fwhm = _largest_fwhm(gain, pairs)
    deviation = float(diff[k])
    warn = fwhm is not None and deviation > fwhm
    return Extrapolation(deviation, int(outside[k]), fwhm, warn)


def mapped_histogram(gain, counts):
    """(x, y) to draw CH1 singles on CH0's channel scale.

    Each CH1 channel i is drawn at P(i) with its counts divided by
    P'(i), i.e. as counts per CH0 channel, so a peak keeps its area and
    its height is comparable with CH0's. This is for display only, and is
    chosen over GainMatch.apply on events expanded from the histogram
    because it is exact, deterministic and costs one polynomial per
    channel: a 16k singles spectrum of a long run holds tens of millions
    of events, and expanding them would cost that much memory just to
    draw a curve. The sorter itself uses apply, event by event.
    """
    counts = np.asarray(counts, dtype=float)
    i = np.arange(counts.size, dtype=float)
    slope = gain.derivative(i)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(slope > 0, counts / slope, np.nan)
    return gain.map(i), y


def _fmt_err(value, err):
    if value is None or not np.isfinite(value):
        return ""
    if err is None or not np.isfinite(err):
        return f"{value:.2f}"
    return f"{value:.2f} ± {err:.2f}"


def _fmt(value, digits=1):
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


# ------------------------------------------------------------------ dialog


class GainMatchDialog(StepDialog):
    """Step 3: fit ch0' = P(ch1) or skip, and write it to the session.

    Reads ``session.singles`` (the default peak source), ``run_info``,
    ``exclude_values``, ``nchan`` and ``x_channel`` / ``y_channel``;
    writes ``session.gain`` and ``session.gain_skipped``. The current
    fit is ``self.gain`` (None until there is one), its pairs
    ``self.pairs``.
    """

    #: Table columns.
    COLUMNS = ("Use", "CH1 centroid", "CH0 centroid", "Energy (keV)",
               "FWHM CH1", "FWHM CH0", "Residual (ch)")

    def __init__(self, session, parent=None):
        super().__init__(session, "Sort List-Mode γγ Data – "
                         "Step 3: Gain Matching", parent)
        self._theme = getattr(parent, "_theme", "light")
        self.gain = None            # the current GainMatch
        self.linear = None          # order-1 fit of the same pairs (guard b)
        self.extrapolation = None   # Extrapolation or None
        self.pairs = []             # GainPair list shown in the table
        self.pair_reason = None     # pair_peaks' failure reason, verbatim
        self.fit_error = None       # fit_gain's refusal, verbatim
        self._calibration = None    # (ch1 counts, ch0 counts) or None
        # Recorded in the gain's "source" and so in a saved sidecar.
        self.source_description = (
            f"this run ({os.path.basename(str(session.path))})"
            if session.path else "this run")
        self.source_lines = None
        self.source_path = None
        self._updating = False
        self._build()
        self.load_from_session()

    # ---------------------------------------------------------------- UI

    def _build(self):
        intro = QLabel(
            "CH1 (X) is mapped onto CH0's (Y) ADC channel scale by a "
            "polynomial fitted to peaks both detectors see; CH0 is not "
            "changed. Find and pair the peaks, check the pairs and the "
            "order, then press Next.")
        intro.setWordWrap(True)
        self.body.addWidget(intro)

        top = QHBoxLayout()
        self.body.addLayout(top)

        # Peak source ---------------------------------------------------
        source_box = QGroupBox("Peaks from")
        sl = QVBoxLayout(source_box)
        self.this_run_radio = QRadioButton("This run")
        self.this_run_radio.setChecked(True)
        self.calibration_radio = QRadioButton("Calibration run")
        self.calibration_radio.setEnabled(False)
        sl.addWidget(self.this_run_radio)
        sl.addWidget(self.calibration_radio)
        row = QHBoxLayout()
        self.load_sdata_button = QPushButton("SData file...")
        self.load_sdata_button.setToolTip(
            "Sort another CoMPASS list-mode file into singles, with the same "
            "excluded flags and channels, and take the peaks from it.")
        self.load_spectra_button = QPushButton("Two spectra...")
        self.load_spectra_button.setToolTip(
            "Take the peaks from two 1-D spectra: one of CH0 (reference) "
            "and one of CH1 (mapped).")
        row.addWidget(self.load_sdata_button)
        row.addWidget(self.load_spectra_button)
        sl.addLayout(row)
        self.peak_source_label = QLabel("")
        self.peak_source_label.setWordWrap(True)
        sl.addWidget(self.peak_source_label)
        top.addWidget(source_box, 1)
        self.this_run_radio.toggled.connect(self._on_peak_source_toggled)
        self.load_sdata_button.clicked.connect(self._on_load_sdata_clicked)
        self.load_spectra_button.clicked.connect(self._on_load_spectra_clicked)

        # Pairing -------------------------------------------------------
        pair_box = QGroupBox("Pairing")
        pl = QGridLayout(pair_box)
        pl.addWidget(QLabel("Source file:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.setToolTip(
            "Optional. With a .sou file each channel's peaks are identified "
            "by energy and paired by energy; without one CH1's peaks are "
            "paired with CH0's by position.")
        self._fill_source_combo()
        self.source_combo.activated.connect(self._on_source_activated)
        pl.addWidget(self.source_combo, 0, 1)
        pl.addWidget(QLabel("Sensitivity:"), 1, 0)
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(2.0, 30.0)
        self.sensitivity.setSingleStep(0.5)
        self.sensitivity.setDecimals(1)
        self.sensitivity.setSuffix(" sigma")
        self.sensitivity.setValue(peak_search.DEFAULT_SENSITIVITY)
        pl.addWidget(self.sensitivity, 1, 1)
        self.find_button = QPushButton("Find and pair peaks")
        self.find_button.clicked.connect(self.find_and_pair)
        pl.addWidget(self.find_button, 2, 0, 1, 2)
        top.addWidget(pair_box, 1)

        # Fit -----------------------------------------------------------
        fit_box = QGroupBox("Fit")
        fl = QGridLayout(fit_box)
        fl.addWidget(QLabel("Low-channel cutoff:"), 0, 0)
        self.cutoff = QSpinBox()
        self.cutoff.setRange(0, max(0, int(self.session.nchan) - 1))
        self.cutoff.setValue(DEFAULT_LOW_CUTOFF)
        self.cutoff.setSuffix(" ch (CH0)")
        self.cutoff.setToolTip(
            "Pairs whose CH0 centroid is below this start unticked: the "
            "threshold edge and noise sit there, and their position is not "
            "set by the gain. Changing it re-applies the default ticks.")
        self.cutoff.valueChanged.connect(self._on_cutoff_changed)
        fl.addWidget(self.cutoff, 0, 1)
        fl.addWidget(QLabel("Order:"), 1, 0)
        self.order_combo = QComboBox()
        self.order_combo.addItems(ORDER_LABELS)
        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        fl.addWidget(self.order_combo, 1, 1)
        self.order_note = QLabel("")
        self.order_note.setWordWrap(True)
        fl.addWidget(self.order_note, 2, 0, 1, 2)
        row = QHBoxLayout()
        self.save_button = QPushButton("Save coefficients...")
        self.load_button = QPushButton("Load coefficients...")
        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)
        row.addWidget(self.save_button)
        row.addWidget(self.load_button)
        fl.addLayout(row, 3, 0, 1, 2)
        top.addWidget(fit_box, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(self.status)

        middle = QHBoxLayout()
        self.body.addLayout(middle, 1)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self._on_item_changed)
        middle.addWidget(self.table, 1)

        self.figure = Figure(figsize=(9.0, 6.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2)
        self.residual_axes = self.figure.add_subplot(grid[0, 0])
        self.map_axes = self.figure.add_subplot(grid[0, 1])
        self.overlay_axes = self.figure.add_subplot(grid[1, :])
        middle.addWidget(self.canvas, 2)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(self.summary)
        # Guard (b): a label, never a popup -- it must stay in view while
        # the user changes ticks and order, not be dismissed once.
        self.extrapolation_label = QLabel("")
        self.extrapolation_label.setWordWrap(True)
        self.extrapolation_label.setStyleSheet("color: #d04000; font-weight: bold;")
        self.extrapolation_label.setVisible(False)
        self.body.addWidget(self.extrapolation_label)

        self.skip_check = QCheckBox("Skip gain matching (CH1 keeps its own scale)")
        self.skip_check.toggled.connect(self._on_skip_toggled)
        self.body.addWidget(self.skip_check)
        self.skip_label = QLabel(SKIP_EXPLANATION)
        self.skip_label.setWordWrap(True)
        self.skip_label.setVisible(False)
        self.body.addWidget(self.skip_label)

        self._update_peak_source_label()

    def _fill_source_combo(self):
        """None, the bundled .sou files, then Browse..., so the packaged
        sources are one click away as in the calibration dialogs."""
        self.source_combo.clear()
        self.source_combo.addItem("None (pair by position)", None)
        for path in bundled_source_files():
            self.source_combo.addItem(os.path.basename(path), path)
        if self.source_path and self.source_combo.findData(self.source_path) < 0:
            self.source_combo.addItem(os.path.basename(self.source_path), self.source_path)
        self.source_combo.addItem("Browse...", "browse")
        index = self.source_combo.findData(self.source_path) if self.source_path else 0
        self.source_combo.setCurrentIndex(max(0, index))

    # ------------------------------------------------------ session state

    def load_from_session(self):
        """Show the session's choice. A stored gain (Back from step 4)
        is shown as it was chosen, pairs and ticks included; a skip shows
        the skip."""
        s = self.session
        if s.gain_skipped:
            self.skip_check.setChecked(True)
        elif s.gain is not None:
            self.skip_check.setChecked(False)
            if s.gain is not self.gain:
                self._show_gain(s.gain, note="The gain match chosen before.")
        self._update_peak_source_label()
        self._update_buttons()

    def apply_to_session(self):
        s = self.session
        if self.skip_check.isChecked():
            s.gain = None
            s.gain_skipped = True
            return None
        if self.gain is None:
            why = self.fit_error or (
                f"The peaks could not be paired: {self.pair_reason}"
                if self.pair_reason else None)
            message = ("There is no gain match yet. Find and pair the peaks, "
                       "load saved coefficients, or tick Skip gain matching.")
            return f"{why}\n\n{message}" if why else message
        s.gain = self.gain
        s.gain_skipped = False
        return None

    # ---------------------------------------------------------- spectra

    def peak_spectra(self):
        """(CH1 counts, CH0 counts) the peaks come from, or None."""
        if self.calibration_radio.isChecked() and self._calibration is not None:
            return self._calibration
        singles = self.session.singles
        if singles is None:
            return None
        try:
            ch1 = singles[self.session.x_channel]
            ch0 = singles[self.session.y_channel]
        except (KeyError, IndexError, TypeError):
            return None
        n = int(self.session.nchan)
        return _padded(ch1, n), _padded(ch0, n)

    def _on_peak_source_toggled(self, _checked):
        self._update_peak_source_label()
        self._update_buttons()
        self._redraw()

    def _update_peak_source_label(self):
        if self.calibration_radio.isChecked() and self._calibration is not None:
            text = f"In use: {self.source_description}"
        elif self.session.singles is None:
            text = "In use: this run -- but its singles are not sorted yet."
        else:
            text = "In use: this run's singles."
        self.peak_source_label.setText(text)

    def _set_calibration(self, ch1, ch0, description):
        n = max(int(self.session.nchan), len(ch1), len(ch0))
        self._calibration = (_padded(ch1, n), _padded(ch0, n))
        self.source_description = description
        self.calibration_radio.setEnabled(True)
        self.calibration_radio.setChecked(True)
        self._update_peak_source_label()
        self._update_buttons()
        self._redraw()

    def _on_load_sdata_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Calibration Run (list mode)", "",
            "CoMPASS list-mode files (*.root);;All files (*)")
        if path:
            self.load_calibration_sdata(path)

    def load_calibration_sdata(self, path):
        """Sort a calibration-run SData file into singles on a worker,
        with this run's excluded flag values and channel assignment, and
        take the peaks from it. Returns True on success."""
        import listmode_io
        import listmode_sort

        s = self.session
        channels = (int(s.y_channel), int(s.x_channel))
        exclude = s.exclude_values
        nchan = int(s.nchan)

        def work(progress):
            try:  # only for the progress bar; iter_events validates anyway
                total = listmode_io.num_entries(path)
            except Exception:  # noqa: BLE001 -- iter_events reports it
                total = 0
            done = [0]

            def chunks():
                for chunk in listmode_io.iter_events(path):
                    done[0] += len(chunk.channel)
                    progress(done[0], total)
                    yield chunk

            return listmode_sort.sort_singles(chunks(), channels=channels,
                                              nchan=nchan, exclude_flags=exclude)

        singles, error = run_with_progress(
            self, "Calibration Run", f"Sorting {os.path.basename(path)}...", work)
        if error:
            self.status.setText(f"The calibration run could not be read: {error}")
            return False
        self._set_calibration(singles[s.x_channel], singles[s.y_channel],
                              f"calibration run {os.path.basename(path)}")
        self.status.setText("Calibration run sorted. Press Find and pair peaks.")
        return True

    def _on_load_spectra_clicked(self):
        ch0, _ = QFileDialog.getOpenFileName(
            self, "Calibration Spectrum of CH0 (reference, Y)", "", SPECTRUM_FILTER)
        if not ch0:
            return
        ch1, _ = QFileDialog.getOpenFileName(
            self, "Calibration Spectrum of CH1 (mapped, X)", os.path.dirname(ch0),
            SPECTRUM_FILTER)
        if not ch1:
            return
        self.load_calibration_spectra(ch1, ch0)

    def load_calibration_spectra(self, ch1_path, ch0_path):
        """Take the peaks from two 1-D spectra. Returns True on success."""
        try:
            ch1 = _load_1d(ch1_path)
            ch0 = _load_1d(ch0_path)
        except (ParseError, OSError) as exc:
            self.status.setText(f"The spectrum could not be read: {exc}")
            return False
        self._set_calibration(
            ch1, ch0, f"spectra {os.path.basename(ch1_path)} (CH1) and "
                      f"{os.path.basename(ch0_path)} (CH0)")
        self.status.setText("Calibration spectra loaded. Press Find and pair peaks.")
        return True

    # ------------------------------------------------------------ source

    def _on_source_activated(self, index):
        data = self.source_combo.itemData(index)
        if data == "browse":
            directory = (os.path.dirname(self.source_path) if self.source_path
                         else bundled_sources_dir() or "")
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Source", directory, "Source files (*.sou);;All files (*)")
            if not path or not self.load_source(path):
                self._fill_source_combo()  # back to the previous choice
        elif data is None:
            self.load_source(None)
        else:
            self.load_source(data)

    def load_source(self, path):
        """Use the .sou at `path` for pairing by energy, or none with
        None. Returns True on success; a failure keeps the previous
        source and says why."""
        if path is None:
            self.source_lines = None
            self.source_path = None
            self._fill_source_combo()
            return True
        try:
            lines = load_sou(path)
        except ParseError as exc:
            self.status.setText(str(exc))
            return False
        self.source_lines = lines
        self.source_path = path
        self._fill_source_combo()
        self.status.setText(f"Source {os.path.basename(path)}: {len(lines)} lines; "
                            f"press Find and pair peaks to pair by energy.")
        return True

    # ---------------------------------------------------- find and pair

    def find_and_pair(self):
        """find_peaks on both spectra and pair_peaks, on a worker (a 16k
        spectrum is searched at four compressions, which takes seconds).
        A pairing failure is shown verbatim and leaves no gain."""
        spectra = self.peak_spectra()
        if spectra is None:
            self.status.setText("There are no singles to find peaks in.")
            return False
        ch1, ch0 = spectra
        sensitivity = float(self.sensitivity.value())
        lines = self.source_lines

        def work(progress):
            p1 = gain_match.find_peaks(ch1, sensitivity)
            progress(1, 3)
            p0 = gain_match.find_peaks(ch0, sensitivity)
            progress(2, 3)
            return p1, p0, gain_match.pair_peaks(p1, p0, lines)

        outcome, error = run_with_progress(
            self, "Gain Matching", "Finding and pairing peaks...", work)
        if error:
            self.set_pairs([], reason=error)
            return False
        p1, p0, paired = outcome
        if not paired.ok:
            self.set_pairs([], reason=paired.reason or "the peaks could not be paired",
                            counts=(len(p1), len(p0)))
            return False
        self.set_pairs(paired.pairs, note=paired.note, counts=(len(p1), len(p0)),
                        method=paired.method)
        return True

    def set_pairs(self, pairs, reason=None, note=None, counts=None, method=None):
        """New pairs from pair_peaks: default ticks, automatic order, fit."""
        self.pairs = [dataclasses.replace(p) for p in pairs]
        self.pair_reason = reason
        self.gain = None
        self.linear = None
        self.extrapolation = None
        self.fit_error = None
        self._apply_default_ticks()
        found = "" if counts is None else f"{counts[0]} CH1 and {counts[1]} CH0 peaks fitted. "
        if reason:
            self.status.setText(f"{found}The peaks could not be paired: {reason}")
        else:
            how = {"source": "by energy", "match": "by position",
                   "few": "by position (few peaks)"}.get(method, "")
            self.status.setText(
                f"{found}{len(self.pairs)} pairs found {how}." + (f" {note}" if note else ""))
        n_used = sum(p.included for p in self.pairs)
        auto = 2 if n_used >= gain_match.MIN_PAIRS_QUADRATIC else 1
        self._set_order(auto)
        self.refit()

    def _apply_default_ticks(self):
        """Guard (c): a pair starts ticked when the matcher trusted it and
        its CH0 centroid is at or above the cutoff. The matcher's verdict
        travels on the pair (GainPair.matcher_ok), so it survives a Back
        visit, a saved sidecar and every refit; a pair it doubted is never
        re-ticked by moving the cutoff."""
        cutoff = float(self.cutoff.value())
        for p in self.pairs:
            p.included = bool(p.matcher_ok and p.ch0 >= cutoff)

    def _on_cutoff_changed(self, _value):
        if self._updating or not self.pairs:
            return
        self._apply_default_ticks()
        self.refit()

    # -------------------------------------------------------------- fit

    def _set_order(self, order):
        self._updating = True
        try:
            self.order_combo.setCurrentIndex(int(order) - 1)
        finally:
            self._updating = False

    def selected_order(self):
        return self.order_combo.currentIndex() + 1

    def _on_order_changed(self, _index):
        if not self._updating:
            self.refit()

    def _on_item_changed(self, item):
        if self._updating or item.column() != 0:
            return
        row = item.row()
        if 0 <= row < len(self.pairs):
            self.pairs[row].included = item.checkState() == Qt.CheckState.Checked
            self.refit()

    def set_included(self, row, included):
        """Tick or untick pair `row`, as a click on its box would."""
        self.table.item(row, 0).setCheckState(
            Qt.CheckState.Checked if included else Qt.CheckState.Unchecked)

    def is_included(self, row):
        return self.table.item(row, 0).checkState() == Qt.CheckState.Checked

    def _update_order_availability(self):
        """Order 2 is offered only with MIN_PAIRS_QUADRATIC included
        pairs; with fewer the choice falls back to 1 and says why."""
        n_used = sum(p.included for p in self.pairs)
        possible = n_used >= gain_match.MIN_PAIRS_QUADRATIC
        item = self.order_combo.model().item(1)
        item.setEnabled(possible)
        if not possible and self.selected_order() == 2:
            self._set_order(1)
        if possible:
            self.order_note.setText("")
            self.order_combo.setToolTip("")
        else:
            why = (f"Order 2 needs at least {gain_match.MIN_PAIRS_QUADRATIC} included "
                   f"pairs (three fix a quadratic exactly, so nothing could "
                   f"disagree with it); {n_used} are included.")
            self.order_note.setText(why)
            self.order_combo.setToolTip(why)

    def refit(self):
        """Fit the included pairs at the selected order, and the order-1
        comparison for guard (b). Called on every change."""
        self._update_order_availability()
        self.gain = None
        self.linear = None
        self.extrapolation = None
        self.fit_error = None
        if self.pairs:
            source = self.source_description
            pairs = [dataclasses.replace(p) for p in self.pairs]
            nchan = int(self.session.nchan)
            try:
                self.gain = gain_match.fit_gain(pairs, order=self.selected_order(),
                                                source=source, nchan=nchan)
            except gain_match.GainMatchError as exc:
                self.fit_error = str(exc)
            if self.gain is not None and self.gain.order == 2:
                try:
                    self.linear = gain_match.fit_gain(pairs, order=1, nchan=nchan)
                except gain_match.GainMatchError:
                    self.linear = None
                self.extrapolation = extrapolation_check(
                    self.gain, self.linear, self.pairs, nchan)
        self._refresh()

    def _show_gain(self, gain, note=""):
        """Show a gain that was not fitted here (loaded, or from the
        session) exactly as it is, with its stored pairs and ticks."""
        self.pairs = [dataclasses.replace(p) for p in gain.pairs]
        self.pair_reason = None
        self.fit_error = None
        self.gain = gain
        self.linear = None
        self.extrapolation = None
        if self.pairs:
            self._update_order_availability()
        self._set_order(gain.order)
        if gain.order == 2:
            try:
                self.linear = gain_match.fit_gain(
                    [dataclasses.replace(p) for p in self.pairs], order=1,
                    nchan=int(self.session.nchan))
            except gain_match.GainMatchError:
                self.linear = None
            self.extrapolation = extrapolation_check(
                gain, self.linear, self.pairs, int(self.session.nchan))
        if note:
            self.status.setText(note)
        self._refresh()

    # ------------------------------------------------- save / load / skip

    def _on_save_clicked(self):
        if self.gain is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gain-Match Coefficients", "gain_match.json",
            "Gain-match files (*.json);;All files (*)")
        if path:
            self.save_coefficients(path)

    def save_coefficients(self, path):
        try:
            self.gain.save(path)
        except OSError as exc:
            self.status.setText(f"Cannot write {path}: {exc}")
            return False
        self.status.setText(f"Coefficients saved to {os.path.basename(path)}.")
        return True

    def _on_load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Gain-Match Coefficients", "",
            "Gain-match files (*.json);;All files (*)")
        if path:
            self.load_coefficients(path)

    def load_coefficients(self, path):
        """Use a saved gain match as it is; no peaks are needed. Changing
        a tick or the order afterwards refits from its stored pairs."""
        try:
            gain = gain_match.GainMatch.load(path)
        except gain_match.GainMatchError as exc:
            self.status.setText(str(exc))
            return False
        self.skip_check.setChecked(False)
        self._show_gain(gain, note=f"Coefficients loaded from {os.path.basename(path)}.")
        return True

    def _on_skip_toggled(self, checked):
        self.skip_label.setVisible(checked)
        self._update_buttons()

    def _update_buttons(self):
        skipped = self.skip_check.isChecked()
        for widget in (self.this_run_radio, self.calibration_radio,
                       self.load_sdata_button, self.load_spectra_button,
                       self.source_combo, self.sensitivity, self.cutoff,
                       self.order_combo, self.load_button, self.table):
            widget.setEnabled(not skipped)
        self.calibration_radio.setEnabled(not skipped and self._calibration is not None)
        self.find_button.setEnabled(not skipped and self.peak_spectra() is not None)
        self.save_button.setEnabled(not skipped and self.gain is not None)
        # Without pairs (coefficients loaded from a file that stored
        # none) there is nothing to refit at another order or with other
        # ticks, so the controls that would refit are off.
        self.order_combo.setEnabled(not skipped and bool(self.pairs))
        self.cutoff.setEnabled(not skipped and bool(self.pairs))

    # ---------------------------------------------------------- display

    def _refresh(self):
        self._fill_table()
        self._update_summary()
        self._update_buttons()
        self._redraw()

    def _fill_table(self):
        self._updating = True
        try:
            residuals = self.gain.residuals if self.gain is not None else []
            self.table.setRowCount(len(self.pairs))
            for row, p in enumerate(self.pairs):
                use = QTableWidgetItem("")
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                use.setCheckState(Qt.CheckState.Checked if p.included
                                  else Qt.CheckState.Unchecked)
                self.table.setItem(row, 0, use)
                residual = residuals[row] if row < len(residuals) else None
                cells = (_fmt_err(p.ch1, p.ch1_err), _fmt_err(p.ch0, p.ch0_err),
                         "" if p.energy is None else f"{p.energy:g}",
                         _fmt(p.fwhm1), _fmt(p.fwhm0), _fmt(residual, 2))
                for col, text in enumerate(cells, start=1):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(row, col, item)
        finally:
            self._updating = False

    def _update_summary(self):
        g = self.gain
        covered = pair_range(self.pairs)
        range_text = ("no included pairs" if covered is None else
                      f"pairs cover CH1 channels {covered[0]:.0f}-{covered[1]:.0f} "
                      f"of 0-{int(self.session.nchan) - 1}; the map is "
                      f"extrapolated outside that range")
        if g is None:
            text = self.fit_error or ""
            if self.pairs and covered is not None:
                text = (text + " " if text else "") + f"({range_text})"
            self.summary.setText(text)
        else:
            terms = " ".join(
                f"{'+' if c >= 0 and k else ''}{c:.6g}{'' if k == 0 else ' ch1' if k == 1 else ' ch1^2'}"
                for k, c in enumerate(g.coefficients))
            chi = ("" if g.reduced_chi2 is None
                   else f", reduced chi2 {g.reduced_chi2:.3g}")
            rms = "" if not np.isfinite(g.rms) else f"rms {g.rms:.3g} ch{chi}; "
            self.summary.setText(f"ch0 = {terms}   ({rms}{range_text})")
        x = self.extrapolation
        if x is not None and x.warn:
            self.extrapolation_label.setText(
                f"Warning: outside the pair range the quadratic leaves the "
                f"straight line through the same pairs by up to "
                f"{x.deviation:.0f} channels (at CH1 channel {x.channel}), more "
                f"than one FWHM ({x.fwhm:.0f} ch). Nothing constrains the "
                f"curvature there; consider order 1, or pairs that reach "
                f"further.")
            self.extrapolation_label.setVisible(True)
        else:
            self.extrapolation_label.setText("")
            self.extrapolation_label.setVisible(False)

    def extrapolation_warning_shown(self):
        """Whether guard (b)'s warning is on screen (isVisible needs a
        shown window; this does not)."""
        return not self.extrapolation_label.isHidden()

    def _redraw(self):
        theme = self._theme
        nchan = int(self.session.nchan)
        for axes in (self.residual_axes, self.map_axes, self.overlay_axes):
            axes.clear()
            style_axes(axes, theme)
        g = self.gain

        # (1) residuals per pair; excluded ones hollow, so the evidence
        # for leaving them out stays visible.
        ax = self.residual_axes
        ax.set_title("Residuals ch0 - P(ch1)", fontsize=9)
        ax.set_xlabel("CH1 channel", fontsize=8)
        ax.axhline(0.0, color="gray", lw=0.8)
        if g is not None and self.pairs:
            for p, r in zip(self.pairs, g.residuals):
                ax.errorbar([p.ch1], [r], yerr=[p.ch0_err], fmt="o", color="C0",
                            mfc="C0" if p.included else "none", ms=5)

        # (2) the map over the whole ADC range, guard (a).
        ax = self.map_axes
        ax.set_title("Map ch0 = P(ch1)", fontsize=9)
        ax.set_xlabel("CH1 channel", fontsize=8)
        x = np.arange(nchan, dtype=float)
        covered = pair_range(self.pairs)
        if g is not None:
            if covered is None:
                ax.plot(x, g.map(x), "--", color="C0", lw=1)
            else:
                lo, hi = covered
                inside = (x >= lo) & (x <= hi)
                ax.plot(x[inside], g.map(x[inside]), "-", color="C0", lw=1.5,
                        label="fitted (pair range)")
                for part in (x < lo, x > hi):
                    if part.any():
                        ax.plot(x[part], g.map(x[part]), "--", color="C0", lw=1)
                if lo > 0:
                    ax.axvspan(0, lo, color="gray", alpha=0.2, lw=0)
                if hi < nchan - 1:
                    ax.axvspan(hi, nchan - 1, color="gray", alpha=0.2, lw=0,
                               label="extrapolated")
            if self.linear is not None:
                ax.plot(x, self.linear.map(x), ":", color="C1", lw=1, label="order 1")
            if self.pairs:
                ax.plot([p.ch1 for p in self.pairs], [p.ch0 for p in self.pairs],
                        "o", color="C3", ms=4)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, nchan - 1)

        # (3) CH0 and mapped CH1 singles overlaid.
        ax = self.overlay_axes
        ax.set_title("CH0 and gain-matched CH1 singles", fontsize=9)
        ax.set_xlabel("CH0 channel", fontsize=8)
        spectra = self.peak_spectra()
        if spectra is not None:
            ch1, ch0 = spectra
            ax.plot(np.arange(ch0.size), ch0, color="C0", lw=0.8, label="CH0")
            if g is not None:
                mx, my = mapped_histogram(g, ch1)
                ax.plot(mx, my, color="C1", lw=0.8, label="CH1 mapped")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlim(0, nchan - 1)
            ax.legend(fontsize=7, loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()
