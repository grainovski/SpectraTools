"""Step 4 of Sort List-Mode gamma-gamma Data: the prompt and random
time windows on the dt = t(CH1) - t(CH0) spectrum.

The user marks one prompt window and one or more random windows; the
dialog shows, live, what listmode_sort will make of them: the counts in
each window, the inclusive bin widths, w = N_prompt / N_random
(window_weight, TV's MrkRange rule), the random level expected under the
prompt and the net prompt. Anything window_weight refuses (overlap, a
window holding no bin, a window outside the sorted range) is shown in
the dialog and blocks Next, so step 5 never starts from windows the
sorter would reject.

Choices worth knowing about:

* **Everything is in dt bins.** The axis is binned at the detected
  timestamp granularity, and the sorted range is the step-1 maximum |dt|
  snapped DOWN to a multiple of it (Task 2 review). With 10 ns ticks every
  dt is a multiple of the granularity, so bin k holds exactly the tick
  k * g; a range that is not a multiple would make the outermost bin
  half-populated and a random window reaching it would under-measure.
  The counts shown are sums over histogram bins, which are the same bins
  listmode_sort.window_mask selects with the granularity passed in --
  the test suite checks the two agree -- but cost nothing per edit,
  where a mask over 10**7 pairs would lag every spin-box click.

* **Marking: SpanSelector plus a mode switch, with hold keys as a
  shortcut.** A drag is the natural gesture for a window (two clicks per
  window, the matrix panel's habit, needs the user to remember which
  click they are on). The radio buttons say what a drag does; holding P
  or R while dragging overrides them for that drag, which is the matrix
  panel's hold-key habit for users who have it in their fingers. A drag
  is snapped to whole bins (nearest bin centre at each end). The spin
  boxes take any value; window_bins counts the bins whose centres lie
  inside, exactly as the sorter will.

* **Defaults.** Prompt: the maximum bin +-10 ns (at least +-1 bin). On
  the Na-22 reference run the prompt peak occupies the -10/0/+10 ns
  ticks (39,924 pairs at 0, 43,387 within +-10 ns), so +-1 tick is the
  whole peak at 10 ns granularity, and +-10 ns stays physically sensible
  if a finer timestamp ever arrives. Random: the two side bands from
  +-500 ns to the edge of the range -- far enough from the prompt that no
  true coincidence tail reaches them, wide enough that the random level
  is well measured (on the reference run 302 bins against 3 prompt
  bins). The user adjusts both.
"""

import math

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

import theme
from fit_mode import is_bare_key_event
from listmode_session import StepDialog
from listmode_sort import SortError, dt_histogram, window_bins, window_weight

TITLE = "Sort List-Mode γγ Data – Step 4 of 5: Time Windows"

#: Default prompt half-width around the maximum bin (see module docstring).
DEFAULT_PROMPT_HALF_WIDTH_PS = 10_000
#: Default inner edge of the two random side bands.
DEFAULT_RANDOM_INNER_PS = 500_000

#: Fewer pairs than this in all random windows together and the random
#: level is not measured, only guessed: 10 pairs is a 32% (1/sqrt(10))
#: uncertainty on everything the subtraction removes. Warned about, not
#: refused -- a clean run with a very narrow range can be genuinely
#: random-free. A FILTERED stream holds none at all (CoMPASS's software
#: coincidence removed them; see the plan's "Facts about the data").
RANDOM_WARN_MIN = 10

PROMPT_COLOR = "#d62728"
RANDOM_COLOR = "#1f77b4"
LIGHT_LINE_COLOR = "black"
DARK_LINE_COLOR = "#e0e0e0"   # theme.DARK_TEXT

MODE_PROMPT = "prompt"
MODE_RANDOM = "random"


def snapped_max_dt(max_dt_ps, granularity_ps):
    """The sorted dt range: max |dt| rounded DOWN to a multiple of the
    granularity, so the outermost bin is a whole bin (see the module
    docstring)."""
    g = int(granularity_ps)
    return (int(max_dt_ps) // g) * g


def default_windows(centres, counts, granularity_ps, max_dt_ps):
    """(prompt, randoms) for a first visit, in ps, on bin centres.

    Prompt: the maximum bin +- DEFAULT_PROMPT_HALF_WIDTH_PS (at least one
    bin each side). Randoms: +-[DEFAULT_RANDOM_INNER_PS, max]; when the
    range is too short for that, the outer half of each side, and none
    at all when not even that leaves room beside the prompt (the dialog
    then says so and the user places them).
    """
    g = int(granularity_ps)
    k_max = int(max_dt_ps) // g
    counts = np.asarray(counts)
    peak = int(centres[int(np.argmax(counts))] // g) if counts.sum() > 0 else 0
    half = max(1, int(round(DEFAULT_PROMPT_HALF_WIDTH_PS / g)))
    lo, hi = max(-k_max, peak - half), min(k_max, peak + half)
    prompt = (lo * g, hi * g)
    inner = math.ceil(DEFAULT_RANDOM_INNER_PS / g)
    if inner >= k_max:
        inner = (k_max + 1) // 2
    inner = max(inner, abs(peak) + half + 1)
    if inner > k_max:
        return prompt, []
    return prompt, [(-k_max * g, -inner * g), (inner * g, k_max * g)]


def _decimals(granularity_ps):
    """Spin-box decimals in ns that show one granularity step exactly."""
    g = int(granularity_ps)
    for d in range(4):
        if (g * 10 ** d) % 1000 == 0:
            return d
    return 3


class _HeldKeyFilter(QObject):
    """Tracks a bare P or R held on the canvas (matrix panel habit)."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self.held = None
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        canvas.mpl_connect("figure_enter_event", lambda _e: canvas.setFocus())

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if is_bare_key_event(event):
                if event.key() == Qt.Key.Key_P:
                    self.held = MODE_PROMPT
                elif event.key() == Qt.Key.Key_R:
                    self.held = MODE_RANDOM
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self.held = None
        return False


class _WindowRow(QWidget):
    """One window: lo and hi in ns, its readout, and (randoms) Remove."""

    def __init__(self, label, max_ns, step_ns, decimals, removable, on_change,
                 on_remove=None):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        row.addWidget(self.name)
        self.lo = self._spin(max_ns, step_ns, decimals, on_change)
        self.hi = self._spin(max_ns, step_ns, decimals, on_change)
        row.addWidget(QLabel("from"))
        row.addWidget(self.lo)
        row.addWidget(QLabel("to"))
        row.addWidget(self.hi)
        row.addWidget(QLabel("ns"))
        self.readout = QLabel()
        row.addWidget(self.readout, 1)
        self.remove_button = None
        if removable:
            self.remove_button = QPushButton("Remove")
            self.remove_button.clicked.connect(lambda: on_remove(self))
            row.addWidget(self.remove_button)

    @staticmethod
    def _spin(max_ns, step_ns, decimals, on_change):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(-max_ns, max_ns)
        spin.setSingleStep(step_ns)
        # Only on Enter / focus-out / arrows: typing "-500" must not flash
        # an overlap error at "-5".
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(lambda _v: on_change())
        return spin

    def set_window(self, window):
        for spin, ps in zip((self.lo, self.hi), window):
            spin.blockSignals(True)
            spin.setValue(ps / 1000.0)
            spin.blockSignals(False)

    def window(self):
        lo = int(round(self.lo.value() * 1000))
        hi = int(round(self.hi.value() * 1000))
        return (min(lo, hi), max(lo, hi))


class TimeWindowDialog(StepDialog):
    """Step 4: mark the prompt and random dt windows.

    Public for the controller and tests: ``windows()``,
    ``set_windows(prompt, randoms)``, ``add_random_window(window=None)``,
    ``remove_random_window(index)``, ``mark_span(xmin_ns, xmax_ns,
    mode=None)`` (what a drag does), ``readout`` (the numbers shown, a
    dict) and ``set_log_y(bool)``.
    """

    def __init__(self, session, parent=None):
        super().__init__(session, TITLE, parent)
        self._theme = getattr(parent, "_theme", "light")
        self._fatal = None
        self.readout = {}
        self._span_artists = []
        self._random_rows = []

        g = session.granularity_ps
        pairs = session.pairs
        if not g or int(g) <= 0:
            self._fatal = ("The timestamp granularity is not known (every "
                           "timestamp was zero), so there is no time axis.")
            g = 1
        if pairs is None:
            self._fatal = "There are no pairs yet; go back to the input step."
        self._g = int(g)
        self._max = snapped_max_dt(session.max_dt_ps, self._g)
        dt = np.zeros(0, dtype=np.int64) if pairs is None else pairs.dt
        self._centres, self._counts = dt_histogram(dt, self._g, self._max)
        self._k = len(self._centres) // 2
        self._cumsum = np.concatenate([[0], np.cumsum(self._counts)])

        self._build_plot()
        self._build_controls()
        self.load_from_session()

    # ------------------------------------------------------------ building
    def _build_plot(self):
        self._figure = Figure(figsize=(7.5, 3.6))
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.axes = self._figure.add_subplot(1, 1, 1)
        theme.style_axes(self.axes, self._theme)
        line_color = DARK_LINE_COLOR if self._theme == "dark" else LIGHT_LINE_COLOR
        self.axes.step(self._centres / 1000.0, self._counts, where="mid",
                       color=line_color, linewidth=0.9)
        self.axes.set_xlabel("Δt = t(CH1) − t(CH0)  [ns]")
        self.axes.set_ylabel(f"Pairs / {self._g / 1000:g} ns")
        self.axes.set_xlim(-self._max / 1000.0 - self._g / 2000.0,
                           self._max / 1000.0 + self._g / 2000.0)
        self._figure.tight_layout()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        theme.style_nav_toolbar_palette(self.toolbar, self._theme)
        self.body.addWidget(self.toolbar)
        self.body.addWidget(self.canvas, 1)
        self._keys = _HeldKeyFilter(self.canvas)
        self._selector = SpanSelector(
            self.axes, self._on_select, "horizontal", useblit=False,
            props=dict(alpha=0.2, facecolor="grey"), interactive=False)

    def _build_controls(self):
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("A drag on the plot:"))
        self.prompt_mode = QRadioButton("sets the prompt window")
        self.random_mode = QRadioButton("adds a random window")
        self.prompt_mode.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.prompt_mode)
        group.addButton(self.random_mode)
        mode_row.addWidget(self.prompt_mode)
        mode_row.addWidget(self.random_mode)
        mode_row.addWidget(QLabel("(or hold P / R while dragging)"))
        mode_row.addStretch(1)
        self.log_y = QCheckBox("Log scale")
        self.log_y.toggled.connect(self.set_log_y)
        mode_row.addWidget(self.log_y)
        self.body.addLayout(mode_row)

        max_ns = self._max / 1000.0
        step_ns = self._g / 1000.0
        dec = _decimals(self._g)
        self._row_args = (max_ns, step_ns, dec)

        prompt_box = QGroupBox("Prompt window")
        pl = QVBoxLayout(prompt_box)
        self.prompt_row = _WindowRow("Prompt", max_ns, step_ns, dec, False,
                                     self._refresh)
        pl.addWidget(self.prompt_row)
        self.body.addWidget(prompt_box)

        random_box = QGroupBox("Random windows")
        self._random_layout = QVBoxLayout(random_box)
        add_row = QHBoxLayout()
        self.add_random_button = QPushButton("Add random window")
        self.add_random_button.clicked.connect(lambda: self.add_random_window())
        add_row.addWidget(self.add_random_button)
        add_row.addStretch(1)
        self._random_layout.addLayout(add_row)
        self.body.addWidget(random_box)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(self.summary_label)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #d62728;")
        self.body.addWidget(self.error_label)
        # A warning, not an error: Next stays enabled.
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #d04000; font-weight: bold;")
        self.warning_label.setVisible(False)
        self.body.addWidget(self.warning_label)

    # ------------------------------------------------------------ windows
    def windows(self):
        """(prompt, [randoms]) in ps, as the spin boxes say."""
        return self.prompt_row.window(), [r.window() for r in self._random_rows]

    def set_windows(self, prompt, randoms):
        self.prompt_row.set_window(prompt)
        for row in list(self._random_rows):
            self._drop_row(row)
        for w in randoms:
            self._new_row(w)
        self._refresh()

    def _new_row(self, window):
        row = _WindowRow(f"Random {len(self._random_rows) + 1}", *self._row_args,
                         True, self._refresh, self._on_remove_row)
        row.set_window(window)
        self._random_rows.append(row)
        # Above the "Add" button, which is the last item.
        self._random_layout.insertWidget(self._random_layout.count() - 1, row)
        return row

    def _drop_row(self, row):
        self._random_rows.remove(row)
        self._random_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        for i, r in enumerate(self._random_rows):
            r.name.setText(f"Random {i + 1}")

    def _on_remove_row(self, row):
        self._drop_row(row)
        self._refresh()

    def remove_random_window(self, index):
        self._drop_row(self._random_rows[index])
        self._refresh()

    def add_random_window(self, window=None):
        """Add `window` (ps), or by default the widest stretch of bins no
        window touches, one bin clear of its neighbours. Returns the
        window added, or None when there is no room."""
        if window is None:
            window = self._free_span()
            if window is None:
                self.error_label.setText(
                    "There is no free part of the range left for another "
                    "random window; shrink or remove one first.")
                return None
        self._new_row(window)
        self._refresh()
        return window

    def _free_span(self):
        k = self._k
        used = np.zeros(2 * k + 1, dtype=bool)
        prompt, randoms = self.windows()
        for w in [prompt] + randoms:
            b = window_bins(w, self._g, self._max)
            if b is not None:
                used[max(0, b[0] + k - 1):min(2 * k + 1, b[1] + k + 2)] = True
        best, start = None, None
        for i in range(2 * k + 2):
            free = i <= 2 * k and not used[i]
            if free and start is None:
                start = i
            elif not free and start is not None:
                if best is None or i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        if best is None:
            return None
        return ((best[0] - k) * self._g, (best[1] - 1 - k) * self._g)

    def mark_span(self, xmin_ns, xmax_ns, mode=None):
        """What a drag from xmin to xmax (ns) does: snap both ends to the
        nearest bin centre inside the range, then set the prompt or add a
        random window according to `mode` (default: a held P/R key, else
        the radio buttons)."""
        if mode is None:
            mode = self._keys.held or (
                MODE_PROMPT if self.prompt_mode.isChecked() else MODE_RANDOM)
        k = self._k
        lo = min(max(int(round(min(xmin_ns, xmax_ns) * 1000 / self._g)), -k), k)
        hi = min(max(int(round(max(xmin_ns, xmax_ns) * 1000 / self._g)), -k), k)
        window = (lo * self._g, hi * self._g)
        if mode == MODE_PROMPT:
            self.prompt_row.set_window(window)
            self._refresh()
        else:
            self.add_random_window(window)
        return window

    def _on_select(self, xmin, xmax):
        try:
            self.mark_span(xmin, xmax)
        except Exception as exc:  # never let a matplotlib callback swallow it
            self.error_label.setText(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ readout
    def _bins(self, window):
        return window_bins(window, self._g, self._max)

    def window_counts(self, window):
        """Pairs whose dt bin lies in `window` (clipped to the range)."""
        b = self._bins(window)
        if b is None:
            return 0
        return int(self._cumsum[b[1] + self._k + 1] - self._cumsum[b[0] + self._k])

    def _describe(self, window):
        b = self._bins(window)
        if b is None:
            return "no bin inside"
        n = b[1] - b[0] + 1
        return f"{n:,} bin{'s' if n != 1 else ''}, {self.window_counts(window):,} pairs"

    def _refresh(self):
        prompt, randoms = self.windows()
        self.prompt_row.readout.setText(self._describe(prompt))
        for row, w in zip(self._random_rows, randoms):
            row.readout.setText(self._describe(w))
        n_p = self.window_counts(prompt)
        n_r_each = [self.window_counts(w) for w in randoms]
        n_r = sum(n_r_each)
        readout = {"prompt_counts": n_p, "random_counts": n_r_each,
                   "random_total": n_r, "w": None, "n_prompt_bins": None,
                   "n_random_bins": None, "expected_random": None,
                   "net": None, "net_sigma": None, "error": None,
                   "random_warning": None}
        error = self._fatal
        if error is None:
            try:
                w, nbp, nbr = window_weight(prompt, randoms, self._g, self._max)
            except SortError as exc:
                error = str(exc)
            else:
                expected = w * n_r
                readout.update(w=w, n_prompt_bins=nbp, n_random_bins=nbr,
                               expected_random=expected, net=n_p - expected,
                               net_sigma=math.sqrt(n_p + w * w * n_r))
        readout["error"] = error
        readout["random_warning"] = (
            None if error is not None else self._random_warning(n_r))
        self.readout = readout
        if readout["w"] is not None:
            self.summary_label.setText(
                f"Prompt: {readout['n_prompt_bins']:,} bins, {n_p:,} pairs.   "
                f"Random: {readout['n_random_bins']:,} bins, {n_r:,} pairs.\n"
                f"w = N_prompt / N_random = {readout['n_prompt_bins']:,} / "
                f"{readout['n_random_bins']:,} = {readout['w']:.6g}.   "
                f"Random level under the prompt: w × {n_r:,} = "
                f"{readout['expected_random']:,.1f}.   "
                f"Net prompt: {readout['net']:,.1f} ± {readout['net_sigma']:,.1f}.")
        else:
            self.summary_label.setText(
                f"Prompt: {n_p:,} pairs.   Random: {n_r:,} pairs.")
        self.error_label.setText(error or "")
        self.warning_label.setText(readout["random_warning"] or "")
        self.warning_label.setVisible(bool(readout["random_warning"]))
        self.next_button.setEnabled(error is None)
        self._draw_spans(prompt, randoms)

    def _random_warning(self, n_random):
        """The warning for random windows that hold (almost) nothing, or
        None. Says why when the input is CoMPASS's FILTERED stream."""
        if n_random >= RANDOM_WARN_MIN:
            return None
        lm = self.session.listmode
        filtered = lm is not None and getattr(lm, "stream", None) == "FILTERED"
        held = ("no pairs at all" if n_random == 0 else
                f"only {n_random:,} pair{'s' if n_random != 1 else ''}")
        text = (f"Warning: the random windows hold {held}, so the random "
                f"level under the prompt is not measured and the "
                f"subtraction removes (almost) nothing.")
        if filtered:
            text += (" This is CoMPASS's FILTERED stream, whose software "
                     "coincidence has already removed the pairs away from "
                     "the prompt peak; sort the UNFILTERED file instead.")
        else:
            text += " Widen the random windows or the pairing range in step 1."
        return text

    def _draw_spans(self, prompt, randoms):
        for artist in self._span_artists:
            try:
                artist.remove()
            except (NotImplementedError, ValueError):
                pass
        self._span_artists = []
        g = self._g / 1000.0
        for w, color in [(prompt, PROMPT_COLOR)] + [(r, RANDOM_COLOR) for r in randoms]:
            b = self._bins(w)
            if b is None:
                continue
            self._span_artists.append(self.axes.axvspan(
                (b[0] - 0.5) * g, (b[1] + 0.5) * g, color=color, alpha=0.25,
                linewidth=0))
        self.canvas.draw_idle()

    def set_log_y(self, on):
        if self.log_y.isChecked() != bool(on):
            self.log_y.setChecked(bool(on))   # re-enters via toggled
            return
        if on:
            self.axes.set_yscale("log", nonpositive="clip")
            top = max(1.0, float(self._counts.max()) if len(self._counts) else 1.0)
            self.axes.set_ylim(0.5, top * 2.0)
        else:
            self.axes.set_yscale("linear")
            self.axes.relim()
            self.axes.autoscale(axis="y")
        self.canvas.draw_idle()

    # ------------------------------------------------------------ session
    def load_from_session(self):
        s = self.session
        if s.prompt_window is not None:
            self.set_windows(tuple(s.prompt_window), [tuple(w) for w in s.random_windows])
        else:
            self.set_windows(*default_windows(self._centres, self._counts,
                                              self._g, self._max))

    def apply_to_session(self):
        if self._fatal:
            return self._fatal
        prompt, randoms = self.windows()
        try:
            w, n_p, n_r = window_weight(prompt, randoms, self._g, self._max)
        except SortError as exc:
            return str(exc)
        s = self.session
        old_prompt = None if s.prompt_window is None else tuple(s.prompt_window)
        old_randoms = [tuple(r) for r in s.random_windows]
        if old_prompt != prompt or old_randoms != randoms or s.weight is None:
            # New windows make any sorted matrices stale.
            s.invalidate_from(5)
        s.prompt_window = prompt
        s.random_windows = list(randoms)
        s.weight = w
        s.n_prompt_bins = n_p
        s.n_random_bins = n_r
        return None
