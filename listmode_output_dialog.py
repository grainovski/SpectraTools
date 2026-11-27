"""Step 5 of Sort List-Mode gamma-gamma Data: sort the prompt and random
matrices, subtract, and write the result.

What happens on "Sort and Save", all on a worker thread
(listmode_session.run_with_progress), because sorting 10**7 pairs and
writing a .mtx in pure Python both take seconds to tens of seconds:

1. P and R are sorted from the pairs whose dt BIN lies in the prompt /
   random windows -- window_mask with the granularity, so the pairs
   summed are the bins window_weight counted (Task 2 review). CH1 (X) is
   gain-matched onto CH0's scale per event by GainMatch.apply; a mapped
   channel outside 0..nchan-1 is sent to -1, which sort_matrix drops and
   counts (it drops anything outside the shape rather than clipping it
   onto the edge, where it would build a false ridge). With the gain
   step skipped X is used as it is.
2. S = P - w R in float64 (listmode_sort.subtract).
3. The ticked files: .mtx from stochastic_round(S) (integer-only format;
   E[cell] = S), sparse JSON with S's exact floats, and dense
   CoMPASS-layout JSON of S compressed by f with
   json_matrix_io.compress_matrix -- the one compress function used for
   output (Task 4 carry-over; listmode_sort.compress is not used here).

Seeds: one base seed (matrix_symmetrize.DEFAULT_SEED) recorded in the
metadata, with distinct derived streams for the prompt's gain dither
(seed), the random's (seed + 1) and the .mtx rounding (seed + 2), so no
two random draws share a stream and the whole result is reproducible.

Each file is written to "<name>.partial" and renamed into place only
when complete, so a failure (disk full, a value the format refuses)
leaves no truncated file under the real name and is never listed as
written. Orientation: the app's [y][x], rows = CH0 (Y), columns = CH1
(X); the writers turn that into each file's own layout.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from json_matrix_io import (
    DENSE_MAX, axis_for_compression, compress_matrix, save_json_matrix,
)
from listmode_session import StepDialog, run_with_progress
from listmode_sort import (
    SortError, sort_matrix, stochastic_round, subtract, to_dense, window_mask,
)
from matrix_symmetrize import DEFAULT_SEED
from mtx_io import save_mtx
import system_memory

TITLE = "Sort List-Mode γγ Data – Step 5 of 5: Sort and Save"

#: Compression factors offered for the dense CoMPASS-layout JSON.
CANDIDATE_FACTORS = (1, 2, 4, 8, 16, 32, 64)
#: CoMPASS's own E-E export is the 16k ADC on a x16 grid, so x16 overlays
#: it directly; used as the default whenever it is allowed.
PREFERRED_FACTOR = 16
#: Opening a dense matrix at least this big asks first (16384^2 float64
#: is 2 GiB); smaller ones open straight away.
OPEN_WARN_BYTES = 256 * 2**20
#: What a Matrix panel holds beyond the dense matrix itself. Measured
#: 2026-11-27 on NUCLINO-IV (Windows 11, process working set from the OS)
#: by opening a 16384 x 16384 float64 result through Open result: +2,090
#: MiB resident with the panel open (the matrix is 2,048 MiB), peak +2,118
#: MiB, a heatmap ~25 MiB more (peak +55), all of it released on close --
#: the same with 10**5 and 10**7 non-zero cells. 128 MiB covers it.
OPEN_OVERHEAD_BYTES = 128 * 2**20

_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_STREAM_PREFIX = re.compile(r"^SData[FR]?_", re.IGNORECASE)


# ----------------------------------------------------------------------
# Pure helpers (no widgets), shared by the dialog and the tests
# ----------------------------------------------------------------------

def dense_factors(nchan):
    """Compression factors that divide nchan and give <= DENSE_MAX bins
    per axis (compress_matrix refuses a ragged last bin, and
    save_json_matrix refuses dense output above DENSE_MAX)."""
    n = int(nchan)
    return [f for f in CANDIDATE_FACTORS if n % f == 0 and n // f <= DENSE_MAX]


def run_id(session):
    """The run's id: from <run>_info.txt if it was read, else the SData
    file name without its SData/SDataF/SDataR prefix."""
    info = session.run_info
    if info is not None and getattr(info, "run_id", None):
        return str(info.run_id)
    if session.path:
        return _STREAM_PREFIX.sub("", Path(session.path).stem) or Path(session.path).stem
    return "listmode"


def default_folder(session):
    """The run folder: the parent of the stream folder (UNFILTERED/,
    FILTERED/, RAW/) the SData file is in, where CoMPASS keeps
    <run>_info.txt and settings.xml."""
    if session.path:
        return Path(session.path).resolve().parent.parent
    return Path.home()


def default_base_name(session):
    return f"{run_id(session)}_gg_sub"


@dataclass
class OutputChoice:
    folder: Path
    base: str
    write_mtx: bool = True
    write_sparse: bool = True
    dense_factor: object = None     # int, or None for no dense JSON

    def files(self):
        """[(kind, Path)] in writing order; kind is 'mtx', 'sparse' or
        'dense'."""
        out = []
        folder = Path(self.folder)
        if self.write_mtx:
            out.append(("mtx", folder / f"{self.base}.mtx"))
        if self.write_sparse:
            out.append(("sparse", folder / f"{self.base}.json"))
        if self.dense_factor is not None:
            out.append(("dense", folder / f"{self.base}_x{int(self.dense_factor)}.json"))
        return out


@dataclass
class SortOutcome:
    prompt: object = None           # listmode_sort.MatrixSort
    random: object = None
    result: object = None           # SparseMatrix, float S
    n_prompt_pairs: int = 0         # pairs in the prompt window
    n_random_pairs: int = 0
    w: float = 0.0
    seed: int = DEFAULT_SEED
    meta: dict = field(default_factory=dict)
    written: list = field(default_factory=list)   # [(Path, bytes)]
    error: str = None               # a writer failed after the sort

    @property
    def dropped(self):
        return int(self.prompt.dropped + self.random.dropped)


def _x_map(session, seed):
    """X energies -> CH0-scale channels, out-of-range ones as -1 (which
    sort_matrix drops and counts); None (identity) when gain matching
    was skipped."""
    if session.gain_skipped:
        return None
    if session.gain is None:
        raise SortError("Gain matching has not been done; go back to step 3, "
                        "or skip it there explicitly.")
    gain, nchan = session.gain, int(session.nchan)

    def apply(e):
        channels, ok = gain.apply(np.asarray(e).astype(np.int64), seed=seed, nchan=nchan)
        return np.where(ok, channels, -1)
    return apply


def _meta(session, outcome, compression):
    s = session
    gain = None if s.gain_skipped or s.gain is None else list(s.gain.coefficients)
    return {
        "compression": int(compression),
        "source_run": {"path": None if s.path is None else str(s.path),
                       "run_id": run_id(s)},
        "windows": {"unit": "ps",
                    "granularity_ps": int(s.granularity_ps),
                    "prompt": [int(v) for v in s.prompt_window],
                    "randoms": [[int(v) for v in r] for r in s.random_windows],
                    "n_prompt_bins": int(s.n_prompt_bins),
                    "n_random_bins": int(s.n_random_bins)},
        "w": float(outcome.w),
        "seed": int(outcome.seed),
        "seeds": {"gain_dither_prompt": int(outcome.seed),
                  "gain_dither_random": int(outcome.seed) + 1,
                  "mtx_stochastic_round": int(outcome.seed) + 2},
        "gain_coefficients": gain,
        "gain": "skipped" if gain is None else "CH1 mapped onto CH0",
        "axes": {"x": "CH1", "y": "CH0"},
        "nchan": int(s.nchan),
        "subtraction": "S = P - w * R",
        "totals": {"prompt": int(outcome.prompt.matrix.total()),
                   "random": int(outcome.random.matrix.total()),
                   "net": float(outcome.result.total())},
        "pairs_in_windows": {"prompt": int(outcome.n_prompt_pairs),
                             "random": int(outcome.n_random_pairs)},
        "dropped": {"prompt": int(outcome.prompt.dropped),
                    "random": int(outcome.random.dropped)},
    }


def _write_atomically(path, write):
    """write(tmp_path), then rename onto `path`; the temporary file is
    removed if anything fails, so `path` is either complete or untouched."""
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path.stat().st_size


def sort_and_write(session, choice, progress=lambda d, t: None, seed=DEFAULT_SEED):
    """Sort P and R, subtract, and write `choice`'s files. Runs on the
    worker thread; returns a SortOutcome. A sorting problem raises; a
    writer problem is returned in outcome.error with the files finished
    before it in outcome.written."""
    s = session
    if s.pairs is None:
        raise SortError("There are no pairs; go back to the input step.")
    if s.prompt_window is None or not s.random_windows or s.weight is None:
        raise SortError("The time windows are not set; go back to step 4.")
    g = int(s.granularity_ps)
    nchan = int(s.nchan)
    shape = (nchan, nchan)
    files = choice.files()
    total = 100 + sum(800 if k == "mtx" else 50 for k, _ in files)
    progress(0, total)

    pairs = s.pairs
    prompt_mask = window_mask(pairs.dt, s.prompt_window, g)
    random_mask = window_mask(pairs.dt, s.random_windows, g)
    outcome = SortOutcome(w=float(s.weight), seed=int(seed),
                          n_prompt_pairs=int(prompt_mask.sum()),
                          n_random_pairs=int(random_mask.sum()))
    outcome.prompt = sort_matrix(pairs.ex, pairs.ey, mask=prompt_mask, shape=shape,
                                 x_map=_x_map(s, seed))
    progress(40, total)
    outcome.random = sort_matrix(pairs.ex, pairs.ey, mask=random_mask, shape=shape,
                                 x_map=_x_map(s, seed + 1))
    progress(80, total)
    outcome.result = subtract(outcome.prompt.matrix, outcome.random.matrix, outcome.w)
    outcome.meta = _meta(s, outcome, 1)
    progress(100, total)

    done = 100
    S = outcome.result
    for kind, path in files:
        try:
            if kind == "mtx":
                rounded = stochastic_round(S, seed=seed + 2)
                base = done

                def mtx_progress(rows, nrows, base=base):
                    progress(base + (800 * rows // max(nrows, 1)), total)
                size = _write_atomically(path, lambda p: save_mtx(
                    p, (shape, rounded.index, rounded.values), progress=mtx_progress))
                done += 800
            elif kind == "sparse":
                size = _write_atomically(path, lambda p: save_json_matrix(
                    p, (shape, S.index, S.values), outcome.meta, layout="sparse"))
                done += 50
            else:
                f = int(choice.dense_factor)
                dense = compress_matrix((shape, S.index, S.values), f)
                meta = dict(outcome.meta, compression=f)
                size = _write_atomically(path, lambda p: save_json_matrix(
                    p, dense, meta, layout="dense"))
                done += 50
        except Exception as exc:
            outcome.error = f"Writing {path.name} failed: {exc}"
            break
        outcome.written.append((Path(path), int(size)))
        progress(done, total)
    return outcome


def _size(n):
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:,} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024.0


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------

class ListModeOutputDialog(StepDialog):
    """Step 5. Next reads "Sort and Save" until the files are written,
    then "Finish"; changing an option after a save turns it back into
    "Sort and Save" for another write.

    Public for the controller and tests: ``choice()`` (the OutputChoice
    the widgets describe), ``sort_and_save()``, ``open_result()``,
    ``outcome`` (the last SortOutcome) and the widgets themselves.
    """

    def __init__(self, session, parent=None):
        super().__init__(session, TITLE, parent)
        self._theme = getattr(parent, "_theme", "light")
        self.outcome = None
        self._done = False
        self._loading = False              # restoring options: do not store them back
        self._run = run_with_progress      # a seam for the tests
        self._build()
        self.load_from_session()

    # ------------------------------------------------------------ building
    def _build(self):
        nchan = int(self.session.nchan)
        formats = QGroupBox(f"Files (asymmetric matrix, X = CH1, Y = CH0, "
                            f"{nchan} × {nchan} channels)")
        fl = QVBoxLayout(formats)
        self.mtx_check = QCheckBox(
            ".mtx – full resolution; integer cells, so each value is "
            "stochastically rounded (seeded; every cell keeps its expected value)")
        self.sparse_check = QCheckBox(
            "Sparse JSON – full resolution, exact values (read by SpectraTools)")
        self.mtx_check.setChecked(True)
        self.sparse_check.setChecked(True)
        fl.addWidget(self.mtx_check)
        fl.addWidget(self.sparse_check)
        dense_row = QHBoxLayout()
        self.dense_check = QCheckBox("Dense CoMPASS-style JSON, compressed by")
        self.factor_combo = QComboBox()
        factors = dense_factors(nchan)
        for f in factors:
            self.factor_combo.addItem(f"×{f}  ({nchan // f} × {nchan // f})", f)
        if PREFERRED_FACTOR in factors:
            self.factor_combo.setCurrentIndex(factors.index(PREFERRED_FACTOR))
        if not factors:
            self.dense_check.setEnabled(False)
            self.factor_combo.setEnabled(False)
            self.dense_check.setToolTip(
                f"No factor divides {nchan} into at most {DENSE_MAX} bins per axis.")
        dense_row.addWidget(self.dense_check)
        dense_row.addWidget(self.factor_combo)
        dense_row.addStretch(1)
        fl.addLayout(dense_row)
        self.body.addWidget(formats)

        where = QGroupBox("Where")
        form = QFormLayout(where)
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.browse_button)
        form.addRow("Folder:", folder_row)
        self.base_edit = QLineEdit()
        form.addRow("Base name:", self.base_edit)
        self.planned_label = QLabel()
        self.planned_label.setWordWrap(True)
        self.planned_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Will write:", self.planned_label)
        self.body.addWidget(where)

        self.summary_label = QLabel(
            "Nothing is sorted or written until you press Sort and Save.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.addWidget(self.summary_label, 1)

        open_row = QHBoxLayout()
        self.open_button = QPushButton("Open result")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_result)
        open_row.addWidget(self.open_button)
        open_row.addStretch(1)
        self.body.addLayout(open_row)

        for check in (self.mtx_check, self.sparse_check, self.dense_check):
            check.toggled.connect(self._options_changed)
        self.factor_combo.currentIndexChanged.connect(self._options_changed)
        self.folder_edit.textChanged.connect(self._options_changed)
        self.base_edit.textChanged.connect(self._options_changed)
        self.next_button.setText("Sort and Save")

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Output folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    # ------------------------------------------------------------ options
    def choice(self):
        dense = None
        if self.dense_check.isChecked() and self.factor_combo.count():
            dense = int(self.factor_combo.currentData())
        return OutputChoice(Path(self.folder_edit.text().strip()),
                            self.base_edit.text().strip(),
                            self.mtx_check.isChecked(), self.sparse_check.isChecked(),
                            dense)

    def _options_changed(self, *_):
        self.factor_combo.setEnabled(self.dense_check.isChecked())
        files = self.choice().files()
        self.planned_label.setText(
            "\n".join(str(p) for _k, p in files) if files else "(nothing ticked)")
        if not self._loading:
            self._store_options()
        if self._done:
            self._done = False
            self.next_button.setText("Sort and Save")

    def _store_options(self):
        """Keep the options on the session as they are edited, so Back
        and a later visit restore them (Back never calls
        apply_to_session)."""
        self.session.output_options = {
            "source_path": self.session.path,
            "folder": self.folder_edit.text(),
            "base": self.base_edit.text(),
            "mtx": self.mtx_check.isChecked(),
            "sparse": self.sparse_check.isChecked(),
            "dense": self.dense_check.isChecked(),
            "factor": (int(self.factor_combo.currentData())
                       if self.factor_combo.count() else None),
        }

    def _restore_options(self, options):
        self.folder_edit.setText(str(options.get("folder", "")))
        self.base_edit.setText(str(options.get("base", "")))
        self.mtx_check.setChecked(bool(options.get("mtx", True)))
        self.sparse_check.setChecked(bool(options.get("sparse", True)))
        self.dense_check.setChecked(bool(options.get("dense", False))
                                    and self.dense_check.isEnabled())
        index = self.factor_combo.findData(options.get("factor"))
        if index >= 0:
            self.factor_combo.setCurrentIndex(index)

    def _problem(self, choice):
        if not choice.files():
            return "Tick at least one file to write."
        if not choice.base:
            return "Give a base name for the files."
        if _BAD_NAME.search(choice.base):
            return ("The base name may not contain any of < > : \" / \\ | ? * "
                    "(it names files in the folder above).")
        if not str(choice.folder) or not Path(choice.folder).is_dir():
            return f"The folder {choice.folder} does not exist."
        return None

    # ------------------------------------------------------------ sorting
    def next_pressed(self):
        """The first press sorts and saves and stays on this step; only
        after a successful save does Next ("Finish") end the operation."""
        if self._done:
            return True
        self.sort_and_save()
        return False

    def sort_and_save(self):
        """Sort, subtract and write. Returns True when every ticked file
        was written."""
        choice = self.choice()
        problem = self._problem(choice)
        if problem:
            QMessageBox.warning(self, self.windowTitle(), problem)
            return False
        existing = [p for _k, p in choice.files() if p.exists()]
        if existing:
            answer = QMessageBox.question(
                self, self.windowTitle(),
                "These files already exist and will be replaced:\n\n"
                + "\n".join(str(p) for p in existing) + "\n\nReplace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return False
        outcome, error = self._run(
            self, self.windowTitle(), "Sorting and writing the matrices…",
            lambda progress: sort_and_write(self.session, choice, progress))
        if error is not None:
            self.summary_label.setText(f"Nothing was written: {error}")
            QMessageBox.warning(self, self.windowTitle(), error)
            return False
        self._record(outcome)
        if outcome.error:
            QMessageBox.warning(self, self.windowTitle(), outcome.error)
            return False
        self._done = True
        self.next_button.setText("Finish")
        return True

    def _record(self, outcome):
        s = self.session
        self.outcome = outcome
        s.prompt_matrix = outcome.prompt.matrix
        s.random_matrix = outcome.random.matrix
        s.result_matrix = outcome.result
        s.dropped = outcome.dropped
        for path, _size_ in outcome.written:
            if str(path) not in s.written:
                s.written.append(str(path))
        self.open_button.setEnabled(True)
        self.summary_label.setText(self.summary_text(outcome))

    @staticmethod
    def summary_text(o):
        p_total = int(o.prompt.matrix.total())
        r_total = int(o.random.matrix.total())
        lines = [
            f"Prompt: {o.n_prompt_pairs:,} pairs in the window, {p_total:,} sorted "
            f"({o.prompt.dropped:,} dropped off the matrix edge).",
            f"Random: {o.n_random_pairs:,} pairs in the windows, {r_total:,} sorted "
            f"({o.random.dropped:,} dropped off the matrix edge).",
            f"w = {o.w:.6g}.   Net = P − w R = {p_total:,} − {o.w:.6g} "
            f"× {r_total:,} = {float(o.result.total()):,.2f}.",
            f"Dropped off the edge in total: {o.dropped:,}.   Seed: {o.seed}.",
        ]
        if o.written:
            lines.append("Written:")
            lines += [f"    {p}  ({_size(n)})" for p, n in o.written]
        if o.error:
            lines.append(o.error)
        return "\n".join(lines)

    # ------------------------------------------------------------ opening
    def _compression_offer(self):
        choice = self.choice()
        return choice.dense_factor if choice.dense_factor not in (None, 1) else None

    def _ask_open(self, full_bytes, factor):
        """'full', 'compressed' or None (cancel)."""
        box = QMessageBox(self)
        box.setWindowTitle("Open result")
        box.setIcon(QMessageBox.Icon.Warning)
        n = int(self.session.nchan)
        text = (f"The full-resolution result is {n:,} × {n:,} cells; as a "
                f"dense float64 matrix it needs {_size(full_bytes)} of memory.")
        compressed = None
        if factor:
            m = n // factor
            text += (f"\n\nThe ×{factor} compressed version ({m:,} × {m:,}) "
                     f"needs {_size(m * m * 8)}.")
            compressed = box.addButton(f"Open ×{factor} compressed",
                                       QMessageBox.ButtonRole.AcceptRole)
        full = box.addButton("Open full resolution", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(compressed or full)
        box.setText(text)
        box.exec()
        clicked = box.clickedButton()
        if clicked is full:
            return "full"
        if compressed is not None and clicked is compressed:
            return "compressed"
        return None

    def open_result(self):
        """Hand S to the main window as a dense float64 [y][x] matrix,
        asking first when it is big. Returns what was opened ('full' /
        'compressed') or None."""
        s = self.session
        S = s.result_matrix
        if S is None:
            return None
        if s.open_matrix is None:
            QMessageBox.warning(self, "Open result",
                                "No main window to open the matrix in.")
            return None
        n = int(S.shape[0]) * int(S.shape[1])
        full_bytes = n * 8
        factor = self._compression_offer()
        how = "full"
        if full_bytes >= OPEN_WARN_BYTES:
            how = self._ask_open(full_bytes, factor)
            if how is None:
                return None
        if how == "compressed":
            m = (int(S.shape[0]) // factor) * (int(S.shape[1]) // factor)
            needed = m * 8 + OPEN_OVERHEAD_BYTES
        else:
            needed = full_bytes + OPEN_OVERHEAD_BYTES
        problem = system_memory.shortfall_message(
            "Opening the result in a Matrix panel", needed)
        if problem:
            QMessageBox.warning(self, "Open result", problem)
            return None
        name = self.choice().base or default_base_name(s)
        try:
            if how == "compressed":
                matrix = compress_matrix((S.shape, S.index, S.values), factor)
                matrix = matrix.astype(np.float64, copy=False)
                axis = axis_for_compression(factor)
                s.open_matrix(f"{name} (×{factor})", matrix, (axis, axis))
            else:
                matrix = to_dense(S, dtype=np.float64)
                s.open_matrix(name, matrix, (None, None))
        except MemoryError:
            QMessageBox.warning(self, "Open result",
                                "Not enough memory to open the matrix.")
            return None
        return how

    # ------------------------------------------------------------ session
    def load_from_session(self):
        s = self.session
        options = s.output_options
        self._loading = True
        try:
            if options is not None and options.get("source_path") == s.path:
                self._restore_options(options)
            else:
                self.folder_edit.setText(str(default_folder(s)))
                self.base_edit.setText(default_base_name(s))
        finally:
            self._loading = False
        self._options_changed()
        self.open_button.setEnabled(s.result_matrix is not None)
        if s.result_matrix is not None and s.written:
            self.summary_label.setText(
                "Already written:\n" + "\n".join(f"    {p}" for p in s.written))

    def apply_to_session(self):
        # Reached only after a successful save (see next_pressed); the
        # session was filled by _record.
        if not self._done:
            return "Sort and save first."
        return None
