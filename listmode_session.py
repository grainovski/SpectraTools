"""Shared state and plumbing for the Sort List-Mode gamma-gamma operation.

The operation is a chain of five dialogs (input, singles, gain matching,
time windows, sort and save) run one after another by
listmode_operation. This module is the contract between them, so that
each dialog can be written and tested on its own:

* ``ListModeSession`` holds every choice and every intermediate result.
  A dialog reads its starting state from the session in
  ``load_from_session`` and writes the user's choices back in
  ``apply_to_session``. Going Back therefore shows what was chosen
  before, and nothing reaches disk until the last step.
* ``StepDialog`` is the common base: Back / Next / Cancel buttons and the
  ``step_result`` the controller reads after ``exec()``.
* ``run_with_progress`` runs slow work (reading, pairing, sorting,
  writing) on a worker thread behind a modal progress dialog, the same
  shape as MainWindow._load_matrix_with_progress, and turns any
  exception into an error message instead of letting it vanish inside a
  Qt slot (reference: Qt silent failures).

Nothing here knows about the main window. The controller is given two
callables for that, stored on the session: ``send_spectra`` (add 1-D
spectra to the main window) and ``open_matrix`` (open a matrix panel).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QMessageBox, QProgressDialog, QPushButton,
    QVBoxLayout,
)


# What a StepDialog reports after exec().
NEXT = "next"
BACK = "back"
CANCEL = "cancel"

DEFAULT_MAX_DT_PS = 2_000_000  # +-2 us, the plan's default pairing range


@dataclass
class ListModeSession:
    """Everything the five steps choose or compute, in step order.

    Fields are filled in as the user advances; a field that a later step
    needs and an earlier step has not set yet is None. When an earlier
    step changes something a later result depends on, the controller
    calls ``invalidate_from(step)`` so stale results are never reused.
    """

    # Hand-offs to the main window (set by the controller; may be None
    # in tests). send_spectra(list of (name, numpy array)) and
    # open_matrix(name, matrix, axes).
    send_spectra: Optional[Callable] = None
    open_matrix: Optional[Callable] = None

    # Step 1: input.
    path: Optional[str] = None
    listmode: Any = None            # listmode_io.ListModeFile
    run_info: Any = None            # listmode_io.RunInfo or None
    exclude_flags: set = field(default_factory=set)  # flag VALUES to drop
    x_channel: int = 1              # X = CH1 (CoMPASS orientation)
    y_channel: int = 0              # Y = CH0, the gain reference
    max_dt_ps: int = DEFAULT_MAX_DT_PS
    nchan: int = 16384

    # Computed after step 1 by the controller (one pass over the file).
    singles: Any = None             # listmode_sort.Singles
    pairs: Any = None               # listmode_sort.Pairs

    # Step 3: gain matching. None = not matched yet; the dialog may also
    # store an identity choice explicitly (gain_skipped = True).
    gain: Any = None                # gain_match.GainMatch
    gain_skipped: bool = False

    # Step 4: time windows, in ps, inclusive; widths and weight from
    # listmode_sort.window_weight.
    prompt_window: Optional[tuple] = None
    random_windows: list = field(default_factory=list)
    weight: Optional[float] = None
    n_prompt_bins: Optional[int] = None
    n_random_bins: Optional[int] = None

    # Step 5: results.
    prompt_matrix: Any = None       # listmode_sort.SparseMatrix
    random_matrix: Any = None
    result_matrix: Any = None       # float S = P - w R
    dropped: int = 0                # pairs mapped off the matrix edge
    written: list = field(default_factory=list)  # paths written

    # Step 5: the output options the user chose (folder, base name,
    # formats, dense factor), so that Back and a later visit show them
    # again. A dict written by the output dialog whenever an option
    # changes; it records the input path it was made for, and the dialog
    # falls back to the defaults for another file (a base name made from
    # one run's ID must not be offered for the next run). None = defaults.
    output_options: Optional[dict] = None

    @property
    def granularity_ps(self):
        return None if self.listmode is None else self.listmode.granularity_ps

    @property
    def exclude_values(self):
        """The excluded flag VALUES as a frozenset, which is what
        listmode_sort's ``exclude_flags`` should be given. Never OR them
        into a bit mask: the default excluded values contain 0xC80, whose
        0x800 bit would then drop every good 0x800 event as well."""
        return frozenset(int(v) for v in self.exclude_flags)

    def invalidate_from(self, step):
        """Forget results computed at or after `step` (1-5).

        Changing the input (step 1) makes the singles, the pairs and a
        gain match fitted to those singles stale. A gain loaded from a
        sidecar or fitted to a separate calibration run would survive in
        principle, but one rule is easier to trust than two: the user can
        reload the sidecar in one click.
        """
        if step <= 1:
            self.singles = None
            self.pairs = None
        if step <= 2:
            self.gain = None
            self.gain_skipped = False
        if step <= 4:
            self.weight = None
            self.n_prompt_bins = None
            self.n_random_bins = None
        if step <= 5:
            self.prompt_matrix = None
            self.random_matrix = None
            self.result_matrix = None
            self.dropped = 0
            self.written = []


class StepDialog(QDialog):
    """Base for the five steps: Back / Next / Cancel and step_result.

    Subclasses build their content into ``self.body`` (a QVBoxLayout),
    implement ``load_from_session`` and ``apply_to_session``, and may
    rename the Next button (the last step says "Sort and Save"). Next
    first asks ``next_pressed``, which may keep the dialog open (step 5
    sorts and saves on its first press and only moves on at the second),
    then calls ``apply_to_session``; if that returns an error message the
    dialog stays open and shows it, otherwise the dialog closes with NEXT.
    An exception from either hook is shown, never swallowed.
    """

    def __init__(self, session, title, parent=None, first=False):
        super().__init__(parent)
        self.session = session
        self.step_result = CANCEL
        self.setWindowTitle(title)
        self.setModal(True)
        outer = QVBoxLayout(self)
        self.body = QVBoxLayout()
        outer.addLayout(self.body, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.back_button = QPushButton("< Back")
        self.back_button.setEnabled(not first)
        self.next_button = QPushButton("Next >")
        self.next_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)
        buttons.addWidget(self.cancel_button)
        outer.addLayout(buttons)
        self.back_button.clicked.connect(self._on_back)
        self.next_button.clicked.connect(self._on_next)
        self.cancel_button.clicked.connect(self.reject)

    # Subclass hooks ------------------------------------------------------
    def load_from_session(self):
        """Show the session's current state."""

    def apply_to_session(self):
        """Write the user's choices to the session. Return None on
        success or a message saying what is wrong."""
        return None

    def next_pressed(self):
        """Called first when Next is pressed. Return True to go on
        (apply_to_session, then close with NEXT) or False to stay on this
        step, having done whatever the press means here."""
        return True

    # Buttons -------------------------------------------------------------
    def _on_back(self):
        self.step_result = BACK
        self.done(QDialog.DialogCode.Rejected)

    def _on_next(self):
        try:
            if not self.next_pressed():
                return
            problem = self.apply_to_session()
        except Exception as exc:  # never let a slot swallow it
            problem = f"{type(exc).__name__}: {exc}"
        if problem:
            QMessageBox.warning(self, self.windowTitle(), problem)
            return
        self.step_result = NEXT
        self.done(QDialog.DialogCode.Accepted)

    def reject(self):
        # Escape and the window's close button arrive here too.
        if self.step_result not in (BACK, NEXT):
            self.step_result = CANCEL
        super().reject()


class _Worker(QThread):
    progressed = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        # Nothing may escape run(): an exception here would be printed to
        # stderr and the dialog would wait forever.
        try:
            result = self._fn(lambda done, total: self.progressed.emit(int(done), int(total)))
        except MemoryError:
            self.failed.emit("Not enough memory for this step.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}" if not str(exc) else str(exc))
        else:
            self.succeeded.emit(result)


def run_with_progress(parent, title, label, fn):
    """Run ``fn(progress)`` on a worker thread behind a modal progress
    dialog and return ``(result, error_message)``; exactly one is None.

    ``fn`` receives ``progress(done, total)``, which it may call from the
    worker thread as often as it likes (it only emits a signal). With no
    calls the bar stays at 0, which is honest for work of unknown length.
    There is no Cancel: the sorting and writing steps have no unwind
    path, and a half-written file would be worse than waiting.
    """
    dialog = QProgressDialog(label, None, 0, 100, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)
    outcome = {"result": None, "error": None}
    worker = _Worker(fn, parent)

    def on_progress(done, total):
        if total > 0:
            dialog.setValue(max(0, min(100, int(done * 100 / total))))

    worker.progressed.connect(on_progress)
    worker.succeeded.connect(lambda r: outcome.__setitem__("result", r))
    worker.failed.connect(lambda m: outcome.__setitem__("error", m))
    worker.finished.connect(dialog.close)
    worker.start()
    dialog.exec()
    worker.wait()
    worker.deleteLater()
    return outcome["result"], outcome["error"]
