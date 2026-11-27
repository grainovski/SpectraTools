"""Controller of the Sort List-Mode gamma-gamma operation.

Runs the five step dialogs (input, singles, gain matching, time windows,
sort and save) one after another around a single ListModeSession, so
that Back shows every earlier choice again and Cancel at any step simply
drops the session: nothing reaches disk before the last step.

The only computation done here is the one between step 1 and step 2:
sorting the singles and forming the pairs. It belongs to neither dialog
(the input dialog only describes the file, the singles dialog only shows
the result) and it is the one result that depends on everything step 1
chooses, so the controller is the natural place to decide when it is
stale. A pure Back/Next must not re-read the file; a changed file, flag
selection, detector assignment or pairing range must.

The step dialogs of steps 3-5 live in modules written separately and
are imported only when their step is reached, so this module (and its
tests, which use stub steps) does not depend on them.
"""

from PySide6.QtWidgets import QMessageBox

from listmode_session import (
    BACK, CANCEL, NEXT, ListModeSession, run_with_progress,
)

TITLE = "Sort List-Mode γγ Data"

#: Index (0-based) of the input step in the step list; the singles and
#: pairs are sorted when the user leaves this step with Next.
INPUT_STEP = 0


# ----------------------------------------------------------------------
# Default steps. Each is a factory (session, parent) -> StepDialog,
# created fresh for every visit.
# ----------------------------------------------------------------------

def _singles_step(session, parent):
    from listmode_singles_dialog import ListModeSinglesDialog
    return ListModeSinglesDialog(session, parent)


def _gain_step(session, parent):
    from gain_match_dialog import GainMatchDialog
    return GainMatchDialog(session, parent)


def _time_window_step(session, parent):
    from time_window_dialog import TimeWindowDialog
    return TimeWindowDialog(session, parent)


def _output_step(session, parent):
    from listmode_output_dialog import ListModeOutputDialog
    return ListModeOutputDialog(session, parent)


def default_steps(initial_path=None, open_path=None):
    """The five step factories, in order. The input step is a closure so
    it can be given the folder to start the file picker in and, for
    File > Open of an SData file, a file to open on its first visit only
    (a later visit shows what the session holds, which may be another
    file the user chose since)."""
    pending = [open_path] if open_path else []

    def _input_step(session, parent):
        from listmode_input_dialog import ListModeInputDialog
        return ListModeInputDialog(
            session, parent, initial_path=initial_path or open_path,
            open_path=pending.pop() if pending else None)

    return [_input_step, _singles_step, _gain_step, _time_window_step,
            _output_step]


# ----------------------------------------------------------------------
# The singles + pairs job
# ----------------------------------------------------------------------

def input_key(session):
    """Everything step 1 chooses that the singles and pairs depend on.
    Two equal keys mean a sort done for one is valid for the other."""
    return (session.path, session.exclude_values, session.x_channel,
            session.y_channel, int(session.max_dt_ps), int(session.nchan))


def _counted(chunks, progress, done, total):
    """Pass chunks through, reporting entries read so far. Progress is
    counted in file entries (before exclusion), which is what n_entries
    is and so what the bar can honestly be a fraction of."""
    for chunk in chunks:
        done += len(chunk.channel)
        progress(done, total)
        yield chunk


def sort_singles_and_pairs(session, progress=lambda done, total: None):
    """(Singles, Pairs) for the session's step-1 choices.

    Two passes over the file rather than one shared pass: the sorter's
    two functions each take an iterable of chunks, and re-reading a few
    hundred thousand events costs well under a second, far less than
    teeing chunks between generators would cost in clarity. Runs on the
    worker thread, so it only reads the session; the caller stores the
    results on the main thread.
    """
    import listmode_io
    import listmode_sort

    source = session.listmode if session.listmode is not None else session.path
    if source is None:
        raise listmode_sort.SortError("No list-mode file has been chosen.")
    n = session.listmode.n_entries if session.listmode is not None else 0
    total = max(1, 2 * n)
    exclude = session.exclude_values
    channels = tuple(sorted({int(session.x_channel), int(session.y_channel)}))
    singles = listmode_sort.sort_singles(
        _counted(listmode_io.iter_events(source), progress, 0, total),
        channels=channels, nchan=session.nchan, exclude_flags=exclude)
    pairs = listmode_sort.pair_events(
        _counted(listmode_io.iter_events(source), progress, n, total),
        session.max_dt_ps, x_channel=session.x_channel,
        y_channel=session.y_channel, exclude_flags=exclude)
    return singles, pairs


def _run_sort_job(session, parent):
    """Sort singles and pairs behind a progress dialog. Returns None on
    success or the error message (already shown to the user)."""
    result, error = run_with_progress(
        parent, TITLE, "Sorting singles and pairing events…",
        lambda progress: sort_singles_and_pairs(session, progress))
    if error is not None:
        QMessageBox.warning(parent, TITLE,
                            f"Could not sort the list-mode data:\n\n{error}")
        return error
    session.singles, session.pairs = result
    return None


# ----------------------------------------------------------------------
# The controller
# ----------------------------------------------------------------------

def _show_step(factory, session, parent):
    """Create one step dialog, run it and dispose of it; return its
    step_result. A fresh dialog per visit is what makes
    load_from_session the single source of what a step shows."""
    dialog = factory(session, parent)
    try:
        dialog.load_from_session()
        dialog.exec()
        return dialog.step_result
    finally:
        dialog.deleteLater()


def run_listmode_operation(parent, send_spectra=None, open_matrix=None,
                           initial_path=None, steps=None, open_path=None,
                           session=None):
    """Run the whole operation; return the finished session, or None if
    the user cancelled (or a step failed, which is reported first).

    `send_spectra` and `open_matrix` are the main window's hand-offs,
    stored on the session for the steps that use them. `steps` replaces
    the step factories (tests use stubs); by default they are the five
    dialogs of the plan, with the input step starting its file picker at
    `initial_path` and, when `open_path` is given, opening that file
    straight away. `session` lets the caller supply the session to run
    in (the main window does, so its hand-offs can read the run's path);
    the hand-offs given here are then stored on it.
    """
    if session is None:
        session = ListModeSession()
    if send_spectra is not None:
        session.send_spectra = send_spectra
    if open_matrix is not None:
        session.open_matrix = open_matrix
    if steps is None:
        steps = default_steps(initial_path, open_path)
    steps = list(steps)
    index = 0
    try:
        while 0 <= index < len(steps):
            before = input_key(session) if index == INPUT_STEP else None
            result = _show_step(steps[index], session, parent)
            if result == CANCEL:
                return None
            if result == BACK:
                index = max(0, index - 1)
                continue
            if result != NEXT:  # anything unexpected counts as Cancel
                return None
            if index == INPUT_STEP:
                if input_key(session) != before:
                    session.invalidate_from(1)
                if session.singles is None or session.pairs is None:
                    if _run_sort_job(session, parent) is not None:
                        continue  # stay on step 1
            index += 1
    except Exception as exc:  # a slot must never swallow it
        QMessageBox.critical(parent, TITLE,
                             f"{type(exc).__name__}: {exc}")
        return None
    return session
