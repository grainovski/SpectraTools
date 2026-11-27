"""The list-mode operation's controller: step order, Back/Next/Cancel,
and when the singles + pairs job runs."""

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

import listmode_operation
from listmode_session import BACK, CANCEL, NEXT, StepDialog
from synthetic_sdata import simple_events, write_sdata

REAL_RUN = (Path(__file__).resolve().parent.parent / "Na22_261126_ANI_STEF_180"
            / "UNFILTERED" / "SData_Na22_261126_ANI_STEF.root")


class Script:
    """Answers for the stub steps, in visit order, and a log of visits."""

    def __init__(self, answers, on_input=None):
        self.answers = list(answers)
        self.visits = []
        self.on_input = on_input  # called at each visit of step 0

    def factories(self, n=5):
        return [self._factory(i) for i in range(n)]

    def _factory(self, index):
        script = self

        class Stub(StepDialog):
            def __init__(self, session, parent=None):
                super().__init__(session, f"Step {index}", parent,
                                 first=index == 0)
                self.body.addWidget(QLabel("stub"))

            def exec(self):
                script.visits.append(index)
                if index == 0 and script.on_input is not None:
                    script.on_input(self.session, len(script.visits))
                self.step_result = script.answers.pop(0)
                return 0

        return Stub


@pytest.fixture
def fake_job(monkeypatch):
    """Replace the progress-dialog job by a counter; `errors` holds
    messages to return on the next calls, in order."""
    state = {"calls": 0, "errors": []}

    def fake(parent, title, label, fn):
        state["calls"] += 1
        if state["errors"]:
            return None, state["errors"].pop(0)
        return ("singles", "pairs"), None

    monkeypatch.setattr(listmode_operation, "run_with_progress", fake)
    warnings = []
    monkeypatch.setattr(listmode_operation.QMessageBox, "warning",
                        lambda *a: warnings.append(a[2]))
    state["warnings"] = warnings
    return state


def test_next_through_all_steps_returns_the_session(qapp, fake_job):
    script = Script([NEXT] * 5)
    sent = object()
    session = listmode_operation.run_listmode_operation(
        None, send_spectra=sent, steps=script.factories())
    assert script.visits == [0, 1, 2, 3, 4]
    assert session is not None and session.send_spectra is sent
    assert (session.singles, session.pairs) == ("singles", "pairs")
    assert fake_job["calls"] == 1


def test_cancel_returns_none(qapp, fake_job):
    script = Script([NEXT, NEXT, CANCEL])
    assert listmode_operation.run_listmode_operation(
        None, steps=script.factories()) is None
    assert script.visits == [0, 1, 2]


def test_back_goes_back_and_pure_back_next_does_not_resort(qapp, fake_job):
    script = Script([NEXT, NEXT, BACK, BACK, NEXT, NEXT, NEXT, NEXT, NEXT])
    session = listmode_operation.run_listmode_operation(
        None, steps=script.factories())
    assert script.visits == [0, 1, 2, 1, 0, 1, 2, 3, 4]
    assert session is not None
    assert fake_job["calls"] == 1  # input unchanged: no second sort


def test_changed_input_invalidates_and_resorts(qapp, fake_job):
    def on_input(session, visit):
        if visit > 1:  # the user changes the pairing range on return
            session.max_dt_ps = 1_000_000

    script = Script([NEXT, NEXT, BACK, BACK, NEXT, NEXT, NEXT, NEXT, NEXT],
                    on_input=on_input)
    # Mark the later results so the invalidation is visible.
    original = listmode_operation._show_step

    def show(factory, session, parent):
        result = original(factory, session, parent)
        if len(script.visits) == 2:
            session.gain = "gain"
        return result

    listmode_operation._show_step = show
    try:
        session = listmode_operation.run_listmode_operation(
            None, steps=script.factories())
    finally:
        listmode_operation._show_step = original
    assert fake_job["calls"] == 2
    assert session.max_dt_ps == 1_000_000
    assert session.gain is None  # cleared by invalidate_from(1)


@pytest.mark.parametrize("field,value", [
    ("path", "other.root"), ("exclude_flags", {0x800}),
    ("x_channel", 2), ("y_channel", 3)])
def test_each_input_field_counts_as_a_change(qapp, fake_job, field, value):
    def on_input(session, visit):
        if visit > 1:
            setattr(session, field, value)

    script = Script([NEXT, BACK, NEXT, NEXT, NEXT, NEXT, NEXT],
                    on_input=on_input)
    listmode_operation.run_listmode_operation(None, steps=script.factories())
    assert fake_job["calls"] == 2


def test_job_error_keeps_you_on_step_one(qapp, fake_job):
    fake_job["errors"] = ["bad file"]
    script = Script([NEXT, NEXT, NEXT, NEXT, NEXT, NEXT])
    session = listmode_operation.run_listmode_operation(
        None, steps=script.factories())
    assert script.visits == [0, 0, 1, 2, 3, 4]
    assert fake_job["calls"] == 2
    assert len(fake_job["warnings"]) == 1 and "bad file" in fake_job["warnings"][0]
    assert session.singles == "singles"


def test_a_failing_step_is_reported_not_raised(qapp, fake_job, monkeypatch):
    shown = []
    monkeypatch.setattr(listmode_operation.QMessageBox, "critical",
                        lambda *a: shown.append(a[2]))

    def broken(session, parent):
        raise RuntimeError("no such dialog")

    script = Script([NEXT])
    steps = script.factories(1) + [broken]
    assert listmode_operation.run_listmode_operation(None, steps=steps) is None
    assert shown and "no such dialog" in shown[0]


def test_default_steps_are_five_lazy_factories():
    steps = listmode_operation.default_steps("somewhere")
    assert len(steps) == 5 and all(callable(f) for f in steps)


def _synthetic_session(tmp_path):
    import listmode_io
    ch, ts, e, fl = simple_events(3000, seed=4)
    path = write_sdata(str(tmp_path / "SData_synth.root"), ch, ts, e, fl,
                       chunks=3)
    lm = listmode_io.open_listmode(path)
    from listmode_session import ListModeSession
    s = ListModeSession(path=path, listmode=lm,
                        exclude_flags=listmode_io.default_excluded_flags(
                            lm.flag_counts),
                        max_dt_ps=200_000)
    return s, (ch, ts, e, fl)


def test_sort_job_matches_the_sorter_and_reports_progress(qapp, tmp_path):
    import listmode_io
    import listmode_sort
    s, (ch, ts, e, fl) = _synthetic_session(tmp_path)
    calls = []
    singles, pairs = listmode_operation.sort_singles_and_pairs(
        s, lambda d, t: calls.append((d, t)))
    keep = fl != 0x8000
    for c in (0, 1):
        want = np.bincount(e[keep & (ch == c)].astype(int), minlength=16384)
        assert np.array_equal(singles[c], want)
    assert singles.excluded == int(np.count_nonzero(~keep))
    ref = listmode_sort.pair_events(listmode_io.iter_events(s.path), 200_000,
                                    exclude_flags=frozenset({0x8000}))
    assert len(pairs.dt) == len(ref.dt) > 0
    assert calls[-1] == (2 * 3000, 2 * 3000)


def test_real_progress_job_runs_headless(qapp, tmp_path, monkeypatch):
    s, _ = _synthetic_session(tmp_path)
    err = listmode_operation._run_sort_job(s, None)
    assert err is None and s.singles is not None and s.pairs is not None


@pytest.mark.skipif(not REAL_RUN.is_file(),
                    reason="Na22_261126_ANI_STEF_180 run not present")
def test_real_run_through_input_and_singles(qapp, monkeypatch):
    """The real Na-22 UNFILTERED run, through the real input dialog and the
    controller's job, stopped at step 2."""
    from listmode_input_dialog import ListModeInputDialog

    seen = {}

    def input_step(session, parent):
        d = ListModeInputDialog(session, parent)

        def exec_():
            assert d.open_path(str(REAL_RUN))
            seen["entries"] = d.info_labels["entries"].text()
            seen["granularity"] = d.info_labels["granularity"].text()
            d.next_button.click()
            return 0

        d.exec = exec_
        return d

    class Stop(StepDialog):
        def __init__(self, session, parent=None):
            super().__init__(session, "stop", parent)
            seen["session"] = session

        def exec(self):
            self.step_result = CANCEL
            return 0

    result = listmode_operation.run_listmode_operation(
        None, steps=[input_step, Stop])
    assert result is None  # cancelled at step 2
    s = seen["session"]
    assert seen["entries"] == "655,336"
    assert seen["granularity"].startswith("10 ns")
    assert set(s.listmode.flag_counts) == {0x0, 0x800}
    assert s.exclude_values == frozenset()
    assert int(s.singles[0].sum()) == 328_431
    assert int(s.singles[1].sum()) == 326_905
    assert s.max_dt_ps == 2_000_000
    assert len(s.pairs.dt) == 43_962
