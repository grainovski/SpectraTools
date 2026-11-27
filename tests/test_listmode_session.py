"""The shared contract of the list-mode operation's dialogs."""

from PySide6.QtWidgets import QLabel

from listmode_session import (
    BACK, CANCEL, NEXT, ListModeSession, StepDialog, run_with_progress,
)


def test_exclude_values_is_a_value_set_not_a_mask():
    s = ListModeSession(exclude_flags={0x80, 0x480, 0xC80, 0x8000, 0x8800})
    assert s.exclude_values == frozenset({0x80, 0x480, 0xC80, 0x8000, 0x8800})
    assert 0x800 not in s.exclude_values


def test_invalidate_from_input_clears_everything_downstream():
    s = ListModeSession(singles="s", pairs="p", gain="g", gain_skipped=True,
                        weight=0.5, result_matrix="m", written=["a"])
    s.invalidate_from(1)
    assert (s.singles, s.pairs, s.gain, s.gain_skipped) == (None, None, None, False)
    assert s.weight is None and s.result_matrix is None and s.written == []


def test_invalidate_from_windows_keeps_gain_and_pairs():
    s = ListModeSession(singles="s", pairs="p", gain="g", weight=0.5,
                        prompt_matrix="P")
    s.invalidate_from(4)
    assert (s.singles, s.pairs, s.gain) == ("s", "p", "g")
    assert s.weight is None and s.prompt_matrix is None


class _Step(StepDialog):
    def __init__(self, session, problem=None):
        super().__init__(session, "Step", first=False)
        self.body.addWidget(QLabel("content"))
        self.problem = problem

    def apply_to_session(self):
        if self.problem:
            return self.problem
        self.session.max_dt_ps = 123
        return None


def test_step_dialog_next_back_cancel(qapp, monkeypatch):
    s = ListModeSession()
    d = _Step(s)
    d.next_button.click()
    assert d.step_result == NEXT and s.max_dt_ps == 123

    d2 = _Step(s)
    d2.back_button.click()
    assert d2.step_result == BACK

    d3 = _Step(s)
    d3.cancel_button.click()
    assert d3.step_result == CANCEL


def test_step_dialog_stays_open_on_a_problem(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr("listmode_session.QMessageBox.warning",
                        lambda *a: shown.append(a[2]))
    d = _Step(ListModeSession(), problem="Pick a file first.")
    d.next_button.click()
    assert shown == ["Pick a file first."]
    assert d.step_result == CANCEL  # not NEXT: still open


def test_run_with_progress_returns_result_or_error(qapp):
    def work(progress):
        for i in range(3):
            progress(i + 1, 3)
        return 42

    assert run_with_progress(None, "t", "working", work) == (42, None)

    def boom(progress):
        raise ValueError("bad window")

    result, error = run_with_progress(None, "t", "working", boom)
    assert result is None and error == "bad window"
