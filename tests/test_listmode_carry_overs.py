"""The Task 5 review items carried into Task 6 of the v7 plan: step-5
options kept on the session, the StepDialog Next hook, step 4's warning
for empty random windows, listmode_io.num_entries / is_listmode_file,
the shared 1-D spectrum loader, GainPair.matcher_ok, the step-1 file
opened on arrival, and the step-1 max-|dt| note that used to clip."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

import gain_match as G
import gain_match_dialog as D
import listmode_io
import listmode_output_dialog as lod
import spectrum_files
import system_memory
from listmode_input_dialog import ListModeInputDialog
from listmode_operation import default_steps
from listmode_output_dialog import ListModeOutputDialog
from listmode_session import CANCEL, NEXT, ListModeSession, StepDialog
from listmode_sort import Pairs, SparseMatrix
from synthetic_sdata import simple_events, write_sdata
from time_window_dialog import RANDOM_WARN_MIN, TimeWindowDialog

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TICK = 10_000
MAX = 2_000_000


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr("listmode_session.QMessageBox.warning",
                        lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(lod.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2]))
    return shown


# ----------------------------------------------------------------------
# StepDialog's Next hook
# ----------------------------------------------------------------------

class _Hooked(StepDialog):
    def __init__(self, go_on):
        super().__init__(ListModeSession(), "hooked")
        self.body.addWidget(QLabel("x"))
        self.go_on = go_on
        self.pressed = 0
        self.applied = 0

    def next_pressed(self):
        self.pressed += 1
        if isinstance(self.go_on, Exception):
            raise self.go_on
        return self.go_on

    def apply_to_session(self):
        self.applied += 1
        return None


def test_next_hook_can_keep_the_step_open(qapp):
    d = _Hooked(False)
    d.next_button.click()
    assert (d.pressed, d.applied, d.step_result) == (1, 0, CANCEL)
    d.go_on = True
    d.next_button.click()
    assert (d.pressed, d.applied, d.step_result) == (2, 1, NEXT)


def test_an_exception_in_the_next_hook_is_shown(qapp, warnings):
    d = _Hooked(RuntimeError("hook failed"))
    d.next_button.click()
    assert warnings == ["RuntimeError: hook failed"]
    assert d.step_result == CANCEL and d.applied == 0


# ----------------------------------------------------------------------
# Step 5: options on the session; Next through the hook
# ----------------------------------------------------------------------

def _output_session(tmp_path, name="Run_A"):
    run = tmp_path / name
    (run / "UNFILTERED").mkdir(parents=True, exist_ok=True)
    return ListModeSession(path=str(run / "UNFILTERED" / f"SData_{name}.root"),
                           nchan=64)


def test_output_options_survive_a_back_visit(qapp, tmp_path):
    s = _output_session(tmp_path)
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    d.base_edit.setText("my_matrix")
    d.mtx_check.setChecked(False)
    d.dense_check.setChecked(True)
    d.factor_combo.setCurrentIndex(d.factor_combo.findData(8))
    d.back_button.click()                     # Back never applies

    again = ListModeOutputDialog(s)           # the controller's next visit
    assert again.folder_edit.text() == str(tmp_path)
    assert again.base_edit.text() == "my_matrix"
    assert not again.mtx_check.isChecked() and again.sparse_check.isChecked()
    assert again.dense_check.isChecked()
    assert again.factor_combo.currentData() == 8
    assert again.choice() == d.choice()


def test_output_options_of_another_run_are_not_reused(qapp, tmp_path):
    s = _output_session(tmp_path, "Run_A")
    d = ListModeOutputDialog(s)
    d.base_edit.setText("Run_A_custom")
    s.path = _output_session(tmp_path, "Run_B").path    # step 1: another file
    again = ListModeOutputDialog(s)
    assert again.base_edit.text() == "Run_B_gg_sub"
    assert Path(again.folder_edit.text()) == (tmp_path / "Run_B").resolve()


def test_first_next_sorts_and_stays_second_finishes(qapp, tmp_path):
    s = _output_session(tmp_path)
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    calls = []

    def fake_sort():
        calls.append(1)
        d._done = True
        d.next_button.setText("Finish")
        return True
    d.sort_and_save = fake_sort
    d.next_button.click()
    assert calls == [1] and d.step_result == CANCEL      # still on step 5
    d.next_button.click()
    assert calls == [1] and d.step_result == NEXT


def test_a_sort_that_raises_is_shown_not_swallowed(qapp, tmp_path, warnings):
    s = _output_session(tmp_path)
    d = ListModeOutputDialog(s)

    def boom():
        raise ValueError("sorter broke")
    d.sort_and_save = boom
    d.next_button.click()
    assert warnings == ["ValueError: sorter broke"] and d.step_result == CANCEL


def test_open_result_refused_when_memory_is_short(qapp, tmp_path, warnings, monkeypatch):
    s = _output_session(tmp_path)
    s.nchan = 16384
    s.result_matrix = SparseMatrix((16384, 16384), np.array([5]), np.array([-0.5]))
    opened = []
    s.open_matrix = lambda *a: opened.append(a)
    d = ListModeOutputDialog(s)
    d._ask_open = lambda nbytes, factor: "full"
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 2**30)
    assert d.open_result() is None and opened == []
    assert "2.1 GiB" in warnings[-1] and "only 1.0 GiB" in warnings[-1]
    # The x16 view fits and still opens.
    d._ask_open = lambda nbytes, factor: "compressed"
    d.dense_check.setChecked(True)
    d.factor_combo.setCurrentIndex(d.factor_combo.findData(16))
    assert d.open_result() == "compressed" and opened[-1][1].shape == (1024, 1024)


# ----------------------------------------------------------------------
# Step 4: random windows that hold (almost) nothing
# ----------------------------------------------------------------------

def _tw_session(n_random, stream="UNFILTERED"):
    rng = np.random.default_rng(1)
    prompt = rng.choice([-TICK, 0, TICK], size=3000)
    k = MAX // TICK
    rand = rng.integers(-k, k + 1, size=n_random) * TICK
    rand = rand[np.abs(rand) >= 600_000]           # all inside the side bands
    dt = np.concatenate([prompt, rand]).astype(np.int64)
    s = ListModeSession(max_dt_ps=MAX, nchan=64)
    s.listmode = SimpleNamespace(granularity_ps=TICK, stream=stream)
    s.pairs = Pairs(dt, np.zeros(len(dt), dtype=np.int64), np.zeros(len(dt), dtype=np.int64))
    return s, len(rand)


def test_empty_random_windows_warn_and_name_the_filtered_stream(qapp):
    s, n = _tw_session(0, stream="FILTERED")
    d = TimeWindowDialog(s)
    assert n == 0 and d.readout["random_total"] == 0
    text = d.readout["random_warning"]
    assert text and "no pairs at all" in text and "FILTERED" in text
    assert not d.warning_label.isHidden() and d.warning_label.text() == text
    assert d.next_button.isEnabled()                # a warning, not a refusal


def test_few_random_pairs_warn_without_blaming_the_stream(qapp):
    s, n = _tw_session(12)
    d = TimeWindowDialog(s)
    assert 0 < d.readout["random_total"] < RANDOM_WARN_MIN
    text = d.readout["random_warning"]
    assert f"only {d.readout['random_total']:,} pair" in text
    assert "FILTERED" not in text


def test_a_measured_random_level_does_not_warn(qapp):
    s, n = _tw_session(3000)
    d = TimeWindowDialog(s)
    assert d.readout["random_total"] >= RANDOM_WARN_MIN
    assert d.readout["random_warning"] is None and d.warning_label.isHidden()


# ----------------------------------------------------------------------
# listmode_io: the public entry points
# ----------------------------------------------------------------------

def test_num_entries_and_is_listmode_file(tmp_path):
    ch, ts, e, fl = simple_events(321)
    path = write_sdata(str(tmp_path / "SData_r.root"), ch, ts, e, fl, chunks=3)
    assert listmode_io.num_entries(path) == 321
    assert listmode_io.is_listmode_file(path)
    other = write_sdata(str(tmp_path / "x.root"), ch, ts, e, fl, tree="Events")
    assert not listmode_io.is_listmode_file(other)
    assert not listmode_io.is_listmode_file(str(FIXTURES / "eu.spe"))
    assert not listmode_io.is_listmode_file(str(tmp_path / "missing.root"))
    with pytest.raises(listmode_io.ParseError):
        listmode_io.num_entries(other)


def test_the_gain_dialog_no_longer_reaches_into_listmode_io_privates():
    source = Path(D.__file__).read_text(encoding="utf-8")
    assert "_open_tree" not in source and "num_entries(" in source


# ----------------------------------------------------------------------
# One 1-D loader for the main window and the gain dialog
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "eu.spe", "demo.spk", "test.txt", "316-2_160V_0785uA.n42",
    "18-082025-AlphaSpec-Mix-Source.lzs",
])
def test_one_loader_serves_both_callers(qapp, name):
    from main_window import MainWindow
    path = str(FIXTURES / name)
    data, calibration = spectrum_files.load_spectrum_file(path)
    assert len(data) > 0
    window = MainWindow()
    spectrum, error, cal = window._try_load_spectrum(path)
    assert error is None
    np.testing.assert_array_equal(spectrum.data, data)
    assert (cal is None) == (calibration is None)
    np.testing.assert_array_equal(D._load_1d(path), np.asarray(data, dtype=float))
    window.close()


def test_both_callers_import_the_same_function():
    import main_window
    assert main_window.load_spectrum_file is spectrum_files.load_spectrum_file
    assert D.load_spectrum_file is spectrum_files.load_spectrum_file


def test_a_file_that_fails_to_load_is_reported_not_raised(qapp, tmp_path, monkeypatch):
    """_try_load_spectrum used to return two values on failure while both
    callers unpack three, so a bad file raised ValueError instead of
    reaching the "could not be loaded" message."""
    from main_window import MainWindow
    bad = tmp_path / "bad.txt"
    bad.write_text("not a spectrum\n")
    shown = []
    monkeypatch.setattr("main_window.QMessageBox.warning",
                        lambda parent, title, text, *a, **k: shown.append((title, text)))
    window = MainWindow()
    spectrum, error, calibration = window._try_load_spectrum(str(bad))
    assert spectrum is None and calibration is None and "bad.txt" in error
    window._load_files([str(bad)])
    assert shown and shown[0][0] == "Some files could not be loaded"
    window.close()


# ----------------------------------------------------------------------
# GainPair.matcher_ok
# ----------------------------------------------------------------------

def _suspect_pairs():
    """Four lines on a straight gain; the third one the matcher doubted."""
    pairs = []
    for c in (2000.0, 4000.0, 6000.0, 8000.0):
        ok = c != 6000.0
        pairs.append(G.GainPair(c, 0.2, 1.01 * c + 5.0, 0.2, fwhm1=40.0,
                                fwhm0=40.0, included=ok, matcher_ok=ok))
    return pairs


def test_matcher_ok_round_trips_and_defaults_for_old_sidecars():
    p = _suspect_pairs()[2]
    d = p.to_dict()
    assert d["matcher_ok"] is False
    assert G.GainPair.from_dict(d).matcher_ok is False
    del d["matcher_ok"]
    assert G.GainPair.from_dict(d).matcher_ok is True


def test_pair_peaks_sets_matcher_ok_from_its_own_verdict():
    q = lambda c: SimpleNamespace(centroid=c, centroid_err=0.1, fwhm=5.0, area=100.0)
    assert G._pair(q(1.0), q(2.0), None, False).matcher_ok is False
    assert G._pair(q(1.0), q(2.0), None, True).matcher_ok is True


def test_a_doubted_pair_stays_unticked_after_back_and_a_cutoff_change(qapp):
    s = ListModeSession()
    dlg = D.GainMatchDialog(s)
    dlg.set_pairs(_suspect_pairs())
    assert [dlg.is_included(r) for r in range(4)] == [True, True, False, True]
    assert dlg.apply_to_session() is None

    back = D.GainMatchDialog(s)                   # Back from step 4
    assert [back.is_included(r) for r in range(4)] == [True, True, False, True]
    back.cutoff.setValue(100)                     # re-applies the default ticks
    back.cutoff.setValue(D.DEFAULT_LOW_CUTOFF)
    assert [back.is_included(r) for r in range(4)] == [True, True, False, True]
    # The user may still tick it by hand; that is a choice, not a default.
    back.set_included(2, True)
    assert back.is_included(2)


def test_a_saved_sidecar_keeps_the_verdict(qapp, tmp_path):
    dlg = D.GainMatchDialog(ListModeSession())
    dlg.set_pairs(_suspect_pairs())
    path = tmp_path / "gain.json"
    assert dlg.save_coefficients(str(path))
    other = D.GainMatchDialog(ListModeSession())
    assert other.load_coefficients(str(path))
    other.cutoff.setValue(50)
    assert [other.is_included(r) for r in range(4)] == [True, True, False, True]


# ----------------------------------------------------------------------
# Step 1: the file opened on arrival; the note that clipped
# ----------------------------------------------------------------------

@pytest.fixture
def sdata(tmp_path):
    ch, ts, e, fl = simple_events(500)
    return write_sdata(str(tmp_path / "SData_run7.root"), ch, ts, e, fl)


def test_input_dialog_opens_the_given_file_on_arrival(qapp, sdata):
    s = ListModeSession()
    d = ListModeInputDialog(s, open_path=sdata)
    assert d.listmode is None                     # nothing before it is shown
    d.load_from_session()                         # what the controller does
    assert d.listmode is not None and d.path == sdata
    assert d.path_edit.text() == sdata
    assert d.info_labels["entries"].text() == "500"
    assert s.path is None                         # reaches the session on Next only
    d.next_button.click()
    assert d.step_result == NEXT and s.path == sdata


def test_the_file_is_opened_on_the_first_visit_only(qapp, sdata, tmp_path):
    steps = default_steps(open_path=sdata)
    s = ListModeSession()
    first = steps[0](s, None)
    first.load_from_session()
    assert first.listmode is not None
    # The user picks another file and goes on; a later visit (Back) must
    # show that file, not re-open the one File > Open started with.
    ch, ts, e, fl = simple_events(200, seed=9)
    other = write_sdata(str(tmp_path / "SData_other.root"), ch, ts, e, fl)
    assert first.open_path(other)
    first.next_button.click()
    second = steps[0](s, None)
    second.load_from_session()
    assert second.path == other and second.info_labels["entries"].text() == "200"


def test_max_dt_note_is_not_clipped_at_700_px(qapp, sdata):
    d = ListModeInputDialog(ListModeSession(), open_path=sdata)
    d.load_from_session()
    d.resize(700, d.sizeHint().height())
    d.show()
    qapp.processEvents()
    note = d.dt_note
    assert note.text().startswith("= ±2 µs")
    assert note.wordWrap()
    right = note.mapTo(d, note.rect().topRight()).x()
    assert right <= d.width(), (right, d.width())
    # The whole wrapped text fits the height it was given.
    assert note.heightForWidth(note.width()) <= note.height()
    assert note.width() >= note.fontMetrics().horizontalAdvance("= ±2 µs, a whole")
    d.close()
