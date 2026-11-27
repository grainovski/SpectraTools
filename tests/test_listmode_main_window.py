"""Sort List-Mode γγ Data as the user reaches it from the main window:
the Operations menu entry, File > Open (and Open ROOT File) of an SData
file, the five real step dialogs, and the hand-offs -- singles into the
spectrum list, the result into a Matrix panel (v7 plan, Task 6).

The step dialogs are the real ones. Only their exec() is replaced by a
driver that does what the user would do in each (open a file, press
Send, tick Skip, press Next...), because a modal exec() cannot be
clicked from a test. Everything behind the buttons -- the sort on its
worker thread, the files written, the panel opened -- runs for real.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QKeySequence

from channel_axis import ChannelAxis
from gain_match_dialog import GainMatchDialog
from json_matrix_io import load_json_matrix_with_meta
from listmode_input_dialog import ListModeInputDialog
from listmode_output_dialog import ListModeOutputDialog
from listmode_session import StepDialog
from listmode_singles_dialog import ListModeSinglesDialog
from main_window import MainWindow
from matrix_panel import MatrixPanel
from synthetic_sdata import simple_events, write_sdata
from time_window_dialog import TimeWindowDialog

TICK = 10_000           # 10 ns, the real run's timestamp granularity


# ----------------------------------------------------------------------
# A synthetic run: prompt coincidences on a random floor
# ----------------------------------------------------------------------

def _run_events(seed=5, n_coinc=1500, n_single=4000, span_ticks=3_000_000):
    """Time-sorted CH0/CH1 events: `n_coinc` true coincidences (both
    channels at the same timestamp, CH0 at 2000 and CH1 at 3000) and
    `n_single` uncorrelated singles over 30 ms, so a +-2 us pairing range
    holds plenty of random pairs as well."""
    rng = np.random.default_rng(seed)
    t_c = rng.integers(0, span_ticks, n_coinc)
    t_s = rng.integers(0, span_ticks, n_single)
    ch = np.concatenate([np.zeros(n_coinc), np.ones(n_coinc),
                         rng.integers(0, 2, n_single)]).astype(np.uint16)
    t = np.concatenate([t_c, t_c, t_s])
    e = np.concatenate([np.full(n_coinc, 2000), np.full(n_coinc, 3000),
                        rng.integers(100, 8000, n_single)]).astype(np.uint16)
    order = np.argsort(t, kind="stable")
    ts = ((t[order] + 100) * TICK).astype(np.uint64)
    return ch[order], ts, e[order], np.zeros(len(ch), dtype=np.uint32)


@pytest.fixture
def run_file(tmp_path):
    """<tmp>/Run_A/UNFILTERED/SData_Run_A.root, as CoMPASS lays a run out."""
    folder = tmp_path / "Run_A" / "UNFILTERED"
    folder.mkdir(parents=True)
    ch, ts, e, fl = _run_events()
    path = write_sdata(str(folder / "SData_Run_A.root"), ch, ts, e, fl)
    return path, (ch, ts, e, fl)


# ----------------------------------------------------------------------
# The driver: what the user does in each real step dialog
# ----------------------------------------------------------------------

class Driver:
    """Replaces StepDialog.exec. `browse_to` is the file the user picks
    with Browse... in step 1 (None: the dialog must already show one)."""

    def __init__(self, browse_to=None, open_how="compressed"):
        self.browse_to = browse_to
        self.open_how = open_how
        self.visits = []
        self.step1_had_file = None
        self.opened = None

    def __call__(self, dialog):
        self.visits.append(type(dialog).__name__)
        if isinstance(dialog, ListModeInputDialog):
            self.step1_had_file = dialog.listmode is not None
            if self.browse_to is not None:
                assert dialog.open_path(self.browse_to)
        elif isinstance(dialog, ListModeSinglesDialog):
            assert dialog.send_button.isEnabled()
            dialog.send_button.click()
        elif isinstance(dialog, GainMatchDialog):
            dialog.skip_check.setChecked(True)
        elif isinstance(dialog, TimeWindowDialog):
            assert dialog.readout["error"] is None, dialog.readout
        elif isinstance(dialog, ListModeOutputDialog):
            dialog.mtx_check.setChecked(False)          # the slow writer
            dialog.dense_check.setChecked(True)
            dialog.factor_combo.setCurrentIndex(dialog.factor_combo.findData(16))
            dialog.next_button.click()                  # Sort and Save
            assert dialog.next_button.text() == "Finish", dialog.summary_label.text()
            dialog._ask_open = lambda nbytes, factor: self.open_how
            self.opened = dialog.open_result()
        dialog.next_button.click()
        return 0


@pytest.fixture
def no_popups(monkeypatch):
    """Every message box recorded, never shown; Yes to the offer."""
    seen = []

    def record(kind, answer=None):
        def f(*a, **k):
            seen.append((kind, a[2] if len(a) > 2 else ""))
            return answer
        return f

    from PySide6.QtWidgets import QMessageBox
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(record(name)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        record("question", QMessageBox.StandardButton.Yes)))
    return seen


def _drive(monkeypatch, driver):
    monkeypatch.setattr(StepDialog, "exec", lambda self: driver(self))


def _check_result(window, run_path, events, driver):
    ch, _ts, e, _fl = events
    run_folder = Path(run_path).resolve().parent.parent
    assert driver.visits == ["ListModeInputDialog", "ListModeSinglesDialog",
                             "GainMatchDialog", "TimeWindowDialog",
                             "ListModeOutputDialog"]

    # Singles: ordinary spectra in the main window, path-shaped, in the run
    # folder, holding exactly each channel's events.
    by_name = {os.path.basename(s.path): s for s in window.spectra}
    assert set(by_name) == {"Run_A_CH0_singles.txt", "Run_A_CH1_singles.txt"}
    for c in (0, 1):
        s = by_name[f"Run_A_CH{c}_singles.txt"]
        assert Path(s.path).parent == run_folder
        assert s.variance is None and s.axis is None
        np.testing.assert_array_equal(
            s.data, np.bincount(e[ch == c], minlength=16384))

    # The files of step 5, in the run folder.
    assert (run_folder / "Run_A_gg_sub.json").is_file()
    assert (run_folder / "Run_A_gg_sub_x16.json").is_file()

    # The result in a Matrix panel: x16 compressed, so the compression's
    # channel axis, float with the negative cells a subtraction leaves.
    assert driver.opened == "compressed"
    panel = window._matrix_panels[-1]
    assert isinstance(panel, MatrixPanel)
    assert panel.matrix.shape == (1024, 1024)
    assert panel.matrix.dtype == np.float64
    assert panel.channel_axes == {"x": ChannelAxis(7.5, 16.0),
                                  "y": ChannelAxis(7.5, 16.0)}
    assert (panel.matrix < 0).any()
    assert os.path.basename(panel.path) == "Run_A_gg_sub_x16.json"
    # The coincidences: CH1 = 3000 on X, CH0 = 2000 on Y, i.e. bins
    # 3000 // 16 and 2000 // 16, survive the subtraction.
    peak = np.unravel_index(np.argmax(panel.matrix), panel.matrix.shape)
    assert peak == (2000 // 16, 3000 // 16)
    dense, _meta, _axes = load_json_matrix_with_meta(run_folder / "Run_A_gg_sub_x16.json")
    np.testing.assert_allclose(panel.matrix, dense, atol=1e-9)
    return panel


# ----------------------------------------------------------------------
# The menu entry
# ----------------------------------------------------------------------

def _all_shortcuts(widget):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QAbstractButton
    out = []
    for a in widget.findChildren(QAction):
        out += [s.toString() for s in a.shortcuts() if not s.isEmpty()]
    for b in widget.findChildren(QAbstractButton):
        if not b.shortcut().isEmpty():
            out.append(b.shortcut().toString())
    return out


def test_menu_entry_follows_subtract_behind_a_separator(qapp):
    window = MainWindow()
    actions = window.operations_menu.actions()
    i = actions.index(window.subtract_action)
    assert actions[i + 1].isSeparator()
    assert actions[i + 2] is window.sort_listmode_action
    assert window.sort_listmode_action.text() == "Sort List-Mode γγ Data..."
    # Always available: it starts from a file, not from loaded spectra.
    assert window.sort_listmode_action.isEnabled()
    window.close()


def test_shortcut_is_unique_in_the_main_window_and_the_matrix_panel(qapp):
    window = MainWindow()
    assert window.sort_listmode_action.shortcut() == QKeySequence("Ctrl+Shift+G")
    shortcuts = _all_shortcuts(window)
    assert shortcuts.count("Ctrl+Shift+G") == 1, shortcuts
    assert len(shortcuts) == len(set(shortcuts)), sorted(shortcuts)
    panel = window._open_matrix_panel("x.json", matrix=np.ones((8, 8)))
    assert "Ctrl+Shift+G" not in _all_shortcuts(panel)
    panel.close()
    window.close()


def test_shortcut_is_in_the_howto_table():
    from help_content import build_howto_html
    html = build_howto_html()
    assert "<tr><td><kbd>Ctrl+Shift+G</kbd></td><td>Sort List-Mode &gamma;&gamma; Data...</td></tr>" in html


def test_menu_to_matrix_panel_end_to_end(qapp, monkeypatch, run_file, no_popups):
    path, events = run_file
    window = MainWindow()
    driver = Driver(browse_to=path)
    _drive(monkeypatch, driver)

    window.sort_listmode_action.trigger()

    assert driver.step1_had_file is False       # the menu starts empty
    panel = _check_result(window, path, events, driver)
    assert not [m for m in no_popups if m[0] in ("warning", "critical")], no_popups
    panel.close()
    window.close()


def test_a_failing_operation_is_reported_not_swallowed(qapp, monkeypatch, no_popups):
    import listmode_operation

    def boom(*a, **k):
        raise RuntimeError("deliberate")
    monkeypatch.setattr(listmode_operation, "run_listmode_operation", boom)
    window = MainWindow()
    assert window._open_listmode_operation() is None
    assert ("critical", "RuntimeError: deliberate") in no_popups
    window.close()


# ----------------------------------------------------------------------
# File > Open of an SData file
# ----------------------------------------------------------------------

def test_file_open_of_sdata_starts_with_the_file_already_open(
        qapp, monkeypatch, run_file, no_popups):
    path, events = run_file
    window = MainWindow()
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileNames",
                        lambda *a, **k: ([path], ""))
    driver = Driver(browse_to=None)              # no Browse: already open
    _drive(monkeypatch, driver)

    window._open_file_dialog()

    offers = [m for m in no_popups if m[0] == "question"]
    assert offers and "list-mode" in offers[0][1], no_popups
    assert driver.step1_had_file is True
    panel = _check_result(window, path, events, driver)
    panel.close()
    window.close()


def test_file_open_filter_offers_root_files(qapp, monkeypatch):
    seen = {}

    def fake(parent, caption, folder, filters, *a, **k):
        seen["filters"] = filters
        return [], ""
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileNames", fake)
    window = MainWindow()
    window._open_file_dialog()
    assert "(*.root)" in seen["filters"]
    assert seen["filters"].endswith("All files (*)")
    window.close()


def test_declining_the_offer_opens_nothing(qapp, monkeypatch, run_file, no_popups):
    from PySide6.QtWidgets import QMessageBox
    path, _ = run_file
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.No))
    started = []
    window = MainWindow()
    monkeypatch.setattr(window, "_open_listmode_operation",
                        lambda **k: started.append(k))
    window._load_files([path])
    assert started == [] and window.spectra == []
    # Declined is not an error: no "could not open" on top of it.
    assert not [m for m in no_popups if m[0] == "warning"], no_popups
    window.close()


def test_open_root_file_menu_offers_it_too(qapp, monkeypatch, run_file, no_popups):
    path, _ = run_file
    window = MainWindow()
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileName",
                        lambda *a, **k: (path, ""))
    started = []
    monkeypatch.setattr(window, "_open_listmode_operation",
                        lambda **k: started.append(k))
    window._open_root_dialog()
    assert started == [{"open_path": path}]
    assert not [m for m in no_popups if m[0] == "warning"], no_popups
    window.close()


def test_a_root_file_with_neither_histograms_nor_events_is_still_refused(
        qapp, monkeypatch, tmp_path, no_popups):
    # A tree that is not CoMPASS's: not list-mode, so no offer.
    ch, ts, e, fl = simple_events(50)
    path = write_sdata(str(tmp_path / "other.root"), ch, ts, e, fl, tree="Events")
    window = MainWindow()
    started = []
    monkeypatch.setattr(window, "_open_listmode_operation",
                        lambda **k: started.append(k))
    window._load_files([path])
    assert started == []
    assert ("warning", "other.root contains no 1D or 2D histograms.") in no_popups
    window.close()


def test_a_histogram_root_file_from_file_open_goes_to_the_object_dialog(
        qapp, monkeypatch, tmp_path, no_popups):
    import uproot
    path = str(tmp_path / "hists.root")
    with uproot.recreate(path) as f:
        f["h"] = (np.arange(10, dtype=float), np.linspace(0, 10, 11))
    seen = []

    class FakeDialog:
        def __init__(self, parent, p, objects):
            seen.append((p, [o[0] for o in objects]))
            self.result_matrix, self.result_spectra = None, []

        def exec(self):
            return 0
    monkeypatch.setattr("root_dialog.RootObjectDialog", FakeDialog)
    window = MainWindow()
    window._load_files([path])
    assert seen and seen[0][0] == path and seen[0][1]
    window.close()


def test_spectra_and_root_files_together(qapp, monkeypatch, run_file, tmp_path, no_popups):
    """The ordinary files are loaded as before; the SData file is offered."""
    path, _ = run_file
    txt = tmp_path / "s.txt"
    txt.write_text("\n".join(str(i) for i in range(20)))
    window = MainWindow()
    started = []
    monkeypatch.setattr(window, "_open_listmode_operation",
                        lambda **k: started.append(k))
    window._load_files([str(txt), path])
    assert [s.path for s in window.spectra] == [str(txt)]
    assert started == [{"open_path": path}]
    window.close()


# ----------------------------------------------------------------------
# Real data (untracked run; skipped by name when absent)
# ----------------------------------------------------------------------

REAL = (Path(__file__).resolve().parent.parent / "Na22_261126_ANI_STEF_180"
        / "UNFILTERED" / "SData_Na22_261126_ANI_STEF.root")


def test_real_sdata_is_recognised_as_list_mode(qapp, monkeypatch, no_popups):
    if not REAL.is_file():
        pytest.skip("untracked run Na22_261126_ANI_STEF_180/ is absent")
    window = MainWindow()
    started = []
    monkeypatch.setattr(window, "_open_listmode_operation",
                        lambda **k: started.append(k))
    window._load_files([str(REAL)])
    assert started == [{"open_path": str(REAL)}]
    # CoMPASS's own histogram file of the same run is NOT list-mode.
    import listmode_io
    hists = REAL.parent / "Hcompass_Na22_261126_ANI_STEF_20261126_115129.root"
    if hists.is_file():
        assert not listmode_io.is_listmode_file(hists)
    window.close()
