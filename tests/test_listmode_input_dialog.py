"""Step 1 of the list-mode operation: the input dialog."""

import numpy as np
import pytest

import listmode_input_dialog as lid
from listmode_input_dialog import ListModeInputDialog, snap_max_dt_ps
from listmode_session import NEXT, ListModeSession
from synthetic_sdata import simple_events, write_sdata


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr("listmode_session.QMessageBox.warning",
                        lambda *a: shown.append(a[2]))
    return shown


def _file(tmp_path, name="SData_run7.root", tree="Data", flags=None,
          shuffle=False, n=2000):
    ch, ts, e, fl = simple_events(n, seed=3)
    if flags is not None:
        fl = flags(fl)
    if shuffle:
        ts = ts.copy()
        ts[[10, 11]] = ts[[11, 10]]
    return write_sdata(str(tmp_path / name), ch, ts, e, fl, tree=tree), \
        (ch, ts, e, fl)


def _dialog(tmp_path, **kw):
    path, events = _file(tmp_path, **kw)
    d = ListModeInputDialog(ListModeSession())
    d.load_from_session()
    assert d.open_path(path)
    return d, path, events


def test_snap_max_dt():
    assert snap_max_dt_ps(2_000_000, 10_000) == 2_000_000
    assert snap_max_dt_ps(2_004_999, 10_000) == 2_000_000
    assert snap_max_dt_ps(2_005_001, 10_000) == 2_010_000
    assert snap_max_dt_ps(1, 10_000) == 10_000       # at least one tick
    assert snap_max_dt_ps(1234.4, None) == 1234


def test_fields_shown(qapp, tmp_path):
    d, path, (ch, ts, e, fl) = _dialog(tmp_path)
    info = d.info_labels
    assert info["tree"].text() == "Data (UNFILTERED)"
    assert info["entries"].text() == "2,000"
    n0, n1 = int(np.sum(ch == 0)), int(np.sum(ch == 1))
    assert info["channels"].text() == f"CH0: {n0:,}, CH1: {n1:,}"
    assert info["granularity"].text() == "10 ns (10,000 ps)"
    assert info["sorted"].text() == "time-sorted"
    assert "no <run>_info.txt" in info["run"].text()
    assert d.path_edit.text() == path
    assert d.warning_label.isHidden() and d.error_label.isHidden()


def test_run_info_shown_when_present(qapp, tmp_path):
    run = tmp_path / "run7"
    (run / "UNFILTERED").mkdir(parents=True)
    (run / "run7_info.txt").write_text(
        "Run ID = run7\nStart time = Mon 1\nStop time = Mon 2\n"
        "CH0@DT5780SCP_1\n\tLive time = 0:10:00.500\n\tReal time = 0:10:01\n",
        encoding="latin-1")
    ch, ts, e, fl = simple_events(500)
    path = write_sdata(str(run / "UNFILTERED" / "SData_run7.root"),
                       ch, ts, e, fl)
    d = ListModeInputDialog(ListModeSession())
    assert d.open_path(path)
    assert d.info_labels["run"].text() == "run7"
    assert d.info_labels["start"].text() == "Mon 1"
    assert d.info_labels["stop"].text() == "Mon 2"
    assert "600.500 s live" in d.info_labels["live"].text()


def test_flag_table_defaults(qapp, tmp_path):
    def flags(fl):
        fl = fl.copy()
        fl[7::301] = 0x480
        return fl

    d, *_ = _dialog(tmp_path, flags=flags)
    values = [int(d.flags_table.item(r, lid.COL_EXCLUDE).data(256))
              for r in range(d.flags_table.rowCount())]
    assert values == [0x0, 0x480, 0x800, 0x8000]
    assert d.excluded_flag_values() == {0x480, 0x8000}
    row = values.index(0x8000)
    assert d.flags_table.item(row, lid.COL_MEANING).text() == "piled-up event"
    assert d.flags_table.item(values.index(0), lid.COL_MEANING).text() \
        == "no flag set"
    d.set_flag_excluded(0x800, True)
    d.set_flag_excluded(0x8000, False)
    assert d.excluded_flag_values() == {0x480, 0x800}


def test_detectors_default_and_must_differ(qapp, tmp_path, warnings):
    d, *_ = _dialog(tmp_path)
    assert (d.x_combo.currentData(), d.y_combo.currentData()) == (1, 0)
    assert [d.x_combo.itemData(i) for i in range(d.x_combo.count())] == [0, 1]
    d.y_combo.setCurrentIndex(d.y_combo.findData(1))
    assert not d.channel_label.isHidden()
    d.next_button.click()
    assert d.step_result != NEXT
    assert warnings == ["The X and Y detectors must be different channels."]
    assert d.session.path is None  # nothing written on refusal


def test_max_dt_snapped_to_granularity(qapp, tmp_path):
    d, *_ = _dialog(tmp_path)
    assert d.max_dt_ps() == 2_000_000
    d.max_dt_spin.setValue(1234.0)       # ns
    assert d.max_dt_ps() == 1_230_000
    d._snap_spin()
    assert d.max_dt_spin.value() == pytest.approx(1230.0)
    assert d.max_dt_spin.singleStep() == pytest.approx(10.0)


def test_filtered_stream_warns_but_is_accepted(qapp, tmp_path, warnings):
    d, *_ = _dialog(tmp_path, name="SDataF_run7.root", tree="Data_F")
    assert d.info_labels["tree"].text() == "Data_F (FILTERED)"
    assert not d.warning_label.isHidden()
    assert "UNFILTERED" in d.warning_label.text()
    assert "random window has nothing to measure" in d.warning_label.text()
    d.next_button.click()
    assert warnings == [] and d.step_result == NEXT


def test_unsorted_file_refused(qapp, tmp_path, warnings):
    d, *_ = _dialog(tmp_path, shuffle=True)
    assert d.info_labels["sorted"].text() == "NOT time-sorted"
    assert "not in time order" in d.warning_label.text()
    d.next_button.click()
    assert d.step_result != NEXT
    assert warnings == [lid.UNSORTED_ERROR]


def test_unreadable_file_shows_an_error(qapp, tmp_path, warnings):
    bad = tmp_path / "SData_bad.root"
    bad.write_bytes(b"this is not ROOT")
    d = ListModeInputDialog(ListModeSession())
    assert d.open_path(str(bad)) is False
    assert not d.error_label.isHidden()
    assert "Not a readable ROOT file" in d.error_label.text()
    assert d.listmode is None
    d.next_button.click()
    assert d.step_result != NEXT
    assert warnings and "Choose a list-mode file" in warnings[0]


def test_no_file_refused(qapp, warnings):
    d = ListModeInputDialog(ListModeSession())
    d.load_from_session()
    d.next_button.click()
    assert d.step_result != NEXT and len(warnings) == 1


def test_apply_to_session(qapp, tmp_path):
    d, path, _ = _dialog(tmp_path)
    d.set_flag_excluded(0x800, True)
    d.max_dt_spin.setValue(506.0)
    assert d.apply_to_session() is None
    s = d.session
    assert s.path == path
    assert s.listmode is d.listmode and s.listmode.n_entries == 2000
    assert s.run_info is None
    assert s.exclude_flags == {0x800, 0x8000}
    assert isinstance(s.exclude_flags, set)
    assert (s.x_channel, s.y_channel) == (1, 0)
    assert s.max_dt_ps == 510_000


def test_load_from_session_restores_choices(qapp, tmp_path):
    d, path, _ = _dialog(tmp_path)
    d.set_flag_excluded(0x8000, False)
    d.x_combo.setCurrentIndex(d.x_combo.findData(0))
    d.y_combo.setCurrentIndex(d.y_combo.findData(1))
    d.max_dt_spin.setValue(800.0)
    assert d.apply_to_session() is None

    again = ListModeInputDialog(d.session)
    again.load_from_session()
    assert again.path_edit.text() == path
    assert again.excluded_flag_values() == set()
    assert (again.x_combo.currentData(), again.y_combo.currentData()) == (0, 1)
    assert again.max_dt_ps() == 800_000
    assert again.apply_to_session() is None
    assert d.session.exclude_flags == set()


def test_browse_uses_file_dialog_and_initial_folder(qapp, tmp_path,
                                                   monkeypatch):
    path, _ = _file(tmp_path)
    asked = {}

    def fake(parent, caption, folder, filt):
        asked["folder"], asked["filter"] = folder, filt
        return path, filt

    monkeypatch.setattr(lid.QFileDialog, "getOpenFileName", fake)
    d = ListModeInputDialog(ListModeSession(), initial_path=str(tmp_path))
    d.browse_button.click()
    assert asked["folder"] == str(tmp_path)
    assert "*.root" in asked["filter"]
    assert d.listmode is not None and d.path_edit.text() == path


def test_browse_cancelled_changes_nothing(qapp, monkeypatch):
    monkeypatch.setattr(lid.QFileDialog, "getOpenFileName",
                        lambda *a: ("", ""))
    d = ListModeInputDialog(ListModeSession())
    d.browse_button.click()
    assert d.listmode is None and d.path is None
