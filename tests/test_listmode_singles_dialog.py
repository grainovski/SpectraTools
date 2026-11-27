"""Step 2 of the list-mode operation: the singles dialog."""

import numpy as np

from listmode_session import NEXT, ListModeSession
from listmode_singles_dialog import ListModeSinglesDialog, run_label
from listmode_sort import Pairs, Singles


def _session(**kw):
    h0 = np.zeros(16384, dtype=np.int64)
    h1 = np.zeros(16384, dtype=np.int64)
    h0[[100, 5000]] = [7, 3]
    h1[200] = 11
    singles = Singles({0: h0, 1: h1}, overflow={0: 0, 1: 2}, excluded=5)
    pairs = Pairs(np.array([0, 10_000, -20_000]), np.array([1, 2, 3]),
                  np.array([4, 5, 6]))
    return ListModeSession(path="C:/runs/x/UNFILTERED/SData_Na22_run.root",
                           singles=singles, pairs=pairs, **kw)


def test_plot_built_and_summary(qapp):
    d = ListModeSinglesDialog(_session())
    d.load_from_session()
    labels = [line.get_label() for line in d.axes.get_lines()]
    assert labels == ["CH0 (Y)", "CH1 (X)"]
    assert d.axes.get_yscale() == "log"
    assert d.axes.get_xlim() == (0, 16383)
    assert len(d.axes.get_lines()[0].get_xdata()) == 16384
    text = d.summary_label.text()
    assert "CH0: 10 counts, none outside" in text
    assert "CH1: 11 counts, 2 outside 0\u201316383" in text
    assert "excluded by their Flags value: 5" in text
    assert "\u2264 2 \u00b5s: 3" in text


def test_log_toggle(qapp):
    d = ListModeSinglesDialog(_session())
    d.load_from_session()
    d.log_check.setChecked(False)
    assert d.axes.get_yscale() == "linear"
    assert len(d.axes.get_lines()) == 2
    d.log_check.setChecked(True)
    assert d.axes.get_yscale() == "log"


def test_send_spectra_gets_two_named_arrays(qapp):
    got = []
    s = _session(send_spectra=got.append)
    d = ListModeSinglesDialog(s)
    d.load_from_session()
    assert d.send_button.isEnabled()
    d.send_button.click()
    assert len(got) == 1
    (n0, a0), (n1, a1) = got[0]
    assert (n0, n1) == ("Na22_run CH0 singles", "Na22_run CH1 singles")
    assert a0 is not None and int(a0.sum()) == 10 and int(a1.sum()) == 11
    assert "Na22_run CH0 singles" in d.sent_label.text()


def test_send_button_disabled_without_callback(qapp):
    d = ListModeSinglesDialog(_session())
    d.load_from_session()
    assert not d.send_button.isEnabled()


def test_send_failure_is_shown(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr("listmode_singles_dialog.QMessageBox.warning",
                        lambda *a: shown.append(a[2]))

    def boom(spectra):
        raise RuntimeError("main window gone")

    d = ListModeSinglesDialog(_session(send_spectra=boom))
    d.load_from_session()
    d.send_button.click()
    assert shown and "main window gone" in shown[0]


def test_next_always_succeeds(qapp):
    d = ListModeSinglesDialog(_session())
    d.load_from_session()
    d.next_button.click()
    assert d.step_result == NEXT
    assert d.back_button.isEnabled()


def test_run_label_prefers_info_run_id():
    class Info:
        run_id = "from_info"

    assert run_label(_session(run_info=Info())) == "from_info"
    assert run_label(_session()) == "Na22_run"
    assert run_label(ListModeSession(path="plain.root")) == "plain"
    assert run_label(ListModeSession()) == "list-mode"


def test_dark_theme(qapp):
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    parent._theme = "dark"
    d = ListModeSinglesDialog(_session(), parent)
    d.load_from_session()
    assert d._theme == "dark"
    import matplotlib.colors as mc
    assert mc.to_hex(d.axes.get_facecolor()) != "#ffffff"
