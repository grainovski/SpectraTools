import functools
import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import matrix_panel
from main_window import MainWindow

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Same fix as test_mtx_io.py/test_matrix_panel.py's own caching: several
# tests below open two panels each (for multi-panel exclusivity), and
# MatrixPanel.__init__ decodes the real gg.mtx fixture via load_mtx every
# time -- several seconds each. Memoize by path and monkeypatch it in for
# every test, since all tests here use the same fixture path and load_mtx
# is a pure function of it.
_cached_load_mtx = functools.lru_cache(maxsize=None)(matrix_panel.load_mtx)


@pytest.fixture(autouse=True)
def _use_cached_load_mtx(monkeypatch):
    monkeypatch.setattr(matrix_panel, "load_mtx", _cached_load_mtx)


def test_open_matrix_action_exists(qapp):
    main_window = MainWindow()
    assert hasattr(main_window, "open_matrix_action")
    assert main_window.open_matrix_action.shortcut().toString() == "Ctrl+Shift+O"


def test_activating_main_window_disables_matrix_panel(qapp):
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))

    main_window._on_activated()

    assert panel.isEnabled() is False


def test_activating_matrix_panel_disables_main_window(qapp):
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))

    panel._on_activated()

    assert main_window.isEnabled() is False


def test_closing_matrix_panel_reenables_main_window(qapp):
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel._on_activated()
    assert main_window.isEnabled() is False

    panel.close()

    assert main_window.isEnabled() is True


def test_open_matrix_dialog_remembers_last_folder(qapp, monkeypatch):
    main_window = MainWindow()
    path = os.path.join(FIXTURES, "gg.mtx")
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileName", lambda *a, **k: (path, ""))

    main_window._open_matrix_dialog()

    assert main_window.settings.last_folder() == os.path.dirname(path)


def test_closing_main_window_closes_matrix_panels(qapp):
    # Regression guard: MatrixPanel is a parentless top-level window (see
    # matrix_panel.py), so Qt's quitOnLastWindowClosed would never fire if
    # MainWindow closed while a panel was still open -- app.exec() would
    # never return, leaving the process running with no visible explanation.
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))

    main_window.close()

    assert panel.isVisible() is False
    assert panel not in main_window._matrix_panels


def test_closing_matrix_panel_closes_heatmap_windows(qapp):
    # Same parentless-top-level-window concern one level down: a heatmap
    # window left open would keep blocking quit even after both the panel
    # and MainWindow are closed.
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel._open_heatmap()
    heatmap = panel._heatmap_windows[0]

    panel.close()

    assert heatmap.isVisible() is False


def test_activating_one_of_two_matrix_panels_disables_the_other(qapp):
    # Regression guard: MainWindow._on_activated already loops over every
    # tracked panel, but MatrixPanel._on_activated originally only knew
    # about main_window -- with two panels open, activating one never
    # disabled the OTHER panel, so both could end up enabled at once,
    # breaking "activating one disables the other" for anything beyond
    # exactly one panel.
    main_window = MainWindow()
    panel_a = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))

    panel_a._on_activated()

    assert panel_a.isEnabled() is True
    assert panel_b.isEnabled() is False
    assert main_window.isEnabled() is False


def test_closing_inactive_panel_does_not_steal_exclusivity_from_active_one(qapp):
    # Regression guard: closeEvent originally re-enabled main_window
    # unconditionally. With two panels open and panel_b the currently
    # active one (main_window and panel_a both disabled), closing the
    # already-disabled panel_a must NOT re-enable main_window out from
    # under panel_b's exclusivity.
    main_window = MainWindow()
    panel_a = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b._on_activated()
    assert main_window.isEnabled() is False
    assert panel_a.isEnabled() is False

    panel_a.close()

    assert main_window.isEnabled() is False
    assert panel_b.isEnabled() is True


def test_open_matrix_dialog_shows_busy_feedback_during_load(qapp, monkeypatch):
    # Decoding a real matrix takes several seconds on the GUI thread --
    # verify the wait cursor and status message are up for the duration of
    # _open_matrix_panel (not just theoretically set somewhere), and are
    # cleared again once _open_matrix_dialog returns. Stubs out
    # _open_matrix_panel (rather than using the real fixture) so this test
    # doesn't pay for another full matrix decode.
    main_window = MainWindow()
    path = os.path.join(FIXTURES, "gg.mtx")
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileName", lambda *a, **k: (path, ""))

    captured = {}

    def fake_open_matrix_panel(p):
        captured["cursor"] = QApplication.overrideCursor()
        captured["message"] = main_window.statusBar().currentMessage()

    monkeypatch.setattr(main_window, "_open_matrix_panel", fake_open_matrix_panel)

    main_window._open_matrix_dialog()

    assert captured["cursor"] is not None
    assert captured["cursor"].shape() == Qt.CursorShape.WaitCursor
    assert captured["message"] == "Loading matrix..."
    assert QApplication.overrideCursor() is None
    assert main_window.statusBar().currentMessage() == ""
