import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from main_window import MainWindow

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


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
