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
def test_open_matrix_dialog_loads_off_the_gui_thread_with_progress(qapp, monkeypatch):
    """Decoding a real matrix is several seconds of CPU. It used to run on
    the GUI thread behind a wait cursor and a status message, which told
    the user something was happening but still left the window unable to
    repaint -- so it looked hung anyway. It now runs in a worker thread
    behind a progress dialog.

    Asserts what the old wait-cursor test could not: the decode really
    happens on a DIFFERENT thread from the GUI, real progress is
    reported rather than a spinner, and the decoded matrix is handed to
    the panel instead of being decoded a second time.
    """
    import threading

    import numpy as np

    main_window = MainWindow()
    path = os.path.join(FIXTURES, "gg.mtx")
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileName", lambda *a, **k: (path, ""))

    gui_thread = threading.current_thread()
    seen = {"thread": None, "progress": []}

    def fake_load_mtx(p, progress=None):
        seen["thread"] = threading.current_thread()
        for done in (0, 4096, 8192):
            if progress is not None:
                progress(done, 8192)
                seen["progress"].append(done)
        return np.zeros((4, 4), dtype=np.int64)

    monkeypatch.setattr("main_window.load_mtx", fake_load_mtx)

    panels = []
    monkeypatch.setattr(
        main_window, "_open_matrix_panel",
        lambda p, matrix=None: panels.append((p, matrix)),
    )

    main_window._open_matrix_dialog()

    assert seen["thread"] is not None, "the matrix was never loaded"
    assert seen["thread"] is not gui_thread, "the decode still ran on the GUI thread"
    assert seen["progress"] == [0, 4096, 8192], "progress was not reported"
    assert panels and panels[0][1] is not None, (
        "the decoded matrix must be handed to the panel, not decoded a second time"
    )


def test_open_matrix_dialog_reports_a_bad_file_instead_of_raising(qapp, monkeypatch):
    # The worker cannot let an exception escape run() -- that would kill
    # the thread with nothing to report. A ParseError must still reach the
    # user as a warning, exactly as it did when the load was synchronous.
    from histogram_io import ParseError

    main_window = MainWindow()
    monkeypatch.setattr(
        "main_window.QFileDialog.getOpenFileName", lambda *a, **k: ("broken.mtx", "")
    )

    def exploding_load(p, progress=None):
        raise ParseError("Not an lc-format matrix file (bad magic): broken.mtx")

    monkeypatch.setattr("main_window.load_mtx", exploding_load)

    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda parent, title, text, *a, **k: warnings.append((title, text)),
    )
    opened = []
    monkeypatch.setattr(
        main_window, "_open_matrix_panel", lambda p, matrix=None: opened.append(p)
    )

    main_window._open_matrix_dialog()

    assert warnings, "a corrupt matrix must still be reported to the user"
    assert "bad magic" in warnings[0][1]
    assert not opened, "no panel should be opened for a file that failed to load"
