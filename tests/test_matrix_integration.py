import functools
import os

import pytest

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


def test_closing_a_matrix_panel_releases_its_matrix(qapp):
    """A closed panel used to keep its whole decoded matrix resident.
    The panel is a parentless top-level widget, so Qt never owned it, and
    the signal connections made in __init__ capture `self` in closures
    that outlive close() -- three open/close cycles on one 8192x8192
    matrix retained 1.6 GB, growing for the life of the session.

    Asserts the payload is released rather than that the widget is
    collected: whether the object itself is freed depends on Qt/Python
    ownership details that are easy to regress, while the memory is what
    actually matters.
    """
    import weakref

    main_window = MainWindow()
    path = os.path.join(FIXTURES, "gg.mtx")

    # Weakrefs to just the panels THIS test opens. A global sweep of every
    # live MatrixPanel would also pick up panels other tests in this
    # module deliberately leave open, and fail for their sake rather than
    # for a real leak.
    closed = []
    for _ in range(2):
        panel = main_window._open_matrix_panel(path)
        assert panel.matrix is not None and panel.matrix.nbytes > 0
        panel.close()
        closed.append(weakref.ref(panel))
        del panel

    still_held = 0
    for ref in closed:
        panel = ref()
        if panel is not None and getattr(panel, "matrix", None) is not None:
            still_held += panel.matrix.nbytes
    assert still_held == 0, (
        f"{still_held / 1e6:,.0f} MB of matrix data still held by closed panels"
    )
    assert main_window._matrix_panels == []


# --- CoMPASS JSON matrices -------------------------------------------------

_JSON_FIXTURE = os.path.join(FIXTURES, "compass", "co60_EvsE.json")


def test_open_matrix_dialog_offers_json_matrices(qapp, monkeypatch):
    main_window = MainWindow()
    seen = {}

    def fake_dialog(parent, caption, folder, filters, *a, **k):
        seen["filters"] = filters
        return ("", "")

    monkeypatch.setattr("main_window.QFileDialog.getOpenFileName", fake_dialog)
    main_window._open_matrix_dialog()

    first = seen["filters"].split(";;")[0]
    assert "*.mtx" in first and "*.json" in first, seen["filters"]


@pytest.mark.parametrize("name", ["co60_EvsE.json", "CO60_EVSE.JSON"])
def test_a_json_matrix_is_read_by_the_json_reader_not_load_mtx(
        qapp, monkeypatch, tmp_path, name):
    """Routed by extension, case-insensitively. The .mtx decoder must never
    see a JSON file -- it would report 'bad magic' for a perfectly good
    matrix."""
    import shutil

    import numpy as np

    from json_matrix_io import load_json_matrix

    path = str(tmp_path / name)
    shutil.copyfile(_JSON_FIXTURE, path)

    main_window = MainWindow()
    monkeypatch.setattr(
        "main_window.QFileDialog.getOpenFileName", lambda *a, **k: (path, ""))

    def must_not_run(*a, **k):
        raise AssertionError("load_mtx was called for a JSON matrix")

    monkeypatch.setattr("main_window.load_mtx", must_not_run)
    panels = []
    monkeypatch.setattr(main_window, "_open_matrix_panel",
                        lambda p, matrix=None: panels.append((p, matrix)))

    main_window._open_matrix_dialog()

    assert panels, "no panel was opened"
    assert panels[0][0] == path
    assert np.array_equal(panels[0][1], load_json_matrix(_JSON_FIXTURE))


def test_a_json_matrix_bypasses_the_mtx_decode_cache(qapp, monkeypatch):
    """The cache exists because decoding .mtx takes seconds. A JSON matrix
    reads in well under a second, and caching it would only add a stale
    copy to keep in step with the file."""
    import matrix_cache

    main_window = MainWindow()
    monkeypatch.setattr(
        "main_window.QFileDialog.getOpenFileName",
        lambda *a, **k: (_JSON_FIXTURE, ""))
    calls = []
    monkeypatch.setattr(matrix_cache, "load", lambda p: calls.append(("load", p)))
    monkeypatch.setattr(matrix_cache, "store",
                        lambda p, m: calls.append(("store", p)))
    monkeypatch.setattr(main_window, "_open_matrix_panel",
                        lambda p, matrix=None: None)
    # Stubbed so a regression FAILS instead of hanging: if a JSON ever
    # reaches load_mtx again, its "bad magic" error opens a modal warning
    # that nobody can click in a headless run.
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda parent, title, text, *a, **k: warnings.append(text))

    main_window._open_matrix_dialog()

    assert calls == [], calls
    assert warnings == [], warnings


def test_a_bad_json_matrix_is_reported_not_raised(qapp, monkeypatch, tmp_path):
    bad = tmp_path / "spectrum.json"
    bad.write_text('{"data": [1, 2, 3]}')

    main_window = MainWindow()
    monkeypatch.setattr(
        "main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(bad), ""))
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda parent, title, text, *a, **k: warnings.append((title, text)))
    opened = []
    monkeypatch.setattr(main_window, "_open_matrix_panel",
                        lambda p, matrix=None: opened.append(p))

    main_window._open_matrix_dialog()

    assert warnings and "1-D" in warnings[0][1], warnings
    assert not opened
