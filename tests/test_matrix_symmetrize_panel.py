"""Symmetrize... as the user reaches it: panel button, dialog, new panel,
and a cut on the symmetrized matrix carrying the corrected variance."""

import os

import numpy as np
import pytest

from main_window import MainWindow
from matrix_cut import compute_projection

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "compass")


def _asymmetric(n=40):
    m = np.zeros((n, n), dtype=np.int64)
    m[12, 12] = 100          # diagonal: same events end up there twice
    m[12, 25] = 30
    m[25, 7] = 11
    return m


def _panel(qapp, matrix, path="synthetic.json"):
    window = MainWindow()
    return window, window._open_matrix_panel(path, matrix=matrix)


def test_the_panel_offers_symmetrize(qapp):
    window, panel = _panel(qapp, _asymmetric())
    assert panel.symmetrize_button.text() == "Symmetrize..."
    assert panel.symmetrize_button.isEnabled()
    panel.close(); window.close()


def test_symmetrizing_opens_a_new_panel_and_leaves_the_original(qapp):
    m = _asymmetric()
    window, panel = _panel(qapp, m)
    original = panel.matrix.copy()

    new = panel._symmetrize_with(0.0, 1.0)

    assert new is not panel and new in window._matrix_panels
    assert np.array_equal(panel.matrix, original), "the original must be untouched"
    assert np.array_equal(new.matrix, m + m.T)
    assert np.array_equal(new.symmetric_diagonal, np.diag(m))
    assert new.path == "synthetic_sym.json", "real extension last, for fit logs"
    assert "(symmetrized)" in new.windowTitle()
    new.close(); panel.close(); window.close()


def test_a_symmetrized_panel_fixes_the_axis_and_cannot_be_folded_again(qapp):
    window, panel = _panel(qapp, _asymmetric())
    new = panel._symmetrize_with(0.0, 1.0)
    assert not new.axis_selector.isEnabled()
    assert new.working_axis == "x"
    assert not new.symmetrize_button.isEnabled()
    assert np.array_equal(new.projections["x"], new.projections["y"])
    new.close(); panel.close(); window.close()


def test_a_cut_on_the_symmetrized_matrix_carries_the_diagonal_variance(qapp):
    """End to end through Activate Cut: the diagonal channel's variance is
    4*D (here 400), not the 2*D = 200 'variance = counts' would give."""
    window, panel = _panel(qapp, _asymmetric())
    new = panel._symmetrize_with(0.0, 1.0)
    state = new.cut_controller.state
    state.add_cut_click(12)
    state.add_cut_click(12)
    new._activate_cut()

    cut = window.spectra[-1]
    assert cut.data[12] == 200
    assert cut.variance[12] == 400
    assert cut.variance[25] == 30            # an ordinary cell stays Poisson
    new.close(); panel.close(); window.close()


def test_an_unsymmetrizable_matrix_is_explained_not_opened(qapp, monkeypatch):
    sym = np.array([[3, 1], [1, 2]], dtype=np.int64)
    window, panel = _panel(qapp, sym)
    messages = []
    monkeypatch.setattr("matrix_panel.QMessageBox.information",
                        lambda parent, title, text, *a, **k: messages.append(text))
    before = len(window._matrix_panels)

    panel._open_symmetrize_dialog()

    assert messages and "already symmetric" in messages[0]
    assert len(window._matrix_panels) == before
    panel.close(); window.close()


def test_lost_counts_are_reported(qapp, monkeypatch):
    m = np.zeros((10, 10), dtype=np.int64)
    m[9, 2] = 5                       # top row; a shift of +5 pushes it off
    m[3, 4] = 1
    window, panel = _panel(qapp, m)
    messages = []
    monkeypatch.setattr("matrix_panel.QMessageBox.information",
                        lambda parent, title, text, *a, **k: messages.append(text))
    new = panel._symmetrize_with(5.0, 1.0)
    assert messages and "5 counts" in messages[0], messages
    new.close(); panel.close(); window.close()


# --- the dialog -------------------------------------------------------------


@pytest.fixture
def co60_projections():
    from json_matrix_io import load_json_matrix
    m = load_json_matrix(os.path.join(FIXTURES, "co60_EvsE.json"))
    return compute_projection(m, "x"), compute_projection(m, "y")


def _ok(dialog):
    from PySide6.QtWidgets import QDialogButtonBox
    return dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)


def test_dialog_starts_by_asking_for_two_lines(qapp, co60_projections):
    from symmetrize_dialog import SymmetrizeDialog
    d = SymmetrizeDialog(None, *co60_projections)
    assert d.mapping is None and not _ok(d).isEnabled()
    assert "two lines" in d.summary.text()


def test_dialog_derives_the_map_from_rough_positions(qapp, co60_projections):
    from symmetrize_dialog import SymmetrizeDialog
    d = SymmetrizeDialog(None, *co60_projections)
    for box, v in zip(d.x_boxes + d.y_boxes, (149, 169, 152, 172)):
        box.setValue(v)
    assert d.mapping is not None and _ok(d).isEnabled()
    a, b = d.mapping
    assert 0.95 < b < 0.99
    assert abs((a + b * 151.87) - 148.78) < 0.4   # Y's 1173 lands on X's


def test_dialog_same_scale_choice_needs_no_lines(qapp, co60_projections):
    from symmetrize_dialog import SymmetrizeDialog
    d = SymmetrizeDialog(None, *co60_projections)
    d.same_radio.setChecked(True)
    assert d.mapping == (0.0, 1.0) and _ok(d).isEnabled()
    assert not d.x_boxes[0].isEnabled()


def test_dialog_explains_a_useless_pair(qapp, co60_projections):
    from symmetrize_dialog import SymmetrizeDialog
    d = SymmetrizeDialog(None, *co60_projections)
    d.refine_check.setChecked(False)
    for box, v in zip(d.x_boxes + d.y_boxes, (149, 169, 152, 152.5)):
        box.setValue(v)
    assert d.mapping is None and not _ok(d).isEnabled()
    assert "2 channels apart" in d.summary.text()


# --- off the GUI thread -----------------------------------------------------


class _AcceptedDialog:
    """Stands in for SymmetrizeDialog: accepted, with a chosen map, and no
    modal loop to hang a headless run."""

    mapping = (0.0, 1.0)

    def __init__(self, *a, **k):
        pass

    def exec(self):
        return 1


def test_the_dialog_path_symmetrizes_off_the_gui_thread(qapp, monkeypatch):
    """A full 8192x8192 fold took 8.6 s. Run on the GUI thread that is a
    frozen window, which is what made matrix loading feel broken before it
    moved to a worker. The dialog path must therefore fold in a thread."""
    import threading

    import matrix_panel
    import symmetrize_dialog

    monkeypatch.setattr(symmetrize_dialog, "SymmetrizeDialog", _AcceptedDialog)
    gui = threading.current_thread()
    seen = {}
    real = matrix_panel.symmetrize

    def spy(*a, **k):
        seen["thread"] = threading.current_thread()
        return real(*a, **k)

    monkeypatch.setattr(matrix_panel, "symmetrize", spy)
    m = _asymmetric()
    window, panel = _panel(qapp, m)
    before = len(window._matrix_panels)

    panel._open_symmetrize_dialog()

    assert seen.get("thread") is not None, "symmetrize never ran"
    assert seen["thread"] is not gui, "the fold ran on the GUI thread"
    assert len(window._matrix_panels) == before + 1
    new = window._matrix_panels[-1]
    assert np.array_equal(new.matrix, m + m.T)
    new.close(); panel.close(); window.close()


def test_a_failure_in_the_worker_is_reported_not_raised(qapp, monkeypatch):
    import matrix_panel
    import symmetrize_dialog
    from matrix_symmetrize import SymmetrizeError

    monkeypatch.setattr(symmetrize_dialog, "SymmetrizeDialog", _AcceptedDialog)

    def boom(*a, **k):
        raise SymmetrizeError("deliberate failure")

    monkeypatch.setattr(matrix_panel, "symmetrize", boom)
    messages = []
    monkeypatch.setattr("matrix_panel.QMessageBox.warning",
                        lambda parent, title, text, *a, **k: messages.append(text))
    window, panel = _panel(qapp, _asymmetric())
    before = len(window._matrix_panels)

    panel._open_symmetrize_dialog()

    assert messages == ["deliberate failure"]
    assert len(window._matrix_panels) == before
    panel.close(); window.close()


def _flush_deletes(qapp):
    """Run the DeferredDelete events deleteLater posted, then collect."""
    import gc

    from PySide6.QtCore import QEvent

    for _ in range(3):
        qapp.processEvents()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()


def test_closing_both_panels_frees_both_matrices(qapp, monkeypatch):
    """Through the REAL dialog path (worker thread + progress dialog), the
    source matrix and the folded one must die with their panels. They
    used to stay alive: the worker, parented to the source panel and never
    deleted, held the source matrix, and its signal closures held S --
    4 GiB still resident after both 16384^2 panels were closed."""
    import weakref

    import symmetrize_dialog

    monkeypatch.setattr(symmetrize_dialog, "SymmetrizeDialog", _AcceptedDialog)
    m = _asymmetric()
    window, panel = _panel(qapp, m)
    source = weakref.ref(panel.matrix)
    del m

    panel._open_symmetrize_dialog()
    new = window._matrix_panels[-1]
    folded = weakref.ref(new.matrix)
    assert folded() is not source()

    new.close()
    panel.close()
    del new, panel
    _flush_deletes(qapp)

    assert folded() is None, "the symmetrized matrix outlived its panel"
    assert source() is None, "the source matrix outlived its panel"
    window.close()


def test_opening_a_matrix_leaves_no_worker_behind(qapp, tmp_path):
    """File > Open Matrix's loader thread is deleted once it is done,
    rather than accumulating under the main window for every file."""
    from main_window import _MatrixLoadWorker

    window = MainWindow()
    path = os.path.join(FIXTURES, sorted(
        f for f in os.listdir(FIXTURES) if f.endswith(".json"))[0])
    matrix, error, _axes = window._load_matrix_with_progress(path)
    assert error is None and matrix is not None
    _flush_deletes(qapp)
    assert not window.findChildren(_MatrixLoadWorker)
    window.close()
