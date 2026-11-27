"""The Matrix panel with a random-subtracted result: float64 cells,
fractional, many of them negative (S = P - w R from Sort List-Mode γγ
Data). Driven as the user drives it -- clicks with the C / G keys held,
Activate Cut, the log-scale action, Show Heatmap..., Symmetrize... --
through a panel opened by the main window, the way step 5 opens one
(v7 plan, Task 6: verify the panel before relying on it).
"""

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from main_window import MainWindow
from matrix_symmetrize import SymmetrizeError, check_symmetrizable, rebin_rows

N = 256


def _subtracted(seed=4, w=1 / 3):
    """P - w R on a sparse grid: a prompt line pair on a random floor,
    so plenty of cells (and whole projection channels) are negative."""
    rng = np.random.default_rng(seed)
    P = np.zeros((N, N))
    R = np.zeros((N, N))
    for _ in range(2000):
        P[rng.integers(N), rng.integers(N)] += 1
    for _ in range(8000):                   # w R outweighs P's floor
        R[rng.integers(N), rng.integers(N)] += 1
    P[60, 140] += 500                       # Y = 60, X = 140
    P[60, 141] += 200
    P[90, 140] += 300
    return P - w * R


@pytest.fixture
def panel(qapp):
    window = MainWindow()
    matrix = _subtracted()
    p = window._open_matrix_panel("C:/runs/Run_A/Run_A_gg_sub.json",
                                  matrix=matrix, axes=(None, None))
    yield window, p, matrix
    p.close()
    window.close()


def _click(panel, xdata, ydata=1.0):
    px, py = panel.axes.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", panel.canvas, px, py, button=1)
    panel.canvas.callbacks.process("button_press_event", event)


def _held_key_click(panel, key, xdata):
    panel.cut_controller._held_key = key
    _click(panel, xdata)
    panel.cut_controller._held_key = None


def test_projections_keep_the_exact_signed_floats(panel):
    window, p, m = panel
    assert p.matrix.dtype == np.float64
    np.testing.assert_allclose(p.projections["x"], m.sum(axis=0), atol=1e-9)
    np.testing.assert_allclose(p.projections["y"], m.sum(axis=1), atol=1e-9)
    assert (p.projections["x"] < 0).any()
    # What the panel draws is that projection, not a rounded copy.
    line = p.axes.get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), m.sum(axis=0), atol=1e-9)
    p.axis_selector.setCurrentIndex(1)
    np.testing.assert_allclose(p.axes.get_lines()[0].get_ydata(), m.sum(axis=1),
                               atol=1e-9)


def test_a_cut_marked_with_clicks_reaches_the_main_window_exactly(panel):
    window, p, m = panel
    _held_key_click(p, "cut", 139.0)
    _held_key_click(p, "cut", 142.0)
    _held_key_click(p, "gate_bg", 180.0)
    _held_key_click(p, "gate_bg", 199.0)
    assert p.activate_cut_button.isEnabled()
    p.activate_cut_button.click()

    cut = window.spectra[-1]
    gate = m[:, 139:143].sum(axis=1)
    bg = m[:, 180:200].sum(axis=1)
    expected = gate - (4 / 20) * bg
    np.testing.assert_allclose(cut.data, expected, atol=1e-9)
    assert (cut.data < 0).any()                      # never clamped
    assert np.all(np.isfinite(cut.variance)) and np.all(cut.variance >= 0)
    # Gated on X at 140: the Y = 60 line is the cut's peak.
    assert int(np.argmax(cut.data)) == 60


def test_log_scale_shows_a_projection_with_negative_channels(panel):
    window, p, m = panel
    p.log_scale_action.setChecked(True)
    assert p.axes.get_yscale() == "log"
    lo, hi = p.axes.get_ylim()
    assert 0 < lo < hi and np.isfinite(hi)
    assert hi >= m.sum(axis=0).max()
    p.log_scale_action.setChecked(False)
    assert p.axes.get_yscale() == "linear"
    lo, hi = p.axes.get_ylim()
    assert lo <= m.sum(axis=0).min()


def test_heatmap_log_colour_scale_spans_the_negative_cells(panel):
    from matplotlib.colors import SymLogNorm
    window, p, m = panel
    p.heatmap_button.click()
    heatmap = p._heatmap_windows[-1]
    norm = heatmap.image.norm
    assert isinstance(norm, SymLogNorm)
    assert norm.vmin == pytest.approx(m.min()) and norm.vmin < 0
    assert norm.vmax == pytest.approx(m.max())
    assert np.all(np.isfinite(norm(np.array([m.min(), 0.0, m.max()]))))
    heatmap.close()


def test_symmetrize_refuses_negative_cells_and_says_why(panel, monkeypatch):
    window, p, m = panel
    messages = []
    monkeypatch.setattr("matrix_panel.QMessageBox.information",
                        lambda parent, title, text, *a, **k: messages.append((title, text)))
    opened = []
    monkeypatch.setattr("symmetrize_dialog.SymmetrizeDialog.exec",
                        lambda self: opened.append(self) or 0)
    before = len(window._matrix_panels)
    p.symmetrize_button.click()
    assert not opened, "the Symmetrize dialog must not open for a subtracted matrix"
    assert len(window._matrix_panels) == before
    (title, text), = messages
    assert title == "Cannot symmetrize"
    assert "negative counts" in text
    assert f"{int((m < 0).sum()):,} cells" in text
    assert "Sort List-Mode" in text


def test_symmetrize_refuses_fractional_cells_instead_of_truncating(qapp, monkeypatch):
    """A scaled or subtracted matrix without a negative cell (a busy
    compressed result) used to be folded with every cell cast to int64:
    0.7 + 0.7 became 0. Now it is refused, like the negative case."""
    window = MainWindow()
    m = np.zeros((16, 16))
    m[3, 5] = 2.7
    m[5, 3] = 0.4
    m[7, 9] = 1.0
    p = window._open_matrix_panel("frac.json", matrix=m)
    messages = []
    monkeypatch.setattr("matrix_panel.QMessageBox.information",
                        lambda parent, title, text, *a, **k: messages.append(text))
    opened = []
    monkeypatch.setattr("symmetrize_dialog.SymmetrizeDialog.exec",
                        lambda self: opened.append(self) or 0)
    p.symmetrize_button.click()
    assert not opened, "the Symmetrize dialog opened for a fractional matrix"
    assert messages and "fractional" in messages[0], messages
    with pytest.raises(SymmetrizeError, match="fractional"):
        p._symmetrize_with(0.0, 1.0)
    with pytest.raises(SymmetrizeError, match="Fractional"):
        rebin_rows(m, 0.5, 1.01)
    p.close()
    window.close()


def test_whole_valued_floats_still_symmetrize(qapp):
    """A float matrix holding whole counts (a JSON read back as float) is
    raw data and folds as before."""
    m = np.zeros((16, 16))
    m[3, 5] = 4.0
    m[7, 9] = 1.0
    check_symmetrizable(m)
    window = MainWindow()
    p = window._open_matrix_panel("whole.json", matrix=m)
    new = p._symmetrize_with(0.0, 1.0)
    assert new.matrix[3, 5] == 4 and new.matrix[5, 3] == 4
    new.close()
    p.close()
    window.close()
