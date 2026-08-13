import functools
import os

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import Qt

import matrix_panel
from main_window import MainWindow
from matrix_panel import MatrixPanel

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Every test in this file constructs a fresh MatrixPanel (needed for
# state isolation between tests), and MatrixPanel.__init__ decodes the
# real 8192x8192 gg.mtx fixture via load_mtx every time -- several
# seconds each. Without sharing decoded results across tests, this
# file's runtime balloons linearly with test count (12 tests ->
# ~190s). Same fix as test_mtx_io.py's own caching: memoize load_mtx
# by path and monkeypatch it in for every test, since all tests here
# use the same fixture path and load_mtx is a pure function of it.
_cached_load_mtx = functools.lru_cache(maxsize=None)(matrix_panel.load_mtx)


@pytest.fixture(autouse=True)
def _use_cached_load_mtx(monkeypatch):
    monkeypatch.setattr(matrix_panel, "load_mtx", _cached_load_mtx)


def test_matrix_panel_loads_matrix_and_computes_both_projections(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.matrix.shape == (8192, 8192)
    assert len(panel.projections["x"]) == 8192
    assert len(panel.projections["y"]) == 8192


def test_matrix_panel_defaults_to_x_projection(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.working_axis == "x"


def test_matrix_panel_switching_axis_updates_working_axis(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel.axis_selector.setCurrentIndex(1)

    assert panel.working_axis == "y"


def test_matrix_panel_canvas_accepts_keyboard_focus(qapp):
    # Regression guard: a bare matplotlib canvas defaults to
    # Qt.FocusPolicy.NoFocus, which structurally cannot receive
    # QKeyEvents -- MatrixCutController's eventFilter KeyPress/KeyRelease
    # branch would be dead code in real usage (holding C/B while hovering
    # the plot would never set _held_key) without an explicit
    # setFocusPolicy upgrade, matching FitModeController's own
    # canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus) (fit_mode.py:405).
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.canvas.focusPolicy() != Qt.FocusPolicy.NoFocus


def _click(panel, xdata, ydata=10.0):
    ax = panel.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", panel.canvas, px, py, button=1)
    panel.canvas.callbacks.process("button_press_event", event)


def _held_key_click(panel, key, xdata, ydata=10.0):
    panel.cut_controller._held_key = key
    _click(panel, xdata, ydata)
    panel.cut_controller._held_key = None


def test_matrix_panel_marking_cut_region_with_two_clicks(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 200.0)

    # pytest.approx, not ==: a simulated click round-trips xdata through
    # ax.transData's forward (data->pixel) then inverse (pixel->data)
    # transform, which is not bit-exact in floating point -- the same
    # tolerant-comparison convention test_fit_mode_ui.py already uses for
    # every click-derived region assertion (e.g.
    # "fit_region == pytest.approx((85.0, 115.0))").
    assert panel.cut_controller.state.cut_region == pytest.approx((100.0, 200.0))


def test_matrix_panel_marking_background_region_with_two_clicks(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "bg", 300.0)
    _held_key_click(panel, "bg", 350.0)

    # Compare the single tuple, not the whole list, via pytest.approx --
    # pytest.approx on a list containing tuples does not recurse into the
    # nested tuples (confirmed: only element-vs-element numeric comparison
    # is supported), matching the indexed-tuple comparison style
    # test_fit_mode_ui.py uses (e.g. "regions[0] == pytest.approx((70.0, 85.0))").
    assert len(panel.cut_controller.state.bg_regions) == 1
    assert panel.cut_controller.state.bg_regions[0] == pytest.approx((300.0, 350.0))


def test_matrix_panel_multiple_background_regions_allowed(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "bg", 300.0)
    _held_key_click(panel, "bg", 350.0)
    _held_key_click(panel, "bg", 500.0)
    _held_key_click(panel, "bg", 550.0)
    _held_key_click(panel, "bg", 700.0)
    _held_key_click(panel, "bg", 750.0)

    assert len(panel.cut_controller.state.bg_regions) == 3


def test_matrix_panel_activate_cut_disabled_without_cut_region(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.activate_cut_button.isEnabled() is False


def test_matrix_panel_activate_cut_enabled_once_cut_region_marked(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 200.0)

    assert panel.activate_cut_button.isEnabled() is True


def test_matrix_panel_activate_cut_adds_spectrum_to_main_window(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 300.0)
    before = len(main_window.spectra)

    panel._activate_cut()

    assert len(main_window.spectra) == before + 1
    added = main_window.spectra[-1]
    assert added.active is True
    assert len(added.data) == 8192  # Y-indexed result (gated on X projection)


def test_matrix_panel_activate_cut_result_matches_direct_computation(qapp):
    from matrix_cut import compute_cut

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 300.0)
    _held_key_click(panel, "bg", 400.0)
    _held_key_click(panel, "bg", 450.0)
    panel._activate_cut()

    expected = compute_cut(panel.matrix, "x", (100.0, 300.0), [(400.0, 450.0)])
    added = main_window.spectra[-1]
    assert added.data == pytest.approx(expected)


def test_matrix_panel_activate_cut_label_includes_working_axis(qapp):
    # An X-gated cut and a Y-gated cut at the same nominal region bounds
    # are physically different results (compute_cut indexes by the
    # complementary axis) -- the label must disambiguate them, especially
    # since gamma-gamma matrices (like this fixture) are often
    # near-symmetric.
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 300.0)
    panel._activate_cut()

    added = main_window.spectra[-1]
    assert panel.working_axis in added.path


def test_matrix_panel_heatmap_button_opens_heatmap_window(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel._open_heatmap()

    assert len(panel._heatmap_windows) == 1
    assert panel._heatmap_windows[0].isVisible()


def test_matrix_panel_heatmap_button_opening_twice_keeps_both_windows(qapp):
    # Regression guard for the actual bug: MatrixHeatmapWindow is
    # parentless and non-modal, so under PySide6's ownership rules the
    # Python-side reference held on the panel is what keeps it alive. A
    # single "self._heatmap_window = ..." attribute (rather than a list)
    # would be overwritten by a second open, dropping the first window's
    # only reference and leaving it eligible for garbage collection --
    # it could simply vanish. The list must retain both.
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel._open_heatmap()
    panel._open_heatmap()

    assert len(panel._heatmap_windows) == 2
    assert panel._heatmap_windows[0] is not panel._heatmap_windows[1]
    assert panel._heatmap_windows[0].isVisible()
    assert panel._heatmap_windows[1].isVisible()


def test_matrix_panel_projection_displayed_as_histogram(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    line = panel.axes.lines[0]
    assert line.get_drawstyle() == "steps-mid"


def test_matrix_panel_wraps_projection_as_single_spectrum(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert len(panel.spectra) == 1
    wrapped = panel.spectra[0]
    assert wrapped.active is True
    assert wrapped.visible is True
    assert wrapped.fits == []
    np.testing.assert_array_equal(wrapped.data, panel.projections["x"])


def test_matrix_panel_rebuilds_spectra_on_axis_switch(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel.axis_selector.setCurrentIndex(1)

    assert len(panel.spectra) == 1
    np.testing.assert_array_equal(panel.spectra[0].data, panel.projections["y"])
