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

    _held_key_click(panel, "gate_bg", 300.0)
    _held_key_click(panel, "gate_bg", 350.0)

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

    _held_key_click(panel, "gate_bg", 300.0)
    _held_key_click(panel, "gate_bg", 350.0)
    _held_key_click(panel, "gate_bg", 500.0)
    _held_key_click(panel, "gate_bg", 550.0)
    _held_key_click(panel, "gate_bg", 700.0)
    _held_key_click(panel, "gate_bg", 750.0)

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
    _held_key_click(panel, "gate_bg", 400.0)
    _held_key_click(panel, "gate_bg", 450.0)
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


def test_matrix_panel_calibration_shared_with_main_window(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    cal = Calibration(kind="linear", a=1.0, b=2.0, c=0.0)

    panel._calibration = cal
    panel._calibration_active = True

    assert main_window._calibration is cal
    assert main_window._calibration_active is True
    assert panel.channel_to_display(100) == pytest.approx(cal.apply(100))
    assert panel.display_to_channel(201.0) == pytest.approx(cal.invert(201.0))


def test_matrix_panel_channel_to_display_passthrough_when_uncalibrated(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.channel_to_display(150) == 150
    assert panel.display_to_channel(150) == 150


def test_matrix_panel_calibrate_button_opens_dialog_and_applies_result(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog
    from PySide6.QtWidgets import QDialog

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    result = Calibration(kind="linear", a=0.0, b=1.5, c=0.0)

    def fake_exec(self):
        self.result_calibration = result
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)

    panel._open_calibration_dialog()

    assert main_window._calibration is result
    assert main_window._calibration_active is True


def test_matrix_panel_calibration_change_correctly_updates_main_window_view(qapp):
    # Regression guard: an earlier version of MatrixPanel._apply_calibration_change
    # called main_window._plot_data(preserve_view=True) directly, which reuses
    # main_window.axes.get_xlim()'s raw numbers verbatim -- captured AFTER the
    # calibration had already been overwritten, so those numbers carry no memory
    # of what they meant under the OLD calibration. That silently shows the wrong
    # region once the axis units change (identity-calibration tests couldn't catch
    # this, since naive reuse is coincidentally correct when b=1.0). The fix
    # delegates to main_window's own _apply_calibration_change, which round-trips
    # the view through channel space computed from the OLD calibration before
    # overwriting it -- this test uses a non-identity calibration and a real,
    # specific zoom to actually prove the round-trip, not just that a redraw
    # happened.
    from calibration import Calibration
    from spectrum import LoadedSpectrum

    main_window = MainWindow()
    data = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", data, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()
    main_window.axes.set_xlim(20.0, 40.0)

    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel._apply_calibration_change(Calibration(kind="linear", a=0.0, b=3.0, c=0.0), True)

    assert main_window.axes.get_xlabel() == "Energy (keV)"
    assert main_window.axes.get_xlim() == pytest.approx((60.0, 120.0))


def test_main_window_calibration_change_refreshes_open_matrix_panels(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    # Construct via _open_matrix_panel, not MatrixPanel(...) directly --
    # that's what actually registers the panel in main_window._matrix_panels,
    # which is what MainWindow._apply_calibration_change iterates over to
    # decide which panels to refresh.
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel.axes.set_xlabel("stale label")

    main_window._apply_calibration_change(Calibration(kind="linear", a=0.0, b=1.0, c=0.0), True)

    assert panel.axes.get_xlabel() != "stale label"


def test_matrix_panel_calibration_change_propagates_to_sibling_panels(qapp):
    # main_window._apply_calibration_change (which MatrixPanel now delegates
    # to) redraws every panel in main_window._matrix_panels, not just the one
    # that initiated the change -- a second, sibling matrix panel must also
    # pick up the update.
    from calibration import Calibration

    main_window = MainWindow()
    panel_a = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b.axes.set_xlabel("stale label")

    panel_a._apply_calibration_change(Calibration(kind="linear", a=0.0, b=1.0, c=0.0), True)

    assert panel_b.axes.get_xlabel() != "stale label"


def test_matrix_panel_cut_marks_stored_in_channel_space(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel._calibration = Calibration(kind="linear", a=0.0, b=2.0, c=0.0)
    panel._calibration_active = True

    _held_key_click(panel, "cut", 200.0)
    _held_key_click(panel, "cut", 400.0)

    # Clicked at display (keV) positions 200/400 with b=2.0 -- channel
    # space is display / 2, so the stored region should be (100, 200),
    # not the raw display values (200, 400).
    assert panel.cut_controller.state.cut_region == pytest.approx((100.0, 200.0))


def test_matrix_panel_pending_cut_click_shows_dashed_preview_line(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 150.0)

    lines = [a for a in panel.cut_controller._artists if hasattr(a, "get_linestyle")]
    assert any(line.get_linestyle() == "--" for line in lines)


def test_matrix_panel_pending_gate_bg_click_shows_dashed_preview_line(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "gate_bg", 300.0)

    lines = [a for a in panel.cut_controller._artists if hasattr(a, "get_linestyle")]
    assert any(line.get_linestyle() == "--" for line in lines)


def test_matrix_panel_plot_data_draws_histogram_with_calibration_aware_axis(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel._calibration = Calibration(kind="linear", a=0.0, b=2.0, c=0.0)
    panel._calibration_active = True

    panel._plot_data()

    line = panel.axes.lines[0]
    assert line.get_drawstyle() == "steps-mid"
    assert panel.axes.get_xlabel() == "Energy (keV)"


def test_matrix_panel_plot_data_preserves_view_when_requested(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.axes.set_xlim(100, 500)

    panel._plot_data(preserve_view=True)

    assert panel.axes.get_xlim() == pytest.approx((100, 500))


def test_matrix_panel_calibrating_preserves_the_current_view(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.axes.set_xlim(100, 500)

    panel._apply_calibration_change(Calibration(kind="linear", a=0.0, b=2.0, c=0.0), True)

    # View was (100, 500) in channel space; after a b=2.0 calibration
    # the same channel window should now read as (200, 1000) in keV.
    assert panel.axes.get_xlim() == pytest.approx((200.0, 1000.0))


def test_matrix_panel_has_trimmed_toolbar_and_zoom_actions(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    action_texts = [a.text() for a in panel.nav_toolbar.actions()]
    assert "Zoom" not in action_texts  # stock rectangle-zoom is trimmed out
    assert panel.zoom_in_action.shortcut().toString() == "Ctrl+="
    assert panel.zoom_out_action.shortcut().toString() == "Ctrl+-"
    assert panel.full_view_action.shortcut().toString() == "Ctrl+0"


def test_matrix_panel_zoom_in_narrows_xlim(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    full_lo, full_hi = panel.axes.get_xlim()
    full_width = full_hi - full_lo

    panel.zoom_in_action.trigger()

    new_lo, new_hi = panel.axes.get_xlim()
    assert (new_hi - new_lo) < full_width


def test_matrix_panel_full_view_action_resets_zoom(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    full_xlim = panel.axes.get_xlim()
    panel.axes.set_xlim(1000, 2000)

    panel.full_view_action.trigger()

    assert panel.axes.get_xlim() == pytest.approx(full_xlim)


def test_matrix_panel_scroll_event_zooms(qapp):
    from matplotlib.backend_bases import MouseEvent

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    full_lo, full_hi = panel.axes.get_xlim()
    full_width = full_hi - full_lo

    px, py = panel.axes.transData.transform((4096.0, 10.0))
    event = MouseEvent("scroll_event", panel.canvas, px, py, button="up")
    panel.canvas.callbacks.process("scroll_event", event)

    new_lo, new_hi = panel.axes.get_xlim()
    assert (new_hi - new_lo) < full_width


def test_matrix_panel_plot_data_autoscales_y(qapp, monkeypatch):
    # Regression guard: _plot_data originally set X limits but never
    # called _autoscale_y (unlike main_window's own _plot_data), so
    # zooming in on X left the Y axis at the full spectrum's count
    # range -- e.g. after a calibration change round-trips the X zoom
    # back into view, the peak the user zoomed in on would still look
    # visually squashed against the full spectrum's tallest peak.
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    calls = []
    monkeypatch.setattr(panel, "_autoscale_y", lambda xlim: calls.append(xlim))

    panel._plot_data()

    assert calls != []


def test_matrix_panel_plot_data_updates_nav_toolbar_history(qapp, monkeypatch):
    # Regression guard: _plot_data never called nav_toolbar.update()/
    # push_current() (unlike main_window's own _plot_data), so the
    # toolbar's Home button kept a stale view recorded from before an
    # axis switch or calibration change -- clicking Home could reapply
    # a different projection's (or a different calibration's) raw
    # xlim/ylim numbers onto the current plot.
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    calls = []
    monkeypatch.setattr(panel.nav_toolbar, "push_current", lambda: calls.append(True))

    panel._plot_data()

    assert calls != []


def test_calibration_change_from_main_window_preserves_matrix_panel_zoom(qapp):
    # Regression guard: main_window._apply_calibration_change's
    # panel-refresh loop originally called panel._plot_data() with no
    # arguments, which always resets to full view -- a calibration
    # change made from MAIN WINDOW'S OWN dialog/toolbar (not from the
    # matrix panel) silently discarded any open panel's zoom, unlike
    # main_window's own view, which was already correctly preserved.
    from calibration import Calibration

    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel.axes.set_xlim(20.0, 40.0)

    main_window._apply_calibration_change(Calibration(kind="linear", a=0.0, b=3.0, c=0.0), True)

    assert panel.axes.get_xlim() == pytest.approx((60.0, 120.0))


def test_calibration_change_from_one_panel_preserves_sibling_panel_zoom(qapp):
    # Same gap as above, from the other direction: a calibration change
    # initiated from panel_a only round-tripped panel_a's OWN view (via
    # MatrixPanel._apply_calibration_change's own logic) -- any OTHER
    # open panel still went through the bare, view-resetting loop.
    from calibration import Calibration

    main_window = MainWindow()
    panel_a = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel_b.axes.set_xlim(20.0, 40.0)

    panel_a._apply_calibration_change(Calibration(kind="linear", a=0.0, b=3.0, c=0.0), True)

    assert panel_b.axes.get_xlim() == pytest.approx((60.0, 120.0))


def test_matrix_panel_theme_delegates_to_main_window(qapp):
    # Regression guard: FitModeController.draw_committed_fits reads
    # self.main_window._theme via getattr(..., "light") -- without a
    # _theme property on MatrixPanel, that always silently fell back to
    # "light" (MatrixPanel has no _theme attribute of its own), so every
    # committed fit/integration overlay in a dark-theme matrix panel was
    # drawn with light-theme colors tuned for the wrong background.
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    main_window._theme = "dark"

    assert panel._theme == "dark"


def test_matrix_panel_has_fit_and_parameters_docks(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.fit_controller.results_dock is not None
    assert panel.fit_controller.parameters_dock is not None
    assert panel.fit_controller.results_dock.parent() is panel
    assert panel.fit_controller.parameters_dock.parent() is panel


def test_matrix_panel_fit_button_disabled_without_fit_region(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.fit_button.isEnabled() is False


def test_matrix_panel_fitting_a_peak_on_the_projection_works(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    data = panel.spectra[0].data
    # Any two bare channel positions bracketing a stretch of real
    # projection data -- this is a real 8192-channel fixture, not
    # synthetic, so a real fit may or may not converge depending on
    # the exact region; the point of this test is that the SAME
    # FitModeController machinery MainWindow uses is reachable and
    # produces a fits-list entry, not that any specific region fits
    # cleanly. Mark a wide fit region and one peak position roughly
    # in its middle, matching how test_fit_mode_ui.py's own tests
    # mark fits.
    lo, hi = 100.0, 300.0

    panel.cut_controller._held_key = None  # not used by fit marking; ensure no interference
    panel.fit_controller._held_key = "r"
    _click(panel, lo)
    _click(panel, hi)
    panel.fit_controller._held_key = "p"
    _click(panel, (lo + hi) / 2)
    panel.fit_controller._held_key = None

    assert panel.fit_controller.state.fit_region == pytest.approx((lo, hi))
    assert len(panel.fit_controller.state.peak_positions) == 1


def test_matrix_panel_integrate_action_reachable(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.integrate_button.shortcut().toString() == "Ctrl+I"
    assert panel.fit_button.shortcut().toString() == "Ctrl+F"
    assert panel.clear_fit_button.shortcut().toString() == "Ctrl+C"
    assert panel.background_preview_button.shortcut().toString() == "Ctrl+B"
