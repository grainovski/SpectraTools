"""Go To (v4.1.1): jump the view to one energy or channel and mark it."""

import functools
import os

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLabel, QMessageBox

import matrix_panel
from calibration import Calibration
from goto_dialog import GoToDialog
from main_window import MainWindow
from matrix_panel import MatrixPanel
from spectrum import GOTO_WINDOW_CHANNELS, LoadedSpectrum, goto_channel_window

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# MatrixPanel.__init__ decodes the real 8192x8192 gg.mtx every time -- several
# seconds each, which this file would pay five times over. Memoised and patched
# in exactly as test_matrix_panel.py does, since every test here uses the same
# fixture path and load_mtx is a pure function of it.
_cached_load_mtx = functools.lru_cache(maxsize=None)(matrix_panel.load_mtx)


@pytest.fixture(autouse=True)
def _use_cached_load_mtx(monkeypatch):
    monkeypatch.setattr(matrix_panel, "load_mtx", _cached_load_mtx)


@pytest.fixture
def warnings(monkeypatch):
    """Collects QMessageBox.warning calls instead of showing them.

    Not optional plumbing: the real warning is MODAL and blocks forever
    under the offscreen platform, so a test that trips a validation error
    without this hangs rather than fails. Same approach as
    test_factor_dialog.py.
    """
    seen = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: seen.append(a))
    )
    return seen


def _window_with_peak(channels=2048, peak_channel=1200):
    main_window = MainWindow()
    data = np.full(channels, 50, dtype=np.int64)
    data[peak_channel] = 9000
    spectrum = LoadedSpectrum("s.spk", data, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()
    return main_window, spectrum


def _dotted_lines(axes):
    return [line for line in axes.get_lines() if line.get_linestyle() == ":"]


def _press(key, modifier):
    return QKeyEvent(QEvent.Type.KeyPress, key, modifier)


# --------------------------------------------------------------- pure logic


def test_window_is_centred_and_the_requested_width():
    lo, hi = goto_channel_window(500.0, 8191)
    assert (lo, hi) == (450.0, 550.0)
    assert hi - lo == GOTO_WINDOW_CHANNELS


def test_window_near_the_start_shifts_rather_than_truncating():
    """A line at channel 5 must still be shown with context around it, not
    pinned against the edge with half a window."""
    lo, hi = goto_channel_window(5.0, 8191)
    assert (lo, hi) == (0.0, 100.0)
    assert hi - lo == GOTO_WINDOW_CHANNELS


def test_window_near_the_end_shifts_too():
    lo, hi = goto_channel_window(8190.0, 8191)
    assert hi == 8191.0
    assert hi - lo == GOTO_WINDOW_CHANNELS


def test_window_narrower_than_the_request_only_when_the_data_is():
    lo, hi = goto_channel_window(3.0, 10)
    assert (lo, hi) == (0.0, 10.0)


def test_window_rejects_a_negative_max_channel():
    with pytest.raises(ValueError):
        goto_channel_window(0.0, -1)


# ------------------------------------------------------------------ dialog


def test_dialog_asks_for_energy_when_calibrated(qapp):
    dialog = GoToDialog(None, True, 10.0, 1000.0)
    assert "keV" in dialog._unit
    dialog._field.setText("500")
    dialog._on_accept()
    assert dialog.result_value == pytest.approx(500.0)


def test_dialog_asks_for_a_channel_when_not_calibrated(qapp):
    dialog = GoToDialog(None, False, 0.0, 2047.0)
    assert dialog._unit == "channel"
    dialog._field.setText("1200")
    dialog._on_accept()
    assert dialog.result_value == pytest.approx(1200.0)


def test_dialog_refuses_a_value_outside_the_spectrum(qapp, warnings):
    dialog = GoToDialog(None, False, 0.0, 2047.0)
    dialog._field.setText("5000")
    dialog._on_accept()
    assert dialog.result_value is None
    assert warnings, "the user must be told why it was refused"
    assert "2047" in " ".join(str(a) for a in warnings[0])


@pytest.mark.parametrize("text", ["", "abc", "12x", "nan", "inf", "-inf"])
def test_dialog_refuses_unusable_text(qapp, warnings, text):
    """float() accepts nan/inf happily, and either would send the view
    somewhere it could not come back from."""
    dialog = GoToDialog(None, False, 0.0, 2047.0)
    dialog._field.setText(text)
    dialog._on_accept()
    assert dialog.result_value is None
    assert warnings


def test_dialog_shows_the_range_before_the_user_types(qapp):
    dialog = GoToDialog(None, True, 10.0, 1034.5)
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert any("10" in t and "1034.5" in t and "keV" in t for t in labels)


# ------------------------------------------------------------- main window


def test_goto_centres_the_view_on_a_channel(qapp):
    main_window, _ = _window_with_peak()
    main_window.goto_display_value(1200)
    lo, hi = main_window.axes.get_xlim()
    assert (lo + hi) / 2 == pytest.approx(1200.0)
    assert hi - lo == pytest.approx(GOTO_WINDOW_CHANNELS)


def test_goto_centres_on_the_right_channel_when_calibrated(qapp):
    """The user types keV; the window must still be exactly
    GOTO_WINDOW_CHANNELS wide and centred on the matching channel."""
    main_window, _ = _window_with_peak()
    main_window._apply_calibration_change(
        Calibration(kind="linear", a=10.0, b=0.5), True
    )
    main_window.goto_display_value(610.0)          # E = 10 + 0.5*1200
    lo, hi = main_window.axes.get_xlim()
    channel_lo = main_window.display_to_channel(lo)
    channel_hi = main_window.display_to_channel(hi)
    assert (channel_lo + channel_hi) / 2 == pytest.approx(1200.0, abs=1e-6)
    assert channel_hi - channel_lo == pytest.approx(GOTO_WINDOW_CHANNELS, abs=1e-6)


def test_goto_draws_a_marker_at_the_target(qapp):
    main_window, _ = _window_with_peak()
    main_window.goto_display_value(1200)
    marks = _dotted_lines(main_window.axes)
    assert len(marks) == 1
    assert marks[0].get_xdata()[0] == pytest.approx(1200.0)


def test_the_marker_follows_a_calibration_change_to_the_same_channel(qapp):
    """Stored in channels, not display units, so switching the axis to keV
    leaves the mark on the same physical spot rather than at a stale
    coordinate."""
    main_window, _ = _window_with_peak()
    main_window.goto_display_value(1200)
    main_window._apply_calibration_change(
        Calibration(kind="linear", a=10.0, b=0.5), True
    )
    marks = _dotted_lines(main_window.axes)
    assert len(marks) == 1
    assert marks[0].get_xdata()[0] == pytest.approx(610.0)


def test_goto_autoscales_y_to_what_it_jumped_to(qapp):
    main_window, _ = _window_with_peak()
    main_window.goto_display_value(1200)
    assert main_window.axes.get_ylim()[1] > 9000


def test_clear_removes_the_goto_marker(qapp):
    main_window, _ = _window_with_peak()
    main_window.goto_display_value(1200)
    main_window.fit_controller.clear()
    assert main_window._goto_marker_channel is None
    assert _dotted_lines(main_window.axes) == []


def test_goto_without_a_spectrum_says_so_rather_than_raising(qapp):
    main_window = MainWindow()
    main_window.open_goto_dialog()
    assert "spectrum" in main_window.statusBar().currentMessage().lower()


def test_goto_is_on_ctrl_g_and_log_scale_moved_to_ctrl_y(qapp):
    main_window = MainWindow()
    assert main_window.goto_action.shortcut().toString() == "Ctrl+G"
    assert main_window.log_scale_action.shortcut().toString() == "Ctrl+Y"
    # Calibration keeps Ctrl+L; nothing else had to move.
    assert main_window.calibration_action.shortcut().toString() == "Ctrl+L"


# ------------------------------------------------------------ matrix panel


def test_the_projection_has_go_to_on_the_same_key(qapp):
    """A projection is a spectrum view; the gesture must not differ."""
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    assert panel.goto_action.shortcut().toString() == "Ctrl+G"


def test_goto_works_in_the_projection(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    target = len(panel.spectra[0].data) // 2
    panel.goto_display_value(target)
    lo, hi = panel.axes.get_xlim()
    assert (lo + hi) / 2 == pytest.approx(float(target))
    assert len(_dotted_lines(panel.axes)) == 1


# -------------------------------------- modifier guard (regression, v4.1.1)


@pytest.mark.parametrize("key,expected", [
    (Qt.Key.Key_B, "b"), (Qt.Key.Key_R, "r"), (Qt.Key.Key_P, "p"),
])
def test_a_bare_marking_key_still_arms_marking(qapp, key, expected):
    main_window, _ = _window_with_peak()
    main_window.fit_controller._held_key = None
    main_window.fit_controller.eventFilter(
        main_window.canvas, _press(key, Qt.KeyboardModifier.NoModifier)
    )
    assert main_window.fit_controller._held_key == expected


@pytest.mark.parametrize("key", [Qt.Key.Key_B, Qt.Key.Key_R, Qt.Key.Key_P])
def test_ctrl_plus_a_marking_key_does_not_arm_marking(qapp, key):
    """Ctrl+B (Preview Background Fit) and Ctrl+R (Rebin) share letters with
    marking keys. Both used to arm marking as well as firing the shortcut,
    and because the shortcut opens a dialog that takes focus, the KeyRelease
    that would disarm it never arrived -- so the next ordinary click placed
    a mark nobody asked for."""
    main_window, _ = _window_with_peak()
    main_window.fit_controller._held_key = None
    main_window.fit_controller.eventFilter(
        main_window.canvas, _press(key, Qt.KeyboardModifier.ControlModifier)
    )
    assert main_window.fit_controller._held_key is None


@pytest.mark.parametrize("key,expected", [
    (Qt.Key.Key_C, "cut"), (Qt.Key.Key_G, "gate_bg"),
])
def test_panel_bare_keys_still_arm_cut_marking(qapp, key, expected):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.cut_controller._held_key = None
    panel.cut_controller.eventFilter(
        panel.canvas, _press(key, Qt.KeyboardModifier.NoModifier)
    )
    assert panel.cut_controller._held_key == expected


@pytest.mark.parametrize("key", [Qt.Key.Key_C, Qt.Key.Key_G])
def test_panel_ctrl_keys_do_not_arm_cut_marking(qapp, key):
    """Ctrl+C is Clear and Ctrl+G is Go To; neither may arm a gate."""
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.cut_controller._held_key = None
    panel.cut_controller.eventFilter(
        panel.canvas, _press(key, Qt.KeyboardModifier.ControlModifier)
    )
    assert panel.cut_controller._held_key is None
