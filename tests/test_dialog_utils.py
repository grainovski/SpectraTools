"""Reopening a window the user has already closed.

Reported against 6.0.2: the efficiency window did not appear the second
time. It is stored on an attribute of its owner AND created with
WA_DeleteOnClose, so closing it destroyed the C++ object while the
attribute kept the Python wrapper. The next open reached for that wrapper
to close the "previous" window and raised RuntimeError. Inside a Qt slot
in a --windowed build the traceback has no stderr to reach, so the window
silently never appeared again.

Three call sites shared the bug: the efficiency window opened from the
menu, the same window opened after a fit, and the calibration plot.
"""

import os

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QDialog

from calibration import Calibration
from calibration_plot_dialog import CalibrationPlotDialog
from dialog_utils import RaiseOnClickFilter, close_previous, window_to_raise
from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo
from main_window import MainWindow
from spectrum import LoadedSpectrum

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


@pytest.fixture(scope="module")
def result():
    data = np.loadtxt(os.path.join(FIXTURES, "demo1.txt"), ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    return EfficiencyResult(
        fit=fit, mc=run_monte_carlo(fit, N, dN, I, dI, iterations=200),
        model="kfr", calibration=Calibration("linear", 50.0, 0.65))


def _closed_dialog(qapp):
    """A dialog the user has closed, i.e. a wrapper with no C++ object."""
    d = QDialog()
    d.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    d.show()
    d.close()
    qapp.processEvents()
    return d


def test_the_scenario_really_does_destroy_the_dialog(qapp):
    """CONTROL. Every test below is only meaningful if closing a
    WA_DeleteOnClose dialog actually destroys its C++ object. If Qt ever
    stopped doing that, the reopen tests would pass for the wrong reason
    and this file would be guarding nothing."""
    dead = _closed_dialog(qapp)
    with pytest.raises(RuntimeError):
        dead.isVisible()


def test_close_previous_ignores_nothing_to_close(qapp):
    assert close_previous(None) is False


def test_close_previous_closes_a_live_dialog(qapp):
    d = QDialog()
    d.show()
    assert d.isVisible()
    assert close_previous(d) is True
    assert not d.isVisible()


def test_close_previous_tolerates_an_already_destroyed_dialog(qapp):
    """The whole point: this must not raise."""
    assert close_previous(_closed_dialog(qapp)) is False


def _press(widget):
    """A left mouse press delivered to `widget`."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(1.0, 1.0), QPointF(1.0, 1.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier)


class _Watched(QDialog):
    """Records raise_/activateWindow instead of asking the window manager,
    which does nothing useful under the offscreen platform the suite runs
    on."""

    def __init__(self, active=False):
        super().__init__()
        self.calls = []
        self._active = active

    def raise_(self):
        self.calls.append("raise_")

    def activateWindow(self):
        self.calls.append("activate")

    def isActiveWindow(self):
        return self._active


def test_clicking_a_child_widget_raises_its_window(qapp):
    """The case that decides the design. A click almost never lands on the
    window itself -- a matplotlib canvas, a button or a table cell eats it
    first -- so a per-window mousePressEvent would catch only clicks on
    bare background. The filter sits on the application and sees them all."""
    from PySide6.QtWidgets import QPushButton

    dialog = _Watched()
    button = QPushButton("inside", dialog)
    consumed = RaiseOnClickFilter(qapp).eventFilter(button, _press(button))

    assert dialog.calls == ["raise_", "activate"]
    assert consumed is False, "the filter must not swallow the click"
    dialog.close()


def test_an_already_active_window_is_not_raised_again(qapp):
    """Raising is a round trip to the window manager on every click."""
    dialog = _Watched(active=True)
    RaiseOnClickFilter(qapp).eventFilter(dialog, _press(dialog))
    assert dialog.calls == []
    dialog.close()


def test_a_click_while_a_popup_is_open_raises_nothing(qapp):
    """A menu or combo drop-down is its own window; raising the window
    underneath it closes it, which would make menus unusable."""
    dialog = _Watched()
    assert window_to_raise(dialog, popup=None) is dialog
    assert window_to_raise(dialog, popup=object()) is None
    dialog.close()


def test_non_mouse_events_and_non_widgets_are_ignored(qapp):
    """The filter sees every event in the application."""
    from PySide6.QtCore import QObject as _Plain

    dialog = _Watched()
    filt = RaiseOnClickFilter(qapp)
    filt.eventFilter(dialog, QEvent(QEvent.Type.Paint))
    assert dialog.calls == []
    # An event can reach an object that is not a widget at all.
    assert window_to_raise(None) is None
    filt.eventFilter(_Plain(), _press(dialog))
    assert dialog.calls == []
    dialog.close()


def test_the_raise_check_can_fail(qapp):
    """Control. Every assertion above would hold for a filter that raised
    nothing, ever -- so pin that it does raise in the ordinary case."""
    dialog = _Watched()
    RaiseOnClickFilter(qapp).eventFilter(dialog, _press(dialog))
    assert dialog.calls == ["raise_", "activate"], (
        "nothing was raised, so the tests above prove nothing")
    dialog.close()


def _points(n=8):
    return [(100.0 * (i + 1), 0.1, 5000.0 - 300.0 * i, 70.0,
             50.0 * (i + 1)) for i in range(n)]


class _Line:
    def __init__(self, energy):
        self.energy = energy
        self.energy_err = 0.01
        self.intensity = 1000.0
        self.intensity_err = 10.0


def test_the_efficiency_window_reopens_after_a_fit(qapp, result):
    """The reported bug, on the path that produced it."""
    plot = CalibrationPlotDialog(
        None, Calibration("linear", 0.0, 0.5), _points(),
        [_Line(50.0 * (i + 1)) for i in range(8)], 4095, "out.txt")

    plot._show_efficiency(result)
    first = plot._efficiency_dialog
    assert first.isVisible()

    first.close()                      # the user closes it
    qapp.processEvents()

    plot._show_efficiency(result)      # used to raise RuntimeError
    assert plot._efficiency_dialog.isVisible()
    assert plot._efficiency_dialog is not first

    plot._efficiency_dialog.close()
    plot.close()


def _window():
    window = MainWindow()
    window._calibration = Calibration("linear", 50.0, 0.65)
    window._calibration_active = True
    s = LoadedSpectrum("a.txt", np.full(256, 1000.0), "#FF0000")
    s.active = True
    window.spectra.append(s)
    return window


def test_the_efficiency_window_reopens_from_the_menu(qapp, result):
    """Same bug, reached through Operations rather than a fit."""
    window = _window()
    window.set_efficiency(result)

    window.show_efficiency()
    first = window._efficiency_dialog
    assert first.isVisible()

    first.close()
    qapp.processEvents()

    window.show_efficiency()           # used to raise RuntimeError
    assert window._efficiency_dialog.isVisible()
    assert window._efficiency_dialog is not first

    window._efficiency_dialog.close()
    window.close()


def test_the_calibration_plot_reopens_after_being_closed(qapp):
    """The third site sharing the defect. Its own comment says it closes
    the previous window so a second calibration cannot leave the first on
    screen -- correct, and it broke once the user had closed it first."""
    window = _window()
    spectrum = window.spectra[0]
    calibration = Calibration("linear", 0.0, 0.5)
    lines = [_Line(50.0 * (i + 1)) for i in range(8)]

    window._show_calibration_plot(spectrum, calibration, _points(), lines, ())
    first = window._calibration_plot
    assert first.isVisible()

    first.close()
    qapp.processEvents()

    window._show_calibration_plot(spectrum, calibration, _points(), lines, ())
    assert window._calibration_plot.isVisible()
    assert window._calibration_plot is not first

    window._calibration_plot.close()
    window.close()
