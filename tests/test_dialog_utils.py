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


def test_an_already_active_window_is_still_raised(qapp):
    """The tempting optimisation, and why it is wrong.

    Skipping windows that are already active looks free. It is the
    opposite: the reported bug had "Calibrate from Fitted Peaks" ACTIVE
    and still underneath the plot, so active does not imply on top, and
    skipping active windows declines to raise precisely the window the
    user clicked to bring forward.
    """
    dialog = _Watched(active=True)
    RaiseOnClickFilter(qapp).eventFilter(dialog, _press(dialog))
    assert dialog.calls == ["raise_", "activate"], (
        "an active-but-buried window was skipped")
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


def _choices(n=6):
    """(channel, area, area_err, channel_err) per fitted peak."""
    return [(100.0 * (i + 1), 5000.0 - 300.0 * i, 70.0, 0.1) for i in range(n)]


def _energy_dialog():
    """A Calibrate-from-Fitted-Peaks dialog with its live plot open.

    The preview opens on the first assigned energy, not before, so the
    energies have to go in for there to be anything to test.
    """
    from energy_assign_dialog import EnergyAssignDialog

    dialog = EnergyAssignDialog(None, _choices(), max_channel=4095,
                                export_default_path="out.txt")
    for row in range(3):
        dialog.table.item(row, EnergyAssignDialog.ENERGY_COLUMN).setText(
            "%.1f" % (50.0 * (row + 1)))
    dialog._refresh_live_plot()
    return dialog


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


def _window_type(widget):
    return widget.windowFlags() & Qt.WindowType.WindowType_Mask


def test_qt_dialog_already_contains_the_window_bit(qapp):
    """CONTROL, and the reason the two tests below assert what they do.

    Qt::Dialog is Qt::Window plus one more bit, so the obvious
    setWindowFlag(Window, True) is a no-op on a dialog and would have
    shipped as a fix that changed nothing -- it did, until this was
    measured. setWindowFlag(Dialog, False) is worse: it leaves type Widget,
    which embeds the window inside its parent instead of floating it. Only
    masking the type bits off and replacing them works.
    """
    assert int(Qt.WindowType.Dialog) & int(Qt.WindowType.Window)

    parent = QDialog()
    naive = QDialog(parent)
    naive.setWindowFlag(Qt.WindowType.Window, True)
    assert _window_type(naive) == Qt.WindowType.Dialog, "no longer a no-op"

    worse = QDialog(parent)
    worse.setWindowFlag(Qt.WindowType.Dialog, False)
    assert _window_type(worse) == Qt.WindowType.Widget, "no longer embeds"
    parent.close()


def test_the_live_plot_has_no_parent_so_it_can_be_stacked(qapp):
    """The reported bug, and the only thing that actually fixes it.

    A widget parent makes the child window Win32-OWNED by the parent, and
    Windows keeps an owned window above its owner unconditionally.
    Measured on real windows: with the plot parented, raise_() on the
    dialog and lower() on the plot BOTH leave the plot on top, and
    changing the Qt window type does not help because ownership follows
    the widget parent rather than the type. Two shipped attempts failed
    on exactly that. Dropping the parent is what frees them.
    """
    dialog = _energy_dialog()
    plot = dialog._live_plot
    assert plot is not None, "no live plot -- the fixture stopped triggering it"

    assert plot.parent() is None, (
        "the plot is parented again and will be pinned above the dialog")
    assert plot.windowTitle() == "Energy Calibration"
    dialog._close_live_plot()
    dialog.close()


def test_the_dialog_is_window_modal_so_the_parentless_plot_stays_live(qapp):
    """The other half, and it is load-bearing.

    exec() makes a dialog APPLICATION-modal, which blocks every window in
    the app that is not one of its descendants -- and the plot is
    deliberately no longer a descendant. Measured: under ApplicationModal
    a parentless window is disabled outright, so clicking a point on the
    plot would do nothing and pointPicked would be dead. WindowModal still
    blocks the main window and leaves the plot alive.
    """
    from energy_assign_dialog import EnergyAssignDialog

    dialog = EnergyAssignDialog(None, _choices(), max_channel=4095,
                                export_default_path="out.txt")
    assert dialog.windowModality() == Qt.WindowModality.WindowModal, (
        "application-modal again -- the parentless plot would be dead")
    dialog.close()


def test_closing_the_dialog_closes_the_plot_it_opened(qapp):
    """The plot has no parent now, so Qt no longer destroys it with the
    dialog. done() has to, or a preview outlives what it belongs to."""
    dialog = _energy_dialog()
    assert dialog._live_plot is not None
    dialog.done(0)                       # Cancel, Escape and the X all land here
    assert dialog._live_plot is None, "the preview was left behind"


def test_the_efficiency_window_opened_from_the_plot_is_unowned(qapp, result):
    """The same defect one level further in. The efficiency window is
    opened by the calibration plot; owned by it, Windows pinned it above
    the plot for good. It is parentless now, and reaches the main window
    through an explicit reference instead of the parent chain."""
    main = _window()
    plot = CalibrationPlotDialog(
        None, Calibration("linear", 0.0, 0.5), _points(),
        [_Line(50.0 * (i + 1)) for i in range(8)], 4095, "out.txt",
        main_window=main)
    plot._show_efficiency(result)
    window = plot._efficiency_dialog

    assert window.parent() is None, "owned again, so it would pin itself"
    assert window._main_window() is main, (
        "Apply cannot reach the main window and would silently do nothing")

    window.close()
    plot.close()
    main.close()


def test_nothing_in_the_calibrate_trio_is_owned_by_anything(qapp):
    """All three windows must be free of each other.

    A widget parent makes a window Win32-owned, and an owned window drags
    its owner up the Z order when activated. Measured: with the dialog
    owned by the main window, clicking the dialog gave
    DIALOG > MAIN > PLOT -- the calibration plot dropped behind the main
    window. Unowned it gives DIALOG > PLOT > MAIN, and the main window can
    be raised by clicking it, neither of which was possible before.
    """
    from energy_assign_dialog import EnergyAssignDialog

    main = _window()
    dialog = EnergyAssignDialog(None, _choices(), max_channel=4095,
                                export_default_path="out.txt",
                                main_window=main)
    assert dialog.parent() is None, (
        "owned by the main window again -- activating it would drag the "
        "main window above the plot")

    for row in range(3):
        dialog.table.item(row, EnergyAssignDialog.ENERGY_COLUMN).setText(
            "%.1f" % (50.0 * (row + 1)))
    dialog._refresh_live_plot()
    plot = dialog._live_plot
    assert plot is not None
    assert plot.parent() is None, "the plot is owned again"

    dialog._close_live_plot()
    dialog.close()
    main.close()


def test_the_main_window_still_reaches_the_plot_without_a_parent_chain(qapp):
    """What the parent used to carry, now passed by hand.

    The plot reports a fitted efficiency to the main window and reads the
    theme from it, and used to find it by walking up parents. With nothing
    parented that walk finds nothing, so a missing hand-off would not
    raise -- the efficiency would simply never be remembered.
    """
    from energy_assign_dialog import EnergyAssignDialog

    main = _window()
    dialog = EnergyAssignDialog(None, _choices(), max_channel=4095,
                                export_default_path="out.txt",
                                main_window=main)
    for row in range(3):
        dialog.table.item(row, EnergyAssignDialog.ENERGY_COLUMN).setText(
            "%.1f" % (50.0 * (row + 1)))
    dialog._refresh_live_plot()

    assert dialog._live_plot._main_window is main, (
        "the plot cannot reach the main window, so a fitted efficiency "
        "would be silently dropped")

    dialog._close_live_plot()
    dialog.close()
    main.close()


def test_the_windows_opened_from_the_main_window_are_unowned(qapp, result):
    """The same rule, applied to every window that outlives its opener.

    Matrix panels and their heatmaps have always been parentless; the
    calibration plot and the efficiency window were not, which pinned them
    above the main window and made the main window impossible to click to
    the front. Modal dialogs are deliberately left alone: a modal belongs
    above the window it blocks, and it does not outlive the operation.
    """
    main = _window()

    main.set_efficiency(result)
    main.show_efficiency()
    assert main._efficiency_dialog.parent() is None, (
        "the efficiency window is owned again and will pin itself above "
        "the main window")

    main._show_calibration_plot(
        main.spectra[0], Calibration("linear", 0.0, 0.5), _points(), [], ())
    assert main._calibration_plot.parent() is None, (
        "the calibration plot is owned again")

    main.close()


def test_apply_still_finds_the_main_window_with_no_parent_chain(qapp, result):
    """The silent failure this would otherwise cause.

    EfficiencyDialog.Apply walks up to whatever can apply an efficiency and
    returns quietly when it finds nothing -- so a missing hand-off does not
    raise, it just makes the button do nothing.
    """
    main = _window()
    main.set_efficiency(result)
    main.show_efficiency()

    assert main._efficiency_dialog._main_window() is main, (
        "Apply cannot reach the main window, so it would silently do nothing")

    main.close()


def test_closing_the_main_window_leaves_nothing_behind(qapp, result):
    """Parentless windows are not destroyed with the main window, and Qt
    will not quit while any visible top-level remains -- app.exec() would
    never return and the process would linger with nothing on screen."""
    from PySide6.QtWidgets import QApplication

    main = _window()
    main.show()
    main.set_efficiency(result)
    main.show_efficiency()
    main._show_calibration_plot(
        main.spectra[0], Calibration("linear", 0.0, 0.5), _points(), [], ())
    qapp.processEvents()

    main.close()
    qapp.processEvents()

    left = [w for w in QApplication.topLevelWidgets()
            if w.isWindow() and w.isVisible()]
    assert left == [], "these would keep the application running: %s" % (
        [w.windowTitle() for w in left],)
