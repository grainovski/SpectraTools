"""Shared window behaviour: bringing one to the front, and closing it.

Two things that both follow from this app opening several non-modal
windows at once -- the calibration plot, the energy assignment dialog, the
efficiency window, a matrix panel and its heatmap can all be on screen
together, overlapping.

Reopening a window the user may already have closed.

Every reopenable window here is stored on an attribute of its owner AND
created with WA_DeleteOnClose. Both decisions are deliberate: the
attribute is what lets the window be reopened or replaced, and the
delete-on-close is what stops a second calibration leaving the first
window on screen showing different coefficients.

Together they leave a trap. When the user closes the window, Qt destroys
the C++ object while the attribute goes on holding the Python wrapper.
The next open reaches for that wrapper to close the "previous" window,
and touching a wrapper whose C++ object is gone raises RuntimeError.

That exception is raised inside a Qt slot. In a --windowed build there is
no stderr for the traceback to reach, so nothing is printed, nothing is
logged, and the window simply never appears again -- see
`test_dialog_utils.py`, which pins the behaviour, and the v5.2.1 audit,
where an exception leaving a Qt slot hid a defect the same way.
"""

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QWidget


class RaiseOnClickFilter(QObject):
    """Bring a window to the front when it is clicked anywhere.

    Nothing in this app used to ask the window manager to reorder its
    windows, so a half-covered plot stayed half-covered however often it
    was clicked.

    This filter is necessary but not sufficient, and the difference cost
    two releases. Asking is only heeded for windows the window manager is
    free to move: a widget parent makes a window Win32-OWNED by its
    parent, and Windows keeps an owned window above its owner whatever it
    is asked -- measured, including with the Qt window type changed, which
    does not affect ownership. Windows that have to stack freely must not
    be parented to each other; see energy_assign_dialog.

    Installed on the QApplication rather than overridden per window,
    because a click almost never reaches the window itself: a matplotlib
    canvas, a button, a table cell or a text field consumes it first, so
    per-window mousePressEvent would catch only clicks on bare background
    -- which is not what "click anywhere" means.

    The filter never consumes the event. It returns False always, so the
    click goes on to do whatever it was going to do; raising is on top of
    that, not instead of it.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            window = window_to_raise(
                obj if isinstance(obj, QWidget) else None,
                QApplication.activePopupWidget())
            if window is not None:
                window.raise_()
                window.activateWindow()
        return False


def window_to_raise(widget, popup=None):
    """The top-level window a click on `widget` should bring to the front,
    or None if the click should change no stacking.

    Split out from the event filter because everything worth getting wrong
    is in here, and this can be tested by calling it.

    Returns None when:

    * there is no widget, or it has no window (events do reach objects
      that are not widgets at all);
    * a popup is open. A menu, a combo box drop-down and a completer are
      all separate windows, and raising the window underneath one closes
      it. `popup` is QApplication.activePopupWidget(), passed in rather
      than read here so a test can set it.

    It deliberately does NOT skip a window that is already active. That
    looks like a free optimisation and is the opposite: the reported bug
    had the dialog ACTIVE and still underneath, so "active" does not imply
    "on top", and skipping active windows would decline to raise exactly
    the window the user clicked to bring forward. The cost is one
    window-manager call per click, at human speed.
    """
    if widget is None or popup is not None:
        return None
    window = widget.window()
    return window


def close_previous(previous):
    """Close and release `previous`, tolerating it already being gone.

    Returns True if it was still alive and has now been closed, False if
    there was none or the user had already closed it. No caller needs the
    answer -- it is returned so a test can tell those two cases apart.
    """
    if previous is None:
        return False
    try:
        previous.close()
        previous.deleteLater()
    except RuntimeError:
        # WA_DeleteOnClose destroyed the C++ object already. A Python
        # wrapper outliving it is the normal end of a dialog's life, not
        # a fault worth reporting.
        return False
    return True
