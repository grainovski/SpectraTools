"""Reopening a window the user may already have closed.

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
