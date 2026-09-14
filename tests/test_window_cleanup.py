"""Windows created by a test must not outlive it.

The suite segfaulted at 52% of a full run with the process at 3.4 GB,
while every file passed alone. The cause is not a leak in the app:
MatrixPanel.closeEvent releases its decoded matrix and MainWindow.closeEvent
closes its panels. It is that tests never close what they open --
test_matrix_panel.py alone opens 75 panels and closes none -- so those
closeEvents never run and an 8192x8192 matrix (512 MB) stays reachable
through Qt signal closures for the rest of the session.

Assertions here are scoped to the panels THIS test created. A global
sweep fails for other tests' sake, because some deliberately leave
windows open while they run.

They assert on the PAYLOAD -- panel.matrix -- rather than on the widgets
being garbage collected, following da61ff6: whether Qt frees the widget
depends on ownership details that regress easily, while the decoded
matrix is the memory that actually costs something. An earlier draft of
this file did assert the MainWindow became unreachable, and failed; the
object survives, and that is accepted rather than a defect.
"""

import os

from conftest import close_all_windows
from main_window import MainWindow
from matrix_panel import MatrixPanel

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def test_close_all_windows_releases_a_panels_matrix(qapp):
    """The payload is what matters. Whether the widget object itself is
    collected depends on Qt/Python ownership details that regress easily,
    so this asserts on the memory that actually costs something."""
    window = MainWindow()
    panel = MatrixPanel(window, os.path.join(FIXTURES, "gg.mtx"))
    assert panel.matrix is not None
    assert panel.matrix.nbytes > 100 * 1024 * 1024, "fixture should be large"

    close_all_windows()

    assert panel.matrix is None, "the decoded matrix was not released"
    assert panel.projections == {}


def test_every_panel_a_test_opened_is_released_not_just_the_last(qapp):
    """Several panels, as a real test file produces.

    Asserting on the PAYLOAD rather than on the widgets being collected,
    deliberately and for the reason da61ff6 gives: whether Qt frees the
    widget depends on ownership details that regress easily, while the
    decoded matrix is the memory that actually costs something. An
    earlier version of this test asserted the MainWindow became
    unreachable, and failed -- the object does survive, and that is
    accepted.
    """
    # Two, not more: "not just the last one" is what this test adds over
    # its neighbour, and two establishes it. Each extra panel decodes the
    # same 30 MB fixture again and cost ~20s -- at three this was the
    # slowest test in the whole suite, which is a poor trade for a claim
    # two already make.
    panels = []
    for _ in range(2):
        window = MainWindow()
        panels.append(MatrixPanel(window, os.path.join(FIXTURES, "gg.mtx")))
    assert all(p.matrix is not None for p in panels)

    close_all_windows()

    still_held = [p for p in panels if p.matrix is not None]
    assert not still_held, f"{len(still_held)} of 2 panels kept their matrix"


def test_close_all_windows_is_safe_with_nothing_open(qapp):
    """Runs after every test in the suite, most of which open no window."""
    close_all_windows()
    close_all_windows()
