"""The matrix panel shows the cursor position the way the main window does:
channel and counts, plus energy when a calibration is active.

The main window has had this in its status bar all along; the projection
view had nothing, because its motion handler only panned.
"""

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from main_window import MainWindow


@pytest.fixture
def panel(qapp):
    m = np.zeros((200, 200), dtype=np.int64)
    m[:, 100] = 1          # column 100 ...
    m[37, :] = 2           # ... row 37 OVERWRITES cell (37, 100) with 2:
    # X projection at 100 = 199*1 + 2 = 201; Y projection at 37 = 200*2 = 400.
    window = MainWindow()
    p = window._open_matrix_panel("synthetic.json", matrix=m)
    yield p
    p.close()
    window.close()


def _move_to(panel, x, y=1.0):
    px, py = panel.axes.transData.transform((x, y))
    event = MouseEvent("motion_notify_event", panel.canvas, px, py)
    panel._on_mouse_move(event)
    return panel.statusBar().currentMessage()


def test_channel_and_counts_on_the_x_projection(panel):
    message = _move_to(panel, 100)
    assert "Channel: 100" in message
    assert "X projection: 201" in message, message


def test_it_follows_the_projection_being_shown(panel):
    panel.axis_selector.setCurrentIndex(panel.axis_selector.findData("y"))
    message = _move_to(panel, 37)
    assert "Channel: 37" in message
    assert "Y projection: 400" in message, message


def test_energy_is_added_when_calibrated(panel):
    from calibration import Calibration

    window = panel.main_window
    window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    window._calibration_active = True
    panel._plot_data()
    # Channel 100 -> 10 + 0.5*100 = 60 keV.
    message = _move_to(panel, 60.0)
    assert "Channel: 100" in message
    assert "Energy: 60.00 keV" in message
    assert "X projection: 201" in message, message


def test_leaving_the_plot_clears_the_readout(panel):
    _move_to(panel, 100)
    event = MouseEvent("motion_notify_event", panel.canvas, -50, -50)
    panel._on_mouse_move(event)
    assert panel.statusBar().currentMessage() == ""


def test_panning_still_pans_and_does_not_report(panel):
    """The handler now does two jobs. A drag must still move the view."""
    before = panel.axes.get_xlim()
    panel._zoom_x(0.25)                 # zoom in so there is room to pan
    zoomed = panel.axes.get_xlim()
    assert zoomed != before
    px0, _ = panel.axes.transData.transform((zoomed[0] + 5, 1))
    panel._pan_last_px = px0
    panel.statusBar().clearMessage()
    event = MouseEvent("motion_notify_event", panel.canvas, px0 - 40, 50)
    panel._on_mouse_move(event)
    assert panel.axes.get_xlim() != zoomed, "the drag did not pan"
    assert panel.statusBar().currentMessage() == ""


def test_a_real_mouse_move_reaches_the_readout(panel):
    """The tests above call _on_mouse_move directly, so they would pass
    even if the canvas were never wired to it. This one sends the event
    through the canvas's own callback registry, as a real move does."""
    px, py = panel.axes.transData.transform((100, 1.0))
    event = MouseEvent("motion_notify_event", panel.canvas, px, py)
    panel.statusBar().clearMessage()
    panel.canvas.callbacks.process("motion_notify_event", event)
    assert "X projection: 201" in panel.statusBar().currentMessage()
