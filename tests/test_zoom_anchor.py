"""Wheel zoom holds the channel under the pointer still.

It used to recentre: the bin under the cursor was moved to the middle of
the new view. Zooming into a peak therefore threw that peak to the centre
on the first tick, so the second tick zoomed into whatever the first had
moved under the pointer, and following a peak in meant chasing it.

The arithmetic is shared by the main window and the matrix panel's
projection, which each had their own copy.
"""

import math

import pytest

from calibration_view import zoomed_limits


FULL_LO, FULL_HI = 0.0, 4095.0
ZOOM_IN = 1 / 1.5          # one wheel tick "up"
ZOOM_OUT = 1.5


def _fraction_across(value, lo, hi):
    return (value - lo) / (hi - lo)


def test_the_channel_under_the_pointer_does_not_move():
    lo, hi, cursor = 0.0, 1000.0, 800.0
    before = _fraction_across(cursor, lo, hi)
    new_lo, new_hi = zoomed_limits((lo, hi), ZOOM_IN, cursor, FULL_LO, FULL_HI)
    after = _fraction_across(cursor, new_lo, new_hi)
    assert after == pytest.approx(before), (
        "the cursor's channel moved from %.0f%% to %.0f%% across the view"
        % (100 * before, 100 * after)
    )


def test_it_holds_over_repeated_ticks():
    """One tick staying put is not enough: following a peak in means
    several, and a small drift per tick would still lose it."""
    lo, hi, cursor = 0.0, 4095.0, 3200.0
    before = _fraction_across(cursor, lo, hi)
    for _ in range(8):
        lo, hi = zoomed_limits((lo, hi), ZOOM_IN, cursor, FULL_LO, FULL_HI)
        assert _fraction_across(cursor, lo, hi) == pytest.approx(before)
    assert hi - lo < 200.0, "eight ticks should have zoomed a long way in"


def test_zooming_out_holds_it_too():
    lo, hi, cursor = 1000.0, 1200.0, 1150.0
    before = _fraction_across(cursor, lo, hi)
    new_lo, new_hi = zoomed_limits((lo, hi), ZOOM_OUT, cursor, FULL_LO, FULL_HI)
    assert _fraction_across(cursor, new_lo, new_hi) == pytest.approx(before)
    assert new_hi - new_lo > hi - lo


def test_recentring_would_fail_this():
    """The control, written as the OLD behaviour. Without it the tests
    above could be satisfied by arithmetic that happens to agree at the
    midpoint, which is where most naive cases sit."""
    lo, hi, cursor = 0.0, 1000.0, 800.0
    half = abs(hi - lo) / 2 * ZOOM_IN
    old_lo, old_hi = max(FULL_LO, cursor - half), min(FULL_HI, cursor + half)
    assert _fraction_across(cursor, old_lo, old_hi) == pytest.approx(0.5)
    assert _fraction_across(cursor, old_lo, old_hi) != pytest.approx(0.8)


# --- the paths that must not change ------------------------------------


def test_the_keyboard_zoom_is_unchanged():
    """Ctrl+= / Ctrl+- pass no cursor. Anchoring on the midpoint is the
    same symmetric zoom as before, and has to stay that way."""
    lo, hi = 200.0, 1200.0
    new_lo, new_hi = zoomed_limits((lo, hi), ZOOM_IN, None, FULL_LO, FULL_HI)
    midpoint = (lo + hi) / 2
    assert (new_lo + new_hi) / 2 == pytest.approx(midpoint)
    assert new_hi - new_lo == pytest.approx((hi - lo) * ZOOM_IN)


def test_an_inverted_axis_keeps_its_orientation():
    """A negative-b calibration makes channel_to_display decreasing and
    matplotlib honours the inversion. Writing ascending limits back would
    flip the axis on every wheel tick."""
    new = zoomed_limits((900.0, 100.0), ZOOM_IN, 300.0, FULL_LO, FULL_HI)
    assert new[0] > new[1], "descending limits came back ascending: %r" % (new,)


def test_it_never_shows_channels_the_data_does_not_have():
    lo, hi = zoomed_limits((10.0, 50.0), 500.0, 30.0, FULL_LO, FULL_HI)
    assert lo >= FULL_LO and hi <= FULL_HI


def test_clamping_at_an_edge_still_returns_a_usable_range():
    """Zooming out hard at channel 0 clamps the low end; the result must
    still be a range, not an inverted or empty one."""
    lo, hi = zoomed_limits((0.0, 20.0), ZOOM_OUT, 0.0, FULL_LO, FULL_HI)
    assert hi > lo


def test_a_degenerate_view_does_not_collapse():
    lo, hi = zoomed_limits((100.0, 100.0), ZOOM_IN, 100.0, FULL_LO, FULL_HI)
    assert hi > lo


# --- and the real wheel, through the real widget -----------------------
#
# The arithmetic being right is not proof the application uses it. These
# drive an actual scroll_event on a real window, the way the user does.


def _scroll_and_measure(window, axes, canvas, channel):
    """Wheel one tick in over `channel`; return where it sat before and
    after, as a fraction across the visible range."""
    from matplotlib.backend_bases import MouseEvent

    lo, hi = axes.get_xlim()
    before = (channel - lo) / (hi - lo)
    px, py = axes.transData.transform((channel, 1.0))
    canvas.callbacks.process(
        "scroll_event", MouseEvent("scroll_event", canvas, px, py, button="up")
    )
    new_lo, new_hi = axes.get_xlim()
    return before, (channel - new_lo) / (new_hi - new_lo), (new_hi - new_lo) < (hi - lo)


def test_the_main_window_wheel_holds_the_channel_under_the_pointer(qapp):
    import numpy as np
    from main_window import MainWindow

    from spectrum import LoadedSpectrum

    window = MainWindow()
    spectrum = LoadedSpectrum(
        "wheel.txt", np.linspace(10.0, 500.0, 4096).astype(np.int64), "#1f77b4"
    )
    spectrum.active = True
    window.spectra.append(spectrum)
    window._plot_data()
    window.axes.set_xlim(0.0, 4095.0)

    before, after, zoomed = _scroll_and_measure(
        window, window.axes, window.canvas, 3000.0
    )
    assert zoomed, "the wheel did not zoom in at all"
    assert after == pytest.approx(before, abs=1e-6), (
        "channel 3000 moved from %.1f%% to %.1f%% across the view"
        % (100 * before, 100 * after)
    )
    window.close()
