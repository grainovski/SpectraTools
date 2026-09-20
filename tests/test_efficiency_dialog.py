"""The button that launches the efficiency calibration, and what disables it."""

import numpy as np
import pytest

from calibration import Calibration
from calibration_plot_dialog import CalibrationPlotDialog


class _Line:
    """Enough of sou_io.SourceLine for build_rows."""

    def __init__(self, energy, intensity=1000.0, intensity_err=10.0):
        self.energy = energy
        self.energy_err = 0.01
        self.intensity = intensity
        self.intensity_err = intensity_err


def _dialog(qapp, points, lines, excluded=()):
    return CalibrationPlotDialog(
        None, Calibration("linear", 0.0, 0.5), points, lines, 4095,
        "out.txt", excluded=excluded)


def _points(n=8):
    """(channel, channel_err, area, area_err, energy) per point."""
    return [(100.0 * (i + 1), 0.1, 5000.0 - 300.0 * i, 70.0,
             50.0 * (i + 1)) for i in range(n)]


def _lines(n=8):
    return [_Line(50.0 * (i + 1)) for i in range(n)]


def test_the_button_is_enabled_with_enough_good_points(qapp):
    d = _dialog(qapp, _points(), _lines())
    assert d.efficiency_button.isEnabled()
    d.close()


def test_too_few_points_disables_it_with_a_reason(qapp):
    """Radware has five free parameters, so ndf = n - 5 must exceed zero."""
    d = _dialog(qapp, _points(4), _lines(4))
    assert not d.efficiency_button.isEnabled()
    assert "6" in d.efficiency_button.toolTip()
    d.close()


def test_a_peak_with_no_matching_source_line_does_not_count(qapp):
    """build_rows skips it, so it cannot reach the fit. If dropping it takes
    the total below the minimum the button must go with it, rather than
    letting the fit fail later on a count the dialog said was fine."""
    d = _dialog(qapp, _points(8), _lines(8)[:6])
    assert d.efficiency_button.isEnabled()      # 6 left, exactly the minimum

    d2 = _dialog(qapp, _points(8), _lines(8)[:5])
    assert not d2.efficiency_button.isEnabled()
    d.close(); d2.close()


def test_a_peak_with_no_usable_uncertainty_does_not_count(qapp):
    """eps = N/I with no error on either side gives an efficiency point of
    infinite weight. build_rows already refuses those rows; the button has
    to agree with it or the two disagree about how many points there are."""
    points = _points(8)
    lines = _lines(8)
    for i in (0, 1, 2):
        ch, dch, area, _area_err, energy = points[i]
        points[i] = (ch, dch, area, 0.0, energy)             # no area error
        lines[i] = _Line(50.0 * (i + 1), intensity_err=0.0)  # nor intensity
    d = _dialog(qapp, points, lines)
    assert not d.efficiency_button.isEnabled()
    d.close()


def test_an_excluded_point_does_not_count(qapp):
    """A point the user rejected from the energy calibration must not
    silently steer the efficiency curve either."""
    d = _dialog(qapp, _points(8), _lines(8), excluded=(350.0, 400.0))
    assert d.efficiency_button.isEnabled()   # 6 left

    d2 = _dialog(qapp, _points(8), _lines(8), excluded=(300.0, 350.0, 400.0))
    assert not d2.efficiency_button.isEnabled()
    d.close(); d2.close()


def test_a_zero_area_disables_it(qapp):
    """eps = N/I must be positive to fit at all, and build_rows does not
    check the sign -- only that an uncertainty exists."""
    points = _points(8)
    points[2] = (300.0, 0.1, 0.0, 70.0, 150.0)
    d = _dialog(qapp, points, _lines())
    assert not d.efficiency_button.isEnabled()
    assert "area" in d.efficiency_button.toolTip().lower()
    d.close()


def test_the_enable_check_can_fail(qapp):
    """Control: the good case must really be enabled, or every assertion
    above passes for the wrong reason."""
    good = _dialog(qapp, _points(), _lines())
    bad = _dialog(qapp, _points(4), _lines(4))
    assert good.efficiency_button.isEnabled()
    assert not bad.efficiency_button.isEnabled()
    good.close(); bad.close()
