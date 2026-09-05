"""The calibration plot window: coefficients, reduced chi-squared, and
the CalEnEff export behind Finish."""

import numpy as np
import pytest

from matplotlib.backend_bases import MouseEvent

from calibration import Calibration
from calibration_plot_dialog import CalibrationPlotDialog
from sou_io import SourceLine


def _lines():
    return [
        SourceLine(121.783, 0.002, 5000.0, 50.0),
        SourceLine(344.276, 0.004, 10000.0, 80.0),
    ]


def _points():
    return [
        (100.0, 0.05, 9000.0, 95.0, 121.783),
        (300.0, 0.08, 4000.0, 63.0, 344.276),
        (500.0, 0.06, 2000.0, 45.0, 566.0),
    ]


def _dialog(qapp, calibration=None, points=None, lines=None):
    return CalibrationPlotDialog(
        None,
        calibration or Calibration(kind="linear", a=10.0, b=1.1),
        points if points is not None else _points(),
        lines if lines is not None else _lines(),
        max_channel=1024,
        default_path="out_En_Area.txt",
    )


def test_coefficients_and_their_uncertainties_are_shown(qapp):
    cal = Calibration(kind="linear", a=10.0, b=1.1,
                      coefficient_errors=(0.02, 0.0013))
    # Held in a variable rather than chained: `_dialog(...).summary_label`
    # drops the only reference to the parentless QDialog the instant its
    # attribute is read, which frees the C++ object -- and everything Qt
    # parented under it, including summary_label -- before `.text()` runs.
    dialog = _dialog(qapp, calibration=cal)
    text = dialog.summary_label.text()
    assert "10.00(20)" in text or "10.0000(200)" in text or "10.000(20)" in text
    assert "b" in text


def test_reduced_chi_squared_is_shown_for_a_weighted_fit(qapp):
    dialog = _dialog(qapp)
    text = dialog.summary_label.text()
    assert "chi" in text.lower()
    assert "undefined" not in text.lower()


def test_an_unweighted_fit_says_so_instead_of_printing_a_number(qapp):
    points = [(100.0, 0.0, 9000.0, 95.0, 121.783),
              (300.0, 0.08, 4000.0, 63.0, 344.276),
              (500.0, 0.06, 2000.0, 45.0, 566.0)]
    dialog = _dialog(qapp, points=points)
    text = dialog.summary_label.text()
    assert "unweighted" in text.lower()


def test_the_minimum_number_of_points_reports_no_degrees_of_freedom(qapp):
    points = [(100.0, 0.05, 9000.0, 95.0, 121.783),
              (300.0, 0.08, 4000.0, 63.0, 344.276)]
    dialog = _dialog(qapp, points=points)
    text = dialog.summary_label.text()
    assert "degrees of freedom" in text.lower()


def test_the_plot_draws_the_points_and_the_curve(qapp):
    dialog = _dialog(qapp)
    axes = dialog.axes
    # One errorbar container for the points, at least one line for the fit.
    assert len(axes.containers) >= 1
    assert len(axes.lines) >= 1


def test_export_writes_only_the_rows_with_intensities(qapp, tmp_path):
    """The 566.0 keV point was typed by hand and has no source line."""
    dialog = _dialog(qapp)
    path = tmp_path / "out.txt"
    summary = dialog.export_to(str(path))

    data = np.loadtxt(str(path), ndmin=2)
    assert data.shape == (2, 7)
    assert "1" in summary  # reports the skipped one
    assert "skip" in summary.lower()


def test_export_normalises_intensity_to_the_strongest_line(qapp, tmp_path):
    dialog = _dialog(qapp)
    path = tmp_path / "out.txt"
    dialog.export_to(str(path))
    data = np.loadtxt(str(path), ndmin=2)
    assert data[:, 5].max() == pytest.approx(100.0)


def test_export_with_nothing_exportable_raises(qapp, tmp_path):
    from caleneff_export import ExportError

    dialog = _dialog(qapp, lines=[])
    with pytest.raises(ExportError):
        dialog.export_to(str(tmp_path / "out.txt"))


# --- v5.1.0: drawn with no calibration yet ------------------------------


def _curve_lines(dialog):
    return [line for line in dialog.axes.lines if line.get_label() == "calibration"]


def test_without_a_calibration_the_points_are_drawn_but_no_curve(qapp):
    """Two points under a quadratic, or a typo mid-edit: the window used
    to close, which looked like a crash. Now the points stay and the
    summary says why there is no line through them."""
    dialog = CalibrationPlotDialog(
        None, None, _points(), _lines(), max_channel=1024,
        default_path="out.txt", reason="two points cannot fix a quadratic",
    )
    assert len(dialog.axes.containers) >= 1          # the points are there
    assert _curve_lines(dialog) == []                 # no curve
    text = dialog.summary_label.text().lower()
    assert "no calibration" in text
    assert "quadratic" in text


def test_set_data_moves_between_a_fit_and_none_without_rebuilding(qapp):
    dialog = _dialog(qapp)
    assert _curve_lines(dialog)
    dialog.set_data(None, _points(), _lines(), reason="typo mid-edit")
    assert _curve_lines(dialog) == []
    assert "typo mid-edit" in dialog.summary_label.text()
    dialog.set_data(Calibration(kind="linear", a=10.0, b=1.1), _points(), _lines())
    assert _curve_lines(dialog)
    assert "linear calibration" in dialog.summary_label.text()


def test_no_points_at_all_is_drawn_as_an_empty_plot(qapp):
    """Clear empties the table; the window stays with nothing in it
    rather than going away."""
    dialog = _dialog(qapp)
    dialog.set_data(None, [], _lines(), reason="no assignments")
    assert _curve_lines(dialog) == []
    assert "no assignments" in dialog.summary_label.text()


# --- picking a point off the plot ---------------------------------------


def _click(dialog, axes, x, y, button=1):
    """Click at the DATA position (x, y) of `axes`, through the canvas's
    own event machinery, so the wiring is exercised and not just the
    handler."""
    canvas = dialog._figure.canvas
    canvas.draw()
    px, py = axes.transData.transform((x, y))
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent("button_press_event", canvas, px, py, button),
    )


def test_clicking_a_point_on_the_curve_picks_it(qapp):
    dialog = _dialog(qapp)
    picked = []
    dialog.pointPicked.connect(picked.append)
    _click(dialog, dialog.axes, 300.0, 344.276)
    assert picked == [1]
    assert dialog.picked_index() == 1
    assert "344.276" in dialog.picked_label.text()


def test_clicking_a_point_on_the_residual_strip_picks_it(qapp):
    """The point of the feature: the residual strip is where a
    misidentified line stands out, and it carries nothing that says
    which line it is."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    dialog = _dialog(qapp, calibration=cal)
    picked = []
    dialog.pointPicked.connect(picked.append)
    _click(dialog, dialog.residual_axes, 500.0, 566.0 - cal.apply(500.0))
    assert picked == [2]
    assert dialog.picked_index() == 2


def test_clicking_empty_space_picks_nothing(qapp):
    """CONTROL: without this the nearest point would be picked wherever
    the click landed, and the ring would mean nothing."""
    dialog = _dialog(qapp)
    picked = []
    dialog.pointPicked.connect(picked.append)
    _click(dialog, dialog.axes, 300.0, 344.276)
    _click(dialog, dialog.axes, 200.0, 700.0)
    assert picked == [1, -1]
    assert dialog.picked_index() is None
    assert dialog.picked_label.text() == ""


def test_a_right_click_is_not_a_pick(qapp):
    dialog = _dialog(qapp)
    _click(dialog, dialog.axes, 300.0, 344.276, button=3)
    assert dialog.picked_index() is None


def test_an_excluded_point_can_be_picked_and_says_it_is_excluded(qapp):
    dialog = _dialog(qapp)
    dialog.set_data(dialog._calibration, _points(), _lines(), excluded=(566.0,))
    _click(dialog, dialog.axes, 500.0, 566.0)
    assert dialog.picked_index() == 2
    assert "excluded" in dialog.picked_label.text()


def test_the_pick_follows_its_point_through_a_redraw(qapp):
    """The Calibrate dialog redraws on every keystroke. A ring left on
    an index rather than on a peak would slide onto a different point
    the moment a row above it was cleared."""
    dialog = _dialog(qapp)
    _click(dialog, dialog.axes, 500.0, 566.0)
    assert dialog.picked_index() == 2
    dialog.set_data(dialog._calibration, _points()[1:], _lines())
    assert dialog.picked_index() == 1, "the ring moved to another peak"
    assert "566" in dialog.picked_label.text()


def test_a_pick_whose_point_is_gone_is_dropped(qapp):
    dialog = _dialog(qapp)
    _click(dialog, dialog.axes, 500.0, 566.0)
    dialog.set_data(dialog._calibration, _points()[:2], _lines())
    assert dialog.picked_index() is None
    assert dialog.picked_label.text() == ""


def test_nothing_can_be_picked_off_a_residual_strip_with_no_calibration(qapp):
    """With no fit there are no residuals -- only the reason, written
    across the strip."""
    dialog = _dialog(qapp, calibration=None)
    dialog.set_data(None, _points(), _lines(), reason="too few points")
    _click(dialog, dialog.residual_axes, 0.5, 0.5)
    assert dialog.picked_index() is None



# --- the residual strip is scaled to the fit, not to what was left out --


def _outlier_points():
    """Three points on the line and one far off it."""
    return [(100.0, 0.05, 9000.0, 95.0, 120.1),
            (300.0, 0.08, 4000.0, 63.0, 340.2),
            (500.0, 0.06, 2000.0, 45.0, 559.9),
            (700.0, 0.06, 1000.0, 30.0, 830.0)]


def test_an_excluded_outlier_does_not_set_the_residual_scale(qapp):
    """Its residual is fifty keV against a tenth of one; letting it set
    the scale squashed every remaining point onto the zero line, which is
    the one thing the strip exists to show."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    dialog = _dialog(qapp, calibration=cal, points=_outlier_points())
    dialog.set_data(cal, _outlier_points(), _lines(), excluded=(830.0,))
    low, high = dialog.residual_axes.get_ylim()
    assert high < 1.0, "the excluded point still set the scale"
    assert low < -0.1 <= 0.2 < high, "an included residual fell outside the view"


def test_with_the_outlier_included_the_scale_does_stretch_to_it(qapp):
    """CONTROL: same points, nothing excluded. Without this the test
    above would pass on a plot that never scaled to anything."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    dialog = _dialog(qapp, calibration=cal, points=_outlier_points())
    _low, high = dialog.residual_axes.get_ylim()
    assert high > 40.0


def test_the_excluded_point_is_still_drawn_and_still_pickable(qapp):
    """Out of the residual view is not out of the window: it stays on
    the curve above, where clicking it names it and its residual."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    dialog = _dialog(qapp, calibration=cal, points=_outlier_points())
    dialog.set_data(cal, _outlier_points(), _lines(), excluded=(830.0,))
    _click(dialog, dialog.axes, 700.0, 830.0)
    assert dialog.picked_index() == 3
    assert "excluded" in dialog.picked_label.text()


def test_one_point_on_the_line_still_gets_a_usable_scale(qapp):
    """A single residual, or a fit passing exactly through every point,
    gives a zero span; the strip must not collapse to nothing."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    dialog = _dialog(qapp, calibration=cal, points=_outlier_points()[:1])
    low, high = dialog.residual_axes.get_ylim()
    assert high > low
