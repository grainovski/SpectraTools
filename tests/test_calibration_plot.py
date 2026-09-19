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
# --- the plot takes each line's uncertainty from the source it loaded ----


def test_the_summary_counts_the_source_lines_own_uncertainties(qapp):
    """The two lines of _lines() carry dE of 0.002 and 0.004 keV. Giving
    the same points a source whose lines are quoted a hundred times more
    loosely must lower the reported chi-squared."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    loose = [SourceLine(121.783, 0.5, 5000.0, 50.0),
             SourceLine(344.276, 0.5, 10000.0, 80.0)]
    tight = _dialog(qapp, calibration=cal)
    slack = _dialog(qapp, calibration=cal, lines=loose)

    def chi(dialog):
        text = dialog.summary_label.text()
        return float(text.split("reduced chi^2 = ")[1].split(",")[0])

    assert chi(slack) < chi(tight)


def test_a_point_with_no_source_line_contributes_no_extra_uncertainty(qapp):
    """CONTROL: the third point of _points() is at 566 keV, which neither
    source line matches, so widening the source must not change it."""
    cal = Calibration(kind="linear", a=10.0, b=1.1)
    dialog = _dialog(qapp, calibration=cal)
    assert dialog._literature_errors([121.783, 344.276, 566.0]) == [0.002, 0.004, 0.0]


def test_no_source_loaded_leaves_every_uncertainty_at_zero(qapp):
    dialog = _dialog(qapp, calibration=Calibration(kind="linear", a=10.0, b=1.1),
                     lines=[])
    assert dialog._literature_errors([121.783, 344.276]) == [0.0, 0.0]


# --- the residual strip has to show how big a residual IS --------------


def _residual_yerr(dialog):
    """The y error bar lengths drawn on the residual strip, if any."""
    out = []
    for container in dialog.residual_axes.containers:
        # An errorbar container holds (line, caplines, barlinecols); the
        # y bars are a LineCollection in the third slot.
        bars = container[2]
        if not bars:
            continue
        for collection in bars:
            for segment in collection.get_segments():
                out.append(abs(segment[1][1] - segment[0][1]))
    return out


def test_the_residual_strip_draws_y_error_bars(qapp):
    """Without them a 2-sigma outlier and a 0.2-sigma one look identical,
    which is the single distinction this strip exists to make. The values
    were already in hand: the centroid sigma carried into keV by the
    calibration's slope, plus the line's stated dE.
    """
    dialog = _dialog(qapp)
    lengths = _residual_yerr(dialog)
    assert lengths, "the residual strip drew no y error bars at all"
    assert all(v > 0.0 for v in lengths)


def test_a_bigger_centroid_uncertainty_draws_a_bigger_bar(qapp):
    """Control: bars of a fixed size would satisfy the test above just as
    well, and would say nothing true about any point."""
    small = _dialog(qapp, points=[
        (100.0, 0.01, 9000.0, 95.0, 121.783),
        (300.0, 0.01, 4000.0, 63.0, 344.276),
        (500.0, 0.01, 2000.0, 45.0, 566.0),
    ])
    large = _dialog(qapp, points=[
        (100.0, 1.00, 9000.0, 95.0, 121.783),
        (300.0, 1.00, 4000.0, 63.0, 344.276),
        (500.0, 1.00, 2000.0, 45.0, 566.0),
    ])
    assert max(_residual_yerr(large)) > max(_residual_yerr(small)) * 10


def test_the_bar_combines_the_centroid_and_the_literature_error(qapp):
    """The same quadrature reduced_chi_squared weights with. A line with a
    stated dE must draw a taller bar than the same point without one."""
    with_de = _dialog(qapp, points=[(100.0, 0.05, 9000.0, 95.0, 121.783)],
                      lines=[SourceLine(121.783, 5.0, 5000.0, 50.0)])
    without = _dialog(qapp, points=[(100.0, 0.05, 9000.0, 95.0, 121.783)],
                      lines=[SourceLine(121.783, 0.0, 5000.0, 50.0)])
    assert max(_residual_yerr(with_de)) > max(_residual_yerr(without))



# --- the strip and the chi-squared must agree on what is undefined -----


def _vertex_case():
    """A quadratic whose turning point sits on a data point, with no
    stated dE anywhere. There dE/dch is zero, so a channel uncertainty
    maps to no energy uncertainty and nothing is left to draw."""
    import math
    from calibration import Calibration
    c = -0.001
    cal = Calibration(kind="quadratic", a=0.0, b=-2 * c * 500.0, c=c)
    pts = [(500.0, 0.05, 9000.0, 95.0, cal.apply(500.0)),
           (200.0, 0.05, 4000.0, 63.0, cal.apply(200.0)),
           (800.0, 0.05, 2000.0, 45.0, cal.apply(800.0)),
           (300.0, 0.05, 2000.0, 45.0, cal.apply(300.0))]
    return cal, pts


def test_a_point_at_the_turning_point_gets_no_error_bar(qapp):
    """It used to get a bar of length zero, which reads as an exact
    measurement -- the opposite of the truth, on precisely the point a
    reader should trust least."""
    import math
    cal, pts = _vertex_case()
    dialog = _dialog(qapp, calibration=cal, points=pts, lines=[])
    bars = dialog._residual_errors([p[0] for p in pts], [p[1] for p in pts],
                                   [p[4] for p in pts])
    assert math.isnan(bars[0]), "the vertex point must draw no bar, got %r" % bars[0]


def test_the_other_points_still_get_real_bars(qapp):
    """Control: the rule must be about the turning point specifically,
    not about quadratics in general."""
    import math
    cal, pts = _vertex_case()
    dialog = _dialog(qapp, calibration=cal, points=pts, lines=[])
    bars = dialog._residual_errors([p[0] for p in pts], [p[1] for p in pts],
                                   [p[4] for p in pts])
    assert all(math.isfinite(v) and v > 0.0 for v in bars[1:]), bars


def test_the_strip_and_the_chi_squared_agree_on_undefined(qapp):
    """The two describe the same quantity, so they cannot disagree about
    which points have one."""
    import math
    from calibration_quality import reduced_chi_squared
    cal, pts = _vertex_case()
    chans = [p[0] for p in pts]
    value, _reason = reduced_chi_squared(cal, chans, [p[4] for p in pts],
                                         [p[1] for p in pts])
    dialog = _dialog(qapp, calibration=cal, points=pts, lines=[])
    bars = dialog._residual_errors(chans, [p[1] for p in pts], [p[4] for p in pts])
    assert value is None and math.isnan(bars[0])
