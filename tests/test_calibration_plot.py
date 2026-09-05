"""The calibration plot window: coefficients, reduced chi-squared, and
the CalEnEff export behind Finish."""

import numpy as np
import pytest

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
