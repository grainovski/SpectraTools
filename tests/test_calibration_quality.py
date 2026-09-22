"""Reduced chi-squared for an energy calibration, and the three cases
where it does not exist."""


import pytest

from calibration import Calibration
from calibration_quality import reduced_chi_squared


def test_a_perfect_fit_has_chi_squared_of_zero():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = [100.0, 200.0, 300.0]
    energies = [cal.apply(c) for c in channels]
    value, reason = reduced_chi_squared(cal, channels, energies, [0.1, 0.1, 0.1])
    assert reason is None
    assert value == pytest.approx(0.0, abs=1e-12)


def test_value_matches_an_independent_computation():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0]
    errors = [0.1, 0.2, 0.1, 0.4]
    energies = [cal.apply(c) + d for c, d in
                zip(channels, (0.05, -0.03, 0.02, 0.10))]

    expected_chi2 = sum(
        ((e - cal.apply(c)) / (s * cal.derivative(c))) ** 2
        for c, e, s in zip(channels, energies, errors)
    )
    expected = expected_chi2 / (len(channels) - 2)

    value, reason = reduced_chi_squared(cal, channels, energies, errors)
    assert reason is None
    assert value == pytest.approx(expected)


def test_quadratic_uses_three_parameters():
    cal = Calibration(kind="quadratic", a=1.0, b=0.5, c=1e-6)
    channels = [100.0, 200.0, 300.0, 400.0]
    errors = [0.1] * 4
    energies = [cal.apply(c) + 0.01 for c in channels]

    expected_chi2 = sum(
        ((e - cal.apply(ch)) / (s * cal.derivative(ch))) ** 2
        for ch, e, s in zip(channels, energies, errors)
    )
    value, _ = reduced_chi_squared(cal, channels, energies, errors)
    assert value == pytest.approx(expected_chi2 / (len(channels) - 3))


def test_unusable_errors_report_an_unweighted_fit():
    """from_points falls back to an unweighted fit when any error is
    zero, negative or non-finite. The weights are then arbitrary and a
    printed chi-squared would look like a goodness of fit."""
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = [100.0, 200.0, 300.0]
    energies = [60.0, 110.0, 160.0]
    for errors in ([0.1, 0.0, 0.1], [0.1, -1.0, 0.1],
                   [0.1, float("nan"), 0.1], [0.1, None, 0.1]):
        value, reason = reduced_chi_squared(cal, channels, energies, errors)
        assert value is None
        assert "unweighted" in reason


def test_exactly_the_minimum_points_has_no_degrees_of_freedom():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    value, reason = reduced_chi_squared(
        cal, [100.0, 200.0], [60.0, 110.0], [0.1, 0.1]
    )
    assert value is None
    assert "degrees of freedom" in reason


def test_missing_errors_entirely_is_unweighted():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    value, reason = reduced_chi_squared(
        cal, [100.0, 200.0, 300.0], [60.0, 110.0, 160.0], None
    )
    assert value is None
    assert "unweighted" in reason


def test_a_zero_derivative_is_reported_rather_than_dividing_by_zero():
    """A quadratic's vertex has dE/dch == 0, so a point sitting exactly
    there has no finite energy uncertainty.

    FOUR channels, not three: a quadratic has p == 3, so three points
    would return "no degrees of freedom" before the derivative is ever
    evaluated, and the test would pass while exercising nothing.
    """
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=-0.005)
    vertex = 100.0  # -b / (2c)
    channels = [50.0, vertex, 150.0, 200.0]
    energies = [cal.apply(c) for c in channels]
    value, reason = reduced_chi_squared(cal, channels, energies, [0.1] * 4)
    assert value is None
    assert "turning point" in reason



# --- the line's own energy uncertainty counts too ------------------------


def _drifting(cal, channels, drift):
    return [cal.apply(c) + d for c, d in zip(channels, drift)]


def test_the_lines_own_uncertainty_lowers_the_reported_value():
    """The residual has two sources and the test is only fair if it
    counts both. A line quoted to a keV cannot be held to the precision
    of a centroid measured to a hundredth of a channel."""
    cal = Calibration(kind="linear", a=0.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0, 500.0]
    energies = _drifting(cal, channels, (0.05, -0.04, 0.06, -0.05, 0.04))
    tight = [0.01] * 5
    without, _r = reduced_chi_squared(cal, channels, energies, tight)
    with_dE, _r = reduced_chi_squared(cal, channels, energies, tight, [0.05] * 5)
    assert without > with_dE > 0.0


def test_no_energy_errors_reproduces_the_old_value_exactly():
    """CONTROL: the new term must be inert when there is nothing to add,
    or every calibration in the archive would have changed meaning."""
    cal = Calibration(kind="linear", a=0.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0]
    energies = _drifting(cal, channels, (0.05, -0.03, 0.02, 0.10))
    errors = [0.02, 0.03, 0.02, 0.04]
    plain, _r = reduced_chi_squared(cal, channels, energies, errors)
    zeros, _r = reduced_chi_squared(cal, channels, energies, errors, [0.0] * 4)
    assert plain == pytest.approx(zeros)


def test_the_value_matches_the_two_term_formula():
    cal = Calibration(kind="linear", a=0.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0, 500.0]
    energies = _drifting(cal, channels, (0.05, -0.03, 0.02, 0.10, -0.07))
    sigma_ch = [0.02, 0.03, 0.02, 0.04, 0.05]
    sigma_e = [0.01, 0.20, 0.0, 0.005, 1.0]
    expected = sum(
        (e - cal.apply(c)) ** 2 / ((s * cal.derivative(c)) ** 2 + se ** 2)
        for c, e, s, se in zip(channels, energies, sigma_ch, sigma_e)
    ) / (len(channels) - 2)
    value, reason = reduced_chi_squared(cal, channels, energies, sigma_ch, sigma_e)
    assert reason is None
    assert value == pytest.approx(expected)


def test_a_line_uncertainty_rescues_a_point_at_the_turning_point():
    """dE/dch is zero at a quadratic's vertex, so the centroid there maps
    to no energy uncertainty at all -- but the LINE still has one, and
    that is enough to weigh the point by."""
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=-0.005)
    channels = [50.0, 100.0, 150.0, 200.0]      # 100 is the vertex
    energies = [cal.apply(c) for c in channels]
    value, reason = reduced_chi_squared(cal, channels, energies, [0.1] * 4)
    assert value is None and "turning point" in reason
    value, reason = reduced_chi_squared(cal, channels, energies, [0.1] * 4,
                                        [0.02] * 4)
    assert reason is None
    assert value == pytest.approx(0.0, abs=1e-12)


def test_unusable_line_uncertainties_count_as_zero_not_as_a_refusal():
    """Unlike a channel error, which would carry infinite weight and so
    invalidates the whole weighting, a missing energy uncertainty is an
    honest statement that the line adds nothing of its own."""
    cal = Calibration(kind="linear", a=0.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0]
    energies = _drifting(cal, channels, (0.05, -0.03, 0.02, 0.10))
    errors = [0.02] * 4
    plain, _r = reduced_chi_squared(cal, channels, energies, errors)
    for bad in ([float("nan")] * 4, [-1.0] * 4, [None] * 4, ["x"] * 4):
        value, reason = reduced_chi_squared(cal, channels, energies, errors, bad)
        assert reason is None
        assert value == pytest.approx(plain)


def test_a_wrong_length_list_of_line_uncertainties_is_ignored():
    cal = Calibration(kind="linear", a=0.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0]
    energies = _drifting(cal, channels, (0.05, -0.03, 0.02, 0.10))
    errors = [0.02] * 4
    plain, _r = reduced_chi_squared(cal, channels, energies, errors)
    value, reason = reduced_chi_squared(cal, channels, energies, errors, [0.05])
    assert reason is None
    assert value == pytest.approx(plain)


# --- both point lists have to correspond ------------------------------


def test_mismatched_energies_length_is_refused_not_silently_truncated():
    """channel_errors' length was checked and energies' was not, so a short
    list made zip() stop early while the divisor kept the full count. The
    result was not an error but a real number that was simply too small --
    the failure mode the module's (None, reason) contract exists to avoid.
    """
    cal = Calibration(kind="linear", a=0.0, b=1.0)
    channels = [10.0, 20.0, 30.0, 40.0]
    value, reason = reduced_chi_squared(
        cal, channels, [10.0, 20.0], [0.1] * 4
    )
    assert value is None
    assert "correspond" in reason


def test_matching_lengths_still_compute():
    """Control: the guard must not refuse well-formed input."""
    cal = Calibration(kind="linear", a=0.0, b=1.0)
    channels = [10.0, 20.0, 30.0, 40.0]
    value, _reason = reduced_chi_squared(
        cal, channels, [10.5, 20.5, 29.5, 40.5], [0.1] * 4
    )
    assert value is not None and value > 0.0
