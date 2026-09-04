"""Reduced chi-squared for an energy calibration, and the three cases
where it does not exist."""

import math

import pytest

from calibration import Calibration, from_points
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
