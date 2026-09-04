"""Tests for the compact uncertainty notation, e.g. 352.7217(14).

The parenthesised digits are the uncertainty expressed in units of the
last decimal place shown, the convention used in nuclear data tables.
"""

import math

import pytest

from value_format import compact


def test_basic_two_significant_digits():
    # 0.00014 -> "14" in the last two decimal places of 352.72172.
    assert compact(352.72172, 0.00014) == "352.72172(14)"


def test_uncertainty_rounded_to_two_significant_digits():
    # 0.00123 -> 0.0012, so the value is shown to 4 decimals.
    assert compact(12.345678, 0.00123) == "12.3457(12)"


def test_large_uncertainty_moves_the_decimal_place():
    # 3.7 -> two sig digits is "3.7", one decimal place.
    assert compact(1234.5678, 3.7) == "1234.6(37)"


def test_uncertainty_of_ten_or_more():
    # 42.0 -> "42", zero decimal places, so no decimal point at all.
    assert compact(1234.5678, 42.0) == "1235(42)"


def test_rounding_carry_in_the_uncertainty():
    # 0.0099 rounds to 0.0099 at two sig digits (not 0.01).
    assert compact(5.0, 0.0099) == "5.0000(99)"


def test_a_carry_that_crosses_a_power_of_ten_moves_the_decimal_place():
    """The one branch the other cases never reach.

    9.99 at two significant digits is 10, not 9.9, so the scale shifts
    from tenths to units and the value must follow it. Without the
    recompute this would render "352.7(100)", quoting three digits of
    uncertainty against a value at the wrong precision.
    """
    assert compact(352.7217, 9.99) == "353(10)"


def test_zero_error_gives_a_bare_value():
    """A position held fixed reports exactly 0.0. Printing '(0)' would
    read as a perfectly known value."""
    assert compact(352.7217, 0.0) == "352.7217"


def test_negative_error_gives_a_bare_value():
    assert compact(352.7217, -1.0) == "352.7217"


def test_nan_error_gives_a_bare_value():
    """peak_fit reports NaN for a parameter the data does not constrain."""
    assert compact(352.7217, float("nan")) == "352.7217"


def test_none_error_gives_a_bare_value():
    assert compact(352.7217, None) == "352.7217"


def test_error_larger_than_value_is_still_shown():
    """Unusual but real. Hiding it would be worse than showing it.

    sigma 2.0 sets the scale to one decimal, so the value keeps its own
    decimal too: "0.5(20)", not "0(20)".
    """
    assert compact(0.5, 2.0) == "0.5(20)"


def test_non_finite_value_is_an_em_dash():
    assert compact(float("nan"), 0.1) == "—"
    assert compact(float("inf"), 0.1) == "—"


def test_negative_value_keeps_its_sign():
    assert compact(-12.345678, 0.00123) == "-12.3457(12)"


def test_bare_value_of_zero_error_is_not_over_rounded():
    """With no usable uncertainty there is nothing to set the precision,
    so fall back to a readable fixed number of decimals."""
    assert compact(1234.5, 0.0) == "1234.5"


def test_uncertainty_of_a_hundred_or_more_is_not_understated():
    """The Volume column shows peak areas, where a Poisson-like error
    passes 100 at around ten thousand counts. An earlier version divided
    by the quantum instead of the printed decimal place and rendered
    170 as "(17)", understating every large peak's area error tenfold.
    """
    assert compact(20000.0, 170.0) == "20000(170)"
    assert compact(100000.0, 380.0) == "100000(380)"


def test_a_very_large_uncertainty_keeps_its_magnitude():
    assert compact(2000000.0, 1697.0) == "2000000(1700)"
