import numpy as np
import pytest

from int64_cast import checked_round_to_int64


class _MarkerError(Exception):
    pass


def _checked(rounded):
    return checked_round_to_int64(rounded, lambda: _MarkerError("out of range"))


def test_accepts_ordinary_values():
    result = _checked(np.array([1.0, -2.0, 12345.0]))
    assert list(result) == [1, -2, 12345]
    assert result.dtype == np.int64


def test_accepts_int64_min_exactly():
    # -2**63 is int64's true minimum -- must NOT raise, even though it's
    # bit-identical to the sentinel .astype() silently produces on a
    # genuine overflow. Distinguishing these two is the whole point of
    # hooking the cast's own invalid-value exception instead of
    # pattern-matching the output value.
    result = _checked(np.array([-(2.0**63)]))
    assert int(result[0]) == np.iinfo(np.int64).min


def test_raises_on_exactly_two_to_the_63():
    # np.iinfo(np.int64).max (2**63 - 1) isn't exactly representable in
    # float64, so a naive `> np.iinfo(np.int64).max` comparison rounds
    # that bound UP to 2**63 and silently misses this exact value. This
    # is the boundary bug that motivated this module.
    with pytest.raises(_MarkerError):
        _checked(np.array([2.0**63]))


def test_raises_on_nan():
    # IEEE754: any `>` comparison against NaN is False, so a naive
    # magnitude check silently lets NaN through too.
    with pytest.raises(_MarkerError):
        _checked(np.array([np.nan]))


def test_raises_on_positive_infinity():
    with pytest.raises(_MarkerError):
        _checked(np.array([np.inf]))


def test_raises_on_negative_infinity():
    with pytest.raises(_MarkerError):
        _checked(np.array([-np.inf]))


def test_raises_if_any_single_value_in_the_array_is_out_of_range():
    with pytest.raises(_MarkerError):
        _checked(np.array([1.0, 2.0, 2.0**63, -50.0]))


def test_accepts_an_empty_array():
    result = _checked(np.array([]))
    assert list(result) == []
    assert result.dtype == np.int64
