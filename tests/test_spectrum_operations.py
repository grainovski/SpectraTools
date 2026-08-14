import numpy as np
import pytest

from spectrum_operations import add, multiply, normalize_factors, rebin, reference_value, subtract


def test_multiply_scales_and_rounds():
    data = np.array([10, 20, 30], dtype=np.int64)
    result = multiply(data, 2.0)
    assert list(result) == [20, 40, 60]
    assert result.dtype == np.int64


def test_multiply_rounds_fractional_results_to_nearest_integer():
    data = np.array([10, 14], dtype=np.int64)
    result = multiply(data, 1.5)
    # 10*1.5=15.0 and 14*1.5=21.0 are both exact, so this test isn't
    # sensitive to numpy's round-half-to-even tie-breaking (see the
    # dedicated banker's-rounding test below).
    assert list(result) == [15, 21]


def test_multiply_uses_bankers_rounding_on_an_exact_half():
    data = np.array([11], dtype=np.int64)
    result = multiply(data, 1.5)
    # 11*1.5 = 16.5 -- np.round uses round-half-to-even, so this rounds
    # DOWN to 16 (nearest even), not up to 17.
    assert list(result) == [16]


def test_multiply_by_one_is_a_no_op():
    data = np.array([1, 2, 3], dtype=np.int64)
    result = multiply(data, 1.0)
    assert list(result) == [1, 2, 3]


def test_multiply_raises_instead_of_silently_wrapping_on_overflow():
    # A large but perfectly finite, positive factor (passes the dialog's
    # own isfinite/>0 validation) still overflows int64 when multiplied
    # against real counts -- numpy's .astype(np.int64) doesn't raise on
    # overflow at all, it silently wraps to the sentinel
    # -9223372036854775808. Confirmed this really would happen absent
    # the guard: np.round(np.array([100.0]) * 1e20).astype(np.int64)
    # produces exactly that sentinel.
    data = np.array([100, 200, 300], dtype=np.int64)
    with pytest.raises(ValueError):
        multiply(data, 1e20)


def test_multiply_raises_on_exactly_int64_max_plus_one():
    # 2**63 sits exactly on the boundary a naive guard gets wrong:
    # np.iinfo(np.int64).max (2**63 - 1) isn't exactly representable in
    # float64, so a bare `> np.iinfo(np.int64).max` comparison rounds
    # that bound UP to 2**63 and silently lets this exact out-of-range
    # value slip through to the sentinel instead of raising. 2 * 2**62
    # is exactly 2**63 in float64 (a pure power of two only needs
    # exponent range) -- confirmed the old check really would miss it:
    # np.abs(np.array([2.0**63])) > np.iinfo(np.int64).max evaluates to
    # False.
    data = np.array([2], dtype=np.int64)
    with pytest.raises(ValueError):
        multiply(data, 2.0**62)


def test_rebin_sums_evenly_divisible_groups():
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = rebin(data, 2)
    assert list(result) == [3, 7, 11]


def test_rebin_zero_pads_the_final_group_when_not_evenly_divisible():
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rebin(data, 2)
    assert list(result) == [3, 7, 5]


def test_rebin_channel_count_is_ceiling_division():
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rebin(data, 3)
    assert len(result) == 2
    assert list(result) == [6, 9]


def test_rebin_by_factor_larger_than_length_is_a_single_bin():
    data = np.array([1, 2, 3], dtype=np.int64)
    result = rebin(data, 10)
    assert list(result) == [6]


def test_rebin_preserves_int64_dtype():
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rebin(data, 2)
    assert result.dtype == np.int64


def test_reference_value_single_channel():
    data = np.array([10, 20, 30, 40])
    assert reference_value(data, channel=2) == 30


def test_reference_value_channel_out_of_range_is_zero():
    data = np.array([10, 20, 30])
    assert reference_value(data, channel=10) == 0
    assert reference_value(data, channel=-1) == 0


def test_reference_value_region_sums_inclusive():
    data = np.array([10, 20, 30, 40, 50])
    assert reference_value(data, region=(1, 3)) == 90


def test_reference_value_region_clamped_to_bounds():
    data = np.array([10, 20, 30])
    assert reference_value(data, region=(-5, 100)) == 60


def test_reference_value_region_entirely_before_the_start_is_zero():
    data = np.array([10, 20, 30])
    # hi=-2 is deliberate, not just "some negative number": after
    # clamping, hi+1=-1 is still within -len(data), so Python's slice
    # indexing would silently wrap it to a real, non-empty slice
    # (data[0:-1] = [10, 20], sum 30) if the lo > hi guard in
    # reference_value() were ever removed. A more negative hi (e.g.
    # -5) doesn't expose this: -5+1=-4 falls outside -len(data), so
    # the slice clamps to empty on its own and the test would pass
    # whether or not the guard exists.
    assert reference_value(data, region=(-5, -2)) == 0


def test_normalize_factors_scales_up_to_the_maximum():
    factors = normalize_factors([50, 100, 25])
    assert factors == [2.0, 1.0, 4.0]


def test_normalize_factors_skips_zero_values():
    factors = normalize_factors([0, 100, 50])
    assert factors == [None, 1.0, 2.0]


def test_normalize_factors_all_zero_is_a_no_op():
    factors = normalize_factors([0, 0, 0])
    assert factors == [1.0, 1.0, 1.0]


def test_add_sums_with_second_spectrum_scaled_by_factor():
    a = np.array([10, 20, 30], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)
    result = add(a, b, 2.0)
    assert list(result) == [12, 24, 36]
    assert result.dtype == np.int64


def test_add_rounds_fractional_results_to_nearest_integer():
    a = np.array([10, 10], dtype=np.int64)
    b = np.array([1, 3], dtype=np.int64)
    result = add(a, b, 0.5)
    # 10+0.5=10.5 -> 10 (round-half-to-even), 10+1.5=11.5 -> 12 (round-half-to-even)
    assert list(result) == [10, 12]


def test_add_does_not_clamp_negative_results():
    # add() itself never produces negatives from positive inputs and a
    # positive factor, but it must not clamp regardless -- this pins
    # down that no clamping code exists, using a factor large enough
    # that the caller could reasonably combine it with subtract() and
    # expect negatives to survive unchanged through add() too if ever
    # composed. Direct negative-result coverage is on subtract() below,
    # which is the realistic way this app produces negative results.
    a = np.array([0, 0], dtype=np.int64)
    b = np.array([5, -5], dtype=np.int64)
    result = add(a, b, 1.0)
    assert list(result) == [5, -5]


def test_subtract_scales_second_spectrum_before_subtracting():
    a = np.array([10, 20, 30], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)
    result = subtract(a, b, 2.0)
    assert list(result) == [8, 16, 24]
    assert result.dtype == np.int64


def test_subtract_allows_negative_results():
    a = np.array([5, 10], dtype=np.int64)
    b = np.array([10, 5], dtype=np.int64)
    result = subtract(a, b, 1.0)
    assert list(result) == [-5, 5]


def test_subtract_by_factor_one_is_plain_subtraction():
    a = np.array([10, 20], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    result = subtract(a, b, 1.0)
    assert list(result) == [7, 16]


def test_add_raises_instead_of_silently_wrapping_on_overflow():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([100, 200], dtype=np.int64)
    with pytest.raises(ValueError):
        add(a, b, 1e20)


def test_subtract_raises_instead_of_silently_wrapping_on_overflow():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([100, 200], dtype=np.int64)
    with pytest.raises(ValueError):
        subtract(a, b, 1e20)
