import numpy as np

from spectrum_operations import multiply, normalize_factors, rebin, reference_value


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
