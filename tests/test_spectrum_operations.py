import numpy as np

from spectrum_operations import multiply


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
