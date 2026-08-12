import numpy as np
import pytest

from matrix_cut import compute_cut, compute_projection


def _small_matrix():
    # 4 rows (Y: 0..3) x 5 columns (X: 0..4), simple deterministic
    # values so projections/cuts are easy to hand-verify.
    return np.array(
        [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [-1, -2, -3, -4, -5],
            [100, 100, 100, 100, 100],
        ],
        dtype=np.int64,
    )


def test_compute_projection_x_sums_over_rows():
    matrix = _small_matrix()
    proj = compute_projection(matrix, "x")
    assert list(proj) == [110, 120, 130, 140, 150]  # column sums


def test_compute_projection_y_sums_over_columns():
    matrix = _small_matrix()
    proj = compute_projection(matrix, "y")
    assert list(proj) == [15, 150, -15, 500]  # row sums


def test_compute_projection_invalid_axis_raises():
    matrix = _small_matrix()
    with pytest.raises(ValueError):
        compute_projection(matrix, "z")


def test_compute_cut_gating_on_x_produces_y_indexed_result():
    # Gate on columns 1-2 (X range), no background -- net = raw sum of
    # those two columns, indexed by row (Y).
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(1, 2), bg_regions=[])
    assert list(result) == [5, 50, -5, 200]  # row-wise sum of columns 1,2


def test_compute_cut_gating_on_y_produces_x_indexed_result():
    # Gate on rows 0-1 (Y range), no background -- net = raw sum of
    # those two rows, indexed by column (X).
    matrix = _small_matrix()
    result = compute_cut(matrix, "y", cut_region=(0, 1), bg_regions=[])
    assert list(result) == [11, 22, 33, 44, 55]  # column-wise sum of rows 0,1


def test_compute_cut_zero_background_regions_means_no_subtraction():
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[])
    assert list(result) == [1, 10, -1, 100]  # just column 0, unsubtracted


def test_compute_cut_applies_gate_width_weighted_subtraction():
    # cut_region (0,1) on axis 'x' -> width 2, sums columns 0+1 per row.
    # bg_regions [(3,4)] on axis 'x' -> width 2, sums columns 3+4 per row.
    # Equal widths (2/2=1.0) -> net = pos - bg exactly.
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 1), bg_regions=[(3, 4)])
    # pos (cols 0+1 per row): [3, 30, -3, 200]
    # bg  (cols 3+4 per row): [9, 90, -9, 200]
    # net = pos - (2/2)*bg = pos - bg
    assert list(result) == [-6, -60, 6, 0]


def test_compute_cut_unequal_width_background_scales_correctly():
    # cut_region (0,0) width=1; bg_regions [(2,4)] width=3.
    # pos (col 0): [1, 10, -1, 100]
    # bg (cols 2+3+4 summed per row): [12, 120, -12, 300]
    # net = pos - (1/3)*bg
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(2, 4)])
    expected = [1 - 12 / 3, 10 - 120 / 3, -1 - (-12) / 3, 100 - 300 / 3]
    assert result == pytest.approx(expected)


def test_compute_cut_multiple_background_regions_are_pooled():
    # Two separate bg regions, total width 2 (one col each) -- summed
    # together (not weighted per-region) before the shared ratio is
    # applied, matching TV's own gate-width weighting.
    matrix = _small_matrix()
    result_combined = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 3), (4, 4)])
    result_single = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4)])
    assert result_combined == pytest.approx(result_single)


def test_compute_cut_preserves_negative_results_unclamped():
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 1), bg_regions=[(3, 4)])
    assert any(v < 0 for v in result)  # confirmed by the prior test's exact values


def test_compute_cut_rounds_half_away_from_zero_like_tv():
    # Region bounds land on a .5 boundary -- TV's NINT rounds
    # half-away-from-zero (0.5 -> 1, -0.5 -> -1), not Python's default
    # round-half-to-even. Using (0.5, 1.5) on a 5-wide axis should
    # select columns 1 and 2 (round(0.5)->1, round(1.5)->2 per NINT).
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0.5, 1.5), bg_regions=[])
    assert list(result) == [5, 50, -5, 200]  # same as columns 1,2 exactly
