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


def test_compute_cut_invalid_axis_raises():
    matrix = _small_matrix()
    with pytest.raises(ValueError):
        compute_cut(matrix, "z", cut_region=(0, 0), bg_regions=[])


def test_compute_cut_out_of_range_cut_region_contributes_zero():
    # cut_region entirely past the last column (valid range 0..4) --
    # the gate selects no real data, so net must be all zeros rather
    # than crashing or silently wrapping to some other column range.
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(10, 20), bg_regions=[])
    assert list(result) == [0, 0, 0, 0]


def test_compute_cut_out_of_range_background_region_does_not_corrupt_result():
    # bg_regions mixes a valid region (3,4) with one entirely past the
    # last column (10,20). A region with no channel inside the matrix
    # contributes neither counts nor width, so the result is identical to
    # using the valid region alone.
    #
    # This is a DELIBERATE DIVERGENCE FROM TV, chosen by the user.
    # TV's MrkRange (vsMark.c:29-37) counts every marked region's width
    # whether or not it overlaps the data, so there a stray region
    # silently weakens the subtraction. Being ignored is more predictable
    # for a mark that plainly has no data under it. See
    # test_a_background_region_fully_outside_is_ignored_unlike_tv, which
    # pins the divergence explicitly against the TV oracle.
    matrix = _small_matrix()
    result_mixed = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4), (10, 20)])
    result_valid_only = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4)])
    assert result_mixed == pytest.approx(result_valid_only)


def test_compute_cut_out_of_range_background_region_does_not_divide_by_zero():
    # The original hazard: with widths derived from CLAMPED indices,
    # (7,100) resolved to lo_idx=7 > hi_idx=4, a raw width of -2, which
    # cancelled (3,4)'s +2 and left bg_width at exactly 0 -- a
    # ZeroDivisionError. Marked widths cannot do this: they come straight
    # from the marks (NINT(x2)-NINT(x1)+1) and are always at least 1, so
    # a total of zero is unreachable and no per-region flooring is needed.
    #
    # (7,100) has no channel inside a 5-column matrix, so it is skipped
    # outright and the result matches the valid region alone -- and,
    # separately, nothing can divide by zero.
    matrix = _small_matrix()
    result_mixed = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4), (7, 100)])
    result_valid_only = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4)])
    assert np.all(np.isfinite(result_mixed))
    assert result_mixed == pytest.approx(result_valid_only)


def test_compute_cut_all_background_regions_out_of_range_means_no_subtraction():
    # A single bg region entirely outside the matrix's bounds -- unlike
    # the "does_not_divide_by_zero" test above, there is no companion
    # valid region here, so bg_width totals exactly 0 even after
    # per-region flooring (bg_regions is non-empty, so the earlier
    # "not bg_regions" guard does not apply). A bg_width of 0 can only
    # happen when every bg region's floored width -- and therefore its
    # sum -- is zero, so the correct result is pos unchanged, not a
    # ZeroDivisionError from dividing by that 0.
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(100, 200)])
    assert list(result) == [1, 10, -1, 100]  # same as pos, unsubtracted


# --- TV parity of the gate-width-weighted cut (v3.1.3) -----------------


def _nint(value):
    return int(np.floor(value + 0.5)) if value >= 0 else -int(np.floor(-value + 0.5))


def _tv_cut(matrix, axis, cut_region, bg_regions):
    """TV's algorithm transcribed literally from the C source, as an
    independent oracle rather than a restatement of ours:

      vsCut.c:80-99   CutCreateSpectrumUpdate -- sum of fac_i * projection_i
      VsCut.c:466-482 Cut_CreateFacGate -- signal fac 1.0, every background
                      fac = -(MrkRange(gate) / MrkRange(bgGate))
      vsMark.c:29-37  MrkRange -- sum(NINT(x2)-NINT(x1)) + region count,
                      computed on the MARKS, unclamped
      vsSpectra.c:962-981 SpcProject -- sums NINT(x1)..NINT(x2) inclusive,
                      clamped to [0, lines); contributes nothing if the
                      clamp leaves the range empty
    """
    lines = matrix.shape[0] if axis == "y" else matrix.shape[1]
    complementary = matrix.shape[1] if axis == "y" else matrix.shape[0]

    def project(region):
        lo = max(0, _nint(region[0]))
        hi = min(_nint(region[1]) + 1, lines)
        if hi <= lo:
            return np.zeros(complementary)
        return (matrix[lo:hi, :].sum(axis=0) if axis == "y"
                else matrix[:, lo:hi].sum(axis=1))

    def mrk_range(regions):
        return sum(_nint(b) - _nint(a) for a, b in regions) + len(regions)

    out = project(cut_region).astype(float)
    if bg_regions:
        factor = -(mrk_range([cut_region]) / mrk_range(bg_regions))
        for region in bg_regions:
            out = out + factor * project(region)
    return out


def test_cut_matches_tv_for_regions_overhanging_the_matrix_edge():
    """The gate-width factor must come from the MARKS, not from bounds
    clamped to the matrix. Clamping the widths shrank the denominator of
    pos_width/bg_width, inflated the factor and over-subtracted: a single
    background region hanging over the low edge produced a net spectrum
    about 7% of TV's.
    """
    rng = np.random.default_rng(11)
    matrix = rng.integers(0, 50, size=(300, 300)).astype(np.int64)

    for cut, bgs in (
        ((100.0, 150.0), [(-10.0, 20.0)]),
        ((100.0, 150.0), [(280.0, 340.0)]),
        ((280.0, 340.0), [(20.0, 40.0)]),
        ((100.0, 150.0), [(-20.0, 10.0), (290.0, 330.0)]),
    ):
        ours = compute_cut(matrix, "y", cut, bgs).astype(float)
        assert np.allclose(ours, _tv_cut(matrix, "y", cut, bgs), atol=1e-9), (
            f"diverges from TV for cut={cut} bg={bgs}"
        )


def test_a_background_region_below_channel_zero_contributes_nothing():
    """Clamping left lo_idx at 0 but hi_idx NEGATIVE, and matrix[0:-3] is
    not an empty slice in numpy -- a negative stop counts back from the
    end, so a background region marked entirely below channel 0 summed
    almost the whole matrix and injected millions of counts of garbage as
    "background", which was then subtracted. Marks past the HIGH edge
    were always safe (lo > hi gives a genuinely empty slice).
    """
    matrix = np.ones((200, 120), dtype=np.int64)

    # Entirely below the matrix: must behave exactly like no background.
    below = compute_cut(matrix, "y", (50.0, 100.0), [(-30.0, -5.0)])
    none = compute_cut(matrix, "y", (50.0, 100.0), [])
    assert np.array_equal(below, none)

    # And the same for the complementary axis.
    below_x = compute_cut(matrix, "x", (10.0, 40.0), [(-25.0, -2.0)])
    none_x = compute_cut(matrix, "x", (10.0, 40.0), [])
    assert np.array_equal(below_x, none_x)


def test_cut_matches_tv_across_randomised_marks():
    """Sweep with marks deliberately spilling past both edges, on a
    matrix that includes negative counts (Subtract-Spectra results and
    random-coincidence subtraction both produce them).

    Background regions are generated so each one OVERLAPS the matrix.
    That is the domain where TV parity is claimed: a background region
    with no overlap at all is deliberately ignored here and counted by
    TV, which
    test_a_background_region_fully_outside_is_ignored_unlike_tv covers
    instead. Constraining the sweep keeps it a real parity check rather
    than one weakened to accommodate a known, intended difference.
    """
    rng = np.random.default_rng(7)
    for matrix in (rng.integers(0, 50, size=(300, 300)).astype(np.int64),
                   rng.integers(-20, 80, size=(180, 240)).astype(np.int64)):
        size = matrix.shape[0]
        for _ in range(60):
            axis = "y" if rng.random() < 0.5 else "x"
            limit = matrix.shape[0] if axis == "y" else matrix.shape[1]
            cut = tuple(sorted(rng.uniform(-40, limit + 40, 2)))
            bgs = []
            for _ in range(int(rng.integers(0, 4))):
                # One end inside the matrix guarantees overlap while
                # still letting the other end hang past an edge.
                inside = rng.uniform(0, limit - 1)
                other = rng.uniform(-40, limit + 40)
                bgs.append(tuple(sorted((inside, other))))
            ours = compute_cut(matrix, axis, cut, bgs).astype(float)
            assert np.allclose(ours, _tv_cut(matrix, axis, cut, bgs), atol=1e-8), (
                f"diverges: axis={axis} cut={cut} bg={bgs}"
            )
        del size


def test_a_background_region_fully_outside_is_ignored_unlike_tv():
    """The one deliberate divergence from TV in this module.

    TV counts a background region's marked width even when the region has
    no channel inside the matrix (MrkRange, vsMark.c:29-37, works purely
    on the marks), so there a stray mark quietly weakens the subtraction.
    Here it is skipped entirely. Pinned against the TV oracle so the
    difference stays visible and intentional rather than becoming an
    unexplained mismatch for whoever next compares the two.
    """
    rng = np.random.default_rng(23)
    matrix = rng.integers(0, 50, size=(300, 300)).astype(np.int64)
    cut = (100.0, 150.0)
    valid = [(20.0, 40.0)]

    for stray in ((400.0, 450.0), (-50.0, -10.0)):
        ours = compute_cut(matrix, "y", cut, valid + [stray]).astype(float)
        ignored = compute_cut(matrix, "y", cut, valid).astype(float)
        assert np.allclose(ours, ignored), "a fully-outside region must be ignored"
        assert not np.allclose(ours, _tv_cut(matrix, "y", cut, valid + [stray])), (
            "TV counts the stray region's width; this divergence is intended, "
            "so the two must NOT agree here"
        )

    # Every background region outside means no usable background at all.
    none_usable = compute_cut(matrix, "y", cut, [(400.0, 450.0), (-50.0, -10.0)])
    assert np.array_equal(none_usable, compute_cut(matrix, "y", cut, []))
    assert np.all(np.isfinite(none_usable))
