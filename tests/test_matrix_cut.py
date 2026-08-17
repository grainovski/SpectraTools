import numpy as np
import pytest

from matrix_cut import (
    as_region_list,
    compute_cut,
    compute_cut_with_variance,
    compute_projection,
    merge_regions,
)


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
    # Two separate bg regions, two lines in total (one column each) --
    # summed together and weighted by one shared ratio, not weighted
    # per-region. TV and HDTV agree on that much; they differ only over
    # where the ratio's width comes from.
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
    # This diverges from TV, whose MrkRange (vsMark.c:29-37) counts every
    # marked region's width whether or not it overlaps the data, so a
    # stray region there silently weakens the subtraction.
    #
    # It used to be an explicit special case. It now falls out of the
    # uniform weighting rule (see compute_cut): a region with no lines
    # inside the matrix contributes no lines, so it cannot contribute
    # width either. See
    # test_a_background_region_fully_outside_is_ignored_unlike_tv, which
    # pins the divergence explicitly against the TV oracle.
    matrix = _small_matrix()
    result_mixed = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4), (10, 20)])
    result_valid_only = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4)])
    assert result_mixed == pytest.approx(result_valid_only)


def test_compute_cut_out_of_range_background_region_does_not_divide_by_zero():
    # The original hazard: with widths derived from CLAMPED indices,
    # (7,100) resolved to lo_idx=7 > hi_idx=4, a raw width of -2, which
    # cancelled (3,4)'s +2 and left the total at exactly 0 -- a
    # ZeroDivisionError.
    #
    # Unreachable now for a structural reason rather than an arithmetic
    # one: merge_regions drops a region with nothing inside the matrix, so
    # (7,100) never reaches the summing loop at all, and every range that
    # does covers at least one line. A line count therefore cannot be
    # negative, and can only total zero when there are no usable
    # background regions whatsoever -- which the caller handles by not
    # subtracting.
    matrix = _small_matrix()
    result_mixed = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4), (7, 100)])
    result_valid_only = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4)])
    assert np.all(np.isfinite(result_mixed))
    assert result_mixed == pytest.approx(result_valid_only)


def test_compute_cut_all_background_regions_out_of_range_means_no_subtraction():
    # A single bg region entirely outside the matrix -- unlike the
    # "does_not_divide_by_zero" test above there is no companion valid
    # region, so the background line count is exactly 0 even though
    # bg_regions is non-empty. That can only happen when every region was
    # dropped for having no overlap, in which case there is no usable
    # background at all and the right answer is pos unchanged, not a
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


def _hdtv_cut(matrix, axis, cut_regions, bg_regions):
    """HDTV's algorithm transcribed literally from the C++ source, as an
    independent oracle rather than a restatement of ours:

      VMatrix.cxx:25-61   AddRegion -- normalise with Min/Max, drop a
                          region wholly outside [lowBin, highBin], clip
                          the rest, then insert into a sorted boundary
                          list with a parity walk that MERGES overlaps
      VMatrix.cxx:105-114 Cut -- walk the boundary list in pairs, adding
                          each line and counting it; bgFac = nCut/nBg
                          (0 when nBg is 0); subtract bg*bgFac

    The boundary-list mechanics are HDTV's, deliberately not the
    sort-and-coalesce our merge_regions uses, so agreement means the two
    approaches agree rather than that one restates the other.

    One substitution: mark-to-index conversion uses this project's _nint
    rather than HDTV's `ceil(x - 0.5)`. HDTV's own two backends disagree
    on that (RMatrix goes through TAxis::FindBin), it differs only on an
    exact half-channel, and it is not what these tests are checking.
    """
    lines = matrix.shape[0] if axis == "y" else matrix.shape[1]
    complementary = matrix.shape[1] if axis == "y" else matrix.shape[0]

    def accumulate(regions):
        reglist = _hdtv_boundary_list(regions, 0, lines - 1)
        total = np.zeros(complementary)
        count = 0
        for k in range(0, len(reglist), 2):
            for line in range(reglist[k], reglist[k + 1] + 1):
                total = total + (matrix[line, :] if axis == "y" else matrix[:, line])
                count += 1
        return total, count

    pos, n_cut = accumulate(as_region_list(cut_regions))
    bg, n_bg = accumulate(as_region_list(bg_regions))
    factor = 0.0 if n_bg == 0 else n_cut / n_bg
    return pos - bg * factor


def _hdtv_boundary_list(regions, low_bin, high_bin):
    """HDTV's VMatrix::AddRegion (src/mfile-root/VMatrix.cxx:25-61),
    transcribed literally: normalise with Min/Max, drop a region wholly
    outside [low_bin, high_bin], clip the rest, then insert into a sorted
    boundary list with a parity walk that merges overlaps. Returns the
    flat list, read in pairs.

    Transcribed faithfully INCLUDING A BUG, which is why it is exposed
    separately -- see
    test_hdtv_double_counts_a_channel_where_two_regions_touch. HDTV's
    walk is order-dependent, and when a newly added region's end lands
    exactly on an existing region's start it emits a touching pair like
    [0, 167, 167, 299] instead of [0, 299]. Because Cut() then iterates
    each pair inclusively (`for(l=l1; l<=l2; l++)`), the shared channel
    is summed and counted twice -- contradicting the intent its sibling
    PolyBg::AddRegion states outright, that values covered by two or
    more regions are considered only once.
    """
    reglist = []
    for a, b in regions:
        lo, hi = _nint(min(a, b)), _nint(max(a, b))
        if hi < low_bin or lo > high_bin:
            continue
        lo, hi = max(lo, low_bin), min(hi, high_bin)
        inside = False
        i = 0
        while i < len(reglist) and reglist[i] < lo:
            inside = not inside
            i += 1
        if not inside:
            reglist.insert(i, lo)
            i += 1
        while i < len(reglist) and reglist[i] < hi:
            inside = not inside
            reglist.pop(i)
        if not inside:
            reglist.insert(i, hi)
    return reglist


def _hdtv_pairs_are_disjoint(regions, limit):
    """False when HDTV's own boundary list double-counts a channel, i.e.
    when its output pairs touch or overlap. Used to skip sweep cases
    where the oracle contradicts itself and there is nothing meaningful
    to compare against."""
    reglist = _hdtv_boundary_list(regions, 0, limit - 1)
    starts, ends = reglist[0::2], reglist[1::2]
    return all(ends[k] < starts[k + 1] for k in range(len(starts) - 1))


def test_hdtv_double_counts_a_channel_where_two_regions_touch():
    """A defect in the ancestor, recorded because our merge deliberately
    does not reproduce it.

    HDTV's boundary walk is order-dependent. Adding (0, 167) to a list
    that already holds (167, 299) leaves the pairs touching at channel
    167, and Cut() sums each pair inclusively, so that channel is summed
    and counted twice -- 301 background lines where there are only 300.
    Supplying the same two regions in the other order merges correctly.

    Ours sorts before coalescing, so it is order-independent and counts
    the shared channel once in both orders. That is the behaviour
    PolyBg::AddRegion's own comment asks for, so this is a divergence
    from HDTV's code but not from HDTV's intent.
    """
    matrix = np.zeros((300, 300))

    assert _hdtv_boundary_list([(167, 299), (0, 167)], 0, 299) == [0, 167, 167, 299]
    assert _hdtv_boundary_list([(0, 167), (167, 299)], 0, 299) == [0, 299]

    assert merge_regions(matrix, "y", [(167, 299), (0, 167)]) == [(0, 299)]
    assert merge_regions(matrix, "y", [(0, 167), (167, 299)]) == [(0, 299)]


def test_overhanging_background_leaves_no_residual_on_a_flat_matrix():
    """The measured evidence for the C1 change, on the one matrix whose
    correct answer is known without reference to any implementation: if
    every channel holds the same counts, a gate-width-weighted cut must
    subtract to exactly zero whatever regions are marked.

    TV's rule fails that. It divides counts drawn from the CLIPPED range
    by a width drawn from the UNCLIPPED marks (MrkRange, vsMark.c:29-37,
    knows nothing about the matrix), so a background mark overhanging an
    edge under-weights the subtraction.
    """
    matrix = np.full((300, 300), 100.0)
    cut = (100, 109)          # 10 lines, fully inside
    bg = [(-5, 14)]           # 20 lines marked, only 0..14 (15) inside

    assert compute_cut(matrix, "y", cut, bg) == pytest.approx(np.zeros(300))

    # What TV's rule gives instead: 10 cut lines against a width of 20
    # leaves a quarter of the gross unsubtracted.
    tv = _tv_cut(matrix, "y", cut, bg)
    assert tv == pytest.approx(np.full(300, 250.0))
    gross = 10 * 100
    assert tv[0] / gross == pytest.approx(0.25)


def test_cut_diverges_from_tv_for_regions_overhanging_the_matrix_edge():
    """The C1 divergence, pinned so it stays visible and intentional.

    These are the cases that used to assert TV parity. The weighting now
    follows HDTV -- both the counts and the line count come from the
    lines actually summed -- so it must match the HDTV oracle and must
    NOT match TV for any region that overhangs an edge.
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
        assert np.allclose(ours, _hdtv_cut(matrix, "y", cut, bgs), atol=1e-9), (
            f"diverges from HDTV for cut={cut} bg={bgs}"
        )
        assert not np.allclose(ours, _tv_cut(matrix, "y", cut, bgs)), (
            f"TV parity is deliberately broken here (cut={cut} bg={bgs}); "
            "the two must NOT agree"
        )


def test_cut_matches_hdtv_across_randomised_marks():
    """The parity claim that replaces the TV sweep: agreement with HDTV
    across marks spilling past both edges AND overlapping each other --
    the two domains where TV parity was given up.

    Includes a matrix with negative counts, since Subtract-Spectra
    results and random-coincidence subtraction both produce them.
    """
    rng = np.random.default_rng(101)
    compared = 0
    skipped = 0
    for matrix in (rng.integers(0, 50, size=(300, 300)).astype(np.int64),
                   rng.integers(-20, 80, size=(180, 240)).astype(np.int64)):
        for _ in range(80):
            axis = "y" if rng.random() < 0.5 else "x"
            limit = matrix.shape[0] if axis == "y" else matrix.shape[1]
            # Several foreground gates as well, so C4's accumulation is
            # covered by the parity check rather than only by unit tests.
            cuts = [tuple(sorted(rng.uniform(-40, limit + 40, 2)))
                    for _ in range(int(rng.integers(1, 4)))]
            bgs = [tuple(sorted(rng.uniform(-40, limit + 40, 2)))
                   for _ in range(int(rng.integers(0, 4)))]

            # Skip the cases where HDTV's own boundary walk double-counts a
            # channel, since its output is self-contradictory there and
            # agreement would mean reproducing its bug. See
            # test_hdtv_double_counts_a_channel_where_two_regions_touch.
            if not (_hdtv_pairs_are_disjoint(cuts, limit)
                    and _hdtv_pairs_are_disjoint(bgs, limit)):
                skipped += 1
                continue

            ours = compute_cut(matrix, axis, cuts, bgs).astype(float)
            assert np.allclose(ours, _hdtv_cut(matrix, axis, cuts, bgs), atol=1e-8), (
                f"diverges from HDTV: axis={axis} cut={cuts} bg={bgs}"
            )
            compared += 1

    # Guard against the skip condition quietly swallowing the whole sweep
    # and leaving a test that asserts nothing.
    assert compared > 100, f"only {compared} cases compared ({skipped} skipped)"


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


def test_cut_still_matches_tv_for_fully_inside_non_overlapping_marks():
    """TV parity survives, NARROWED to the domain where it is still
    claimed rather than loosened to tolerate the two known differences.

    Both divergences need a mark that is unusual in a specific way: C1
    needs one overhanging a matrix edge, C2 needs two that overlap each
    other. Marks that are wholly inside and mutually disjoint -- which is
    what ordinary use produces -- must still agree with TV bit for bit,
    and this sweep is what proves the changes did not reach further than
    intended.

    Deliberately NOT weakened with a looser tolerance or a wider
    generator: a sweep relaxed until it passes stops checking parity at
    all. The two differences are pinned by
    test_cut_diverges_from_tv_for_regions_overhanging_the_matrix_edge and
    test_overlapping_background_regions_diverge_from_tv instead.
    """
    rng = np.random.default_rng(7)
    for matrix in (rng.integers(0, 50, size=(300, 300)).astype(np.int64),
                   rng.integers(-20, 80, size=(180, 240)).astype(np.int64)):
        for _ in range(60):
            axis = "y" if rng.random() < 0.5 else "x"
            limit = matrix.shape[0] if axis == "y" else matrix.shape[1]
            cut = tuple(sorted(rng.uniform(0, limit - 1, 2)))

            # Carve disjoint background regions out of the axis by
            # walking left to right, so no two can overlap.
            bgs = []
            cursor = 0.0
            for _ in range(int(rng.integers(0, 4))):
                lo = cursor + rng.uniform(1.0, 12.0)
                hi = lo + rng.uniform(1.0, 20.0)
                if hi >= limit - 1:
                    break
                bgs.append((lo, hi))
                # +1 clears the inclusive channel the previous region's
                # own NINT(hi) occupies, so the next one cannot touch it.
                cursor = hi + 1.0

            ours = compute_cut(matrix, axis, cut, bgs).astype(float)
            assert np.allclose(ours, _tv_cut(matrix, axis, cut, bgs), atol=1e-8), (
                f"TV parity lost for fully-inside disjoint marks: "
                f"axis={axis} cut={cut} bg={bgs}"
            )


def test_a_background_region_fully_outside_is_ignored_unlike_tv():
    """TV counts a background region's marked width even when the region
    has no channel inside the matrix (MrkRange, vsMark.c:29-37, works
    purely on the marks), so there a stray mark quietly weakens the
    subtraction. Here it is ignored. Pinned against the TV oracle so the
    difference stays visible and intentional rather than becoming an
    unexplained mismatch for whoever next compares the two.

    Introduced in v3.1.4 as a special case on explicit user instruction,
    and no longer implemented as one: with the weighting taken from the
    lines actually summed, a region contributing no lines contributes no
    width by construction. Kept as a test because the BEHAVIOUR is still
    a promise to the user, however it happens to be implemented -- and
    because it inverted once already between v3.1.3 and v3.1.4, so what
    is intended needs stating.
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


# --- C2: overlapping regions are merged (v4.0.0) -----------------------


def _ramped_matrix(size=300):
    """Line L holds 10*L counts in every channel, so WHICH lines a
    background region covers changes the estimate. A flat matrix cannot
    detect double-counting at all -- its mean is the same whichever lines
    are used -- which is why the C1 evidence matrix is flat and this one
    is sloped.
    """
    matrix = np.zeros((size, size))
    for line in range(size):
        matrix[line, :] = 10.0 * line
    return matrix


def test_overlapping_background_regions_count_each_line_once():
    """Nested and asymmetrically-overlapping background marks must give
    the same answer as the single merged region they cover.

    The symmetric case agrees even WITHOUT merging: double-counting
    reweights the background's shape toward the overlap, and that
    cancels exactly when the overlap's mean equals the whole region's
    mean. A test built on a symmetric overlap passes with the bug
    present, so the asymmetric and nested cases are the load-bearing
    ones here.
    """
    matrix = _ramped_matrix()
    cut = (100, 109)
    merged = compute_cut(matrix, "y", cut, [(0, 29)])

    # Hand-derivable: gross = 10*(100+...+109) = 10450; the background
    # mean over lines 0..29 is 10*14.5 = 145 per line, times 10 cut
    # lines = 1450.
    assert merged == pytest.approx(np.full(300, 9000.0))

    for regions in ([(0, 19), (10, 29)],           # symmetric -- cancels anyway
                    [(0, 19), (15, 29)],           # asymmetric
                    [(0, 29), (25, 29)],           # nested
                    [(25, 29), (0, 29)],           # nested, given in reverse order
                    [(0, 9), (5, 14), (10, 29)]):  # chained
        assert compute_cut(matrix, "y", cut, regions) == pytest.approx(merged), (
            f"overlapping regions {regions} did not merge"
        )


def test_overlapping_background_regions_diverge_from_tv():
    """The C2 divergence, pinned with its measured magnitude.

    TV does not merge either, so this is a difference from TV as well as
    a fix -- but unlike C1 there is no reading in which a twice-marked
    channel should count twice, so nothing was traded away for it.
    """
    matrix = _ramped_matrix()
    cut = (100, 109)

    nested = [(0, 29), (25, 29)]
    ours = compute_cut(matrix, "y", cut, nested)
    tv = _tv_cut(matrix, "y", cut, nested)
    assert ours == pytest.approx(np.full(300, 9000.0))
    assert tv == pytest.approx(np.full(300, 8821.428571428572))
    assert float(tv[0] / ours[0] - 1.0) == pytest.approx(-0.0198, abs=5e-4)

    asymmetric = [(0, 19), (15, 29)]
    assert compute_cut(matrix, "y", cut, asymmetric) == pytest.approx(np.full(300, 9000.0))
    assert _tv_cut(matrix, "y", cut, asymmetric) == pytest.approx(np.full(300, 8964.285714285714))


def test_merge_regions_clips_drops_merges_and_normalises():
    matrix = np.zeros((100, 100))

    # Clipped to the matrix, not wrapped.
    assert merge_regions(matrix, "y", [(-10, 5)]) == [(0, 5)]
    assert merge_regions(matrix, "y", [(95, 400)]) == [(95, 99)]

    # Wholly outside, either side -- dropped, so it contributes neither
    # counts nor width. The low side is the one that used to inject
    # garbage counts through a negative slice stop.
    assert merge_regions(matrix, "y", [(-40, -5)]) == []
    assert merge_regions(matrix, "y", [(200, 300)]) == []

    # Merged, and sorted regardless of the order given.
    assert merge_regions(matrix, "y", [(0, 20), (10, 30)]) == [(0, 30)]
    assert merge_regions(matrix, "y", [(10, 30), (0, 20)]) == [(0, 30)]
    assert merge_regions(matrix, "y", [(0, 30), (25, 28)]) == [(0, 30)]

    # Touching at a single channel still merges -- channel 20 would
    # otherwise be summed twice.
    assert merge_regions(matrix, "y", [(0, 20), (20, 30)]) == [(0, 30)]

    # Genuinely disjoint regions stay separate.
    assert merge_regions(matrix, "y", [(0, 10), (20, 30)]) == [(0, 10), (20, 30)]

    # Reversed marks are normalised rather than producing a negative width.
    assert merge_regions(matrix, "y", [(30, 10)]) == [(10, 30)]

    # The two axes are sized independently.
    rect = np.zeros((10, 50))
    assert merge_regions(rect, "y", [(0, 100)]) == [(0, 9)]
    assert merge_regions(rect, "x", [(0, 100)]) == [(0, 49)]


def test_a_reversed_region_is_normalised_rather_than_guarded_against():
    """Marks arrive from drag gestures and can be stored either way
    round. Normalising in merge_regions makes a backwards region simply a
    region, instead of something a downstream negative-width check has to
    catch.
    """
    matrix = _ramped_matrix()
    forwards = compute_cut(matrix, "y", (100, 109), [(0, 29)])
    backwards = compute_cut(matrix, "y", (109, 100), [(29, 0)])
    assert backwards == pytest.approx(forwards)


# --- C4: several foreground gates (v4.0.0) -----------------------------


def test_a_bare_region_and_a_single_element_list_agree():
    """Every existing caller passes a bare (lo, hi) tuple and must keep
    working, so both forms have to be accepted and mean the same thing.
    """
    matrix = _small_matrix()
    assert as_region_list((1, 2)) == [(1, 2)]
    assert as_region_list([(1, 2)]) == [(1, 2)]
    assert as_region_list([]) == []
    assert as_region_list(None) == []
    # A pair of regions is not mistaken for one bare region, despite also
    # having length 2.
    assert as_region_list([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]

    assert compute_cut(matrix, "x", (1, 2), []) == pytest.approx(
        compute_cut(matrix, "x", [(1, 2)], [])
    )


def test_several_foreground_gates_are_summed():
    matrix = _small_matrix()
    both = compute_cut(matrix, "x", [(0, 0), (4, 4)], [])
    apart = (compute_cut(matrix, "x", (0, 0), [])
             + compute_cut(matrix, "x", (4, 4), []))
    assert both == pytest.approx(apart)


def test_several_gates_accumulate_the_cut_line_count():
    """n_cut must total across gates, so two separate one-channel gates
    weight the background exactly as one two-channel gate does.
    """
    matrix = _small_matrix()
    split = compute_cut(matrix, "x", [(0, 0), (1, 1)], [(3, 4)])
    contiguous = compute_cut(matrix, "x", (0, 1), [(3, 4)])
    assert split == pytest.approx(contiguous)
    assert split == pytest.approx([-6, -60, 6, 0])


def test_overlapping_foreground_gates_count_each_line_once():
    matrix = _small_matrix()
    overlapping = compute_cut(matrix, "x", [(0, 2), (1, 3)], [])
    merged = compute_cut(matrix, "x", (0, 3), [])
    assert overlapping == pytest.approx(merged)


def test_a_foreground_gate_outside_the_matrix_contributes_nothing():
    matrix = _small_matrix()
    with_stray = compute_cut(matrix, "x", [(1, 2), (50, 60)], [])
    alone = compute_cut(matrix, "x", (1, 2), [])
    assert with_stray == pytest.approx(alone)


# --- C3: per-channel variance (v4.0.0) ---------------------------------


def test_variance_of_an_unsubtracted_cut_is_its_own_counts():
    """With no background the cut is a plain sum of counts, so Poisson
    applies directly and the variance equals the sum.
    """
    matrix = np.full((10, 6), 4.0)
    net, variance = compute_cut_with_variance(matrix, "y", (0, 4), [])
    assert net == pytest.approx(np.full(6, 20.0))
    assert variance == pytest.approx(np.full(6, 20.0))


def test_variance_is_positive_where_the_net_is_zero():
    """The whole reason this exists. Equal cut and background widths on a
    flat matrix subtract to exactly zero, but the uncertainty on that
    zero is not zero -- it is the two Poisson terms added. Weighting such
    a channel by sqrt(net) is meaningless, and the sqrt(max(y, 1.0))
    floor the fitter used would have called its error 1.
    """
    matrix = np.full((100, 6), 4.0)
    net, variance = compute_cut_with_variance(matrix, "y", (0, 9), [(20, 29)])
    assert net == pytest.approx(np.zeros(6))
    assert variance == pytest.approx(np.full(6, 80.0))  # 40 + 1**2 * 40


def test_variance_scales_the_background_term_by_the_squared_factor():
    matrix = np.full((100, 4), 5.0)
    net, variance = compute_cut_with_variance(matrix, "y", (0, 4), [(20, 39)])
    # 5 cut lines -> pos 25; 20 background lines -> bg 100; factor 0.25.
    assert net == pytest.approx(np.zeros(4))
    assert variance == pytest.approx(np.full(4, 25.0 + 0.25 ** 2 * 100.0))
    # Not the un-squared factor, which would give 25 + 25 = 50.
    assert not np.allclose(variance, np.full(4, 50.0))


def test_variance_agrees_with_compute_cut_and_is_never_negative_or_nan():
    """A matrix can already hold subtracted, negative counts, for which a
    variance read off the counts is not meaningful. Zero is the honest
    floor there rather than a negative variance or a NaN from its square
    root -- and sqrt(variance) is exactly what the fitter will take.
    """
    rng = np.random.default_rng(5)
    matrix = rng.integers(-30, 60, size=(120, 80)).astype(np.int64)
    for cut, bgs in (((10, 30), [(50, 70)]),
                     ((10, 30), []),
                     ((0, 5), [(-10, 4), (100, 140)]),
                     ((10, 30), [(40, 60), (55, 75)])):
        net, variance = compute_cut_with_variance(matrix, "y", cut, bgs)
        assert net == pytest.approx(compute_cut(matrix, "y", cut, bgs))
        assert np.all(variance >= 0.0)
        assert np.all(np.isfinite(variance))
        assert np.all(np.isfinite(np.sqrt(variance)))
