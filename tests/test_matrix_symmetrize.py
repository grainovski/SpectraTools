"""Symmetrizing a gamma-gamma matrix: S = M' + M'.T after mapping Y onto X.

Matrices here are the app's [y][x] (rows are Y). The design and the
measurements behind every choice tested below are in
docs/superpowers/specs/2026-10-10-matrix-symmetrization-design.md.
"""

import os

import numpy as np
import pytest

from matrix_cut import compute_cut_with_variance, compute_projection
from matrix_symmetrize import (
    SymmetrizeError,
    check_symmetrizable,
    map_from_lines,
    rebin_rows,
    refine_centroid,
    symmetrize,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "compass")


def _co60():
    from json_matrix_io import load_json_matrix
    return load_json_matrix(os.path.join(FIXTURES, "co60_EvsE.json"))


# --- what may be symmetrized ----------------------------------------------


def test_an_asymmetric_square_count_matrix_is_accepted():
    check_symmetrizable(np.array([[1, 2], [0, 3]]))


@pytest.mark.parametrize("matrix, needle", [
    (np.zeros((2, 3), dtype=np.int64), "square"),
    (np.array([[1, -1], [0, 3]]), "negative"),
    (np.array([[1, 2], [2, 3]]), "already symmetric"),
    (np.zeros((0, 0), dtype=np.int64), "empty"),
])
def test_what_cannot_be_symmetrized_is_refused_with_the_reason(matrix, needle):
    with pytest.raises(SymmetrizeError) as err:
        check_symmetrizable(matrix)
    assert needle in str(err.value), str(err.value)


def test_symmetrize_itself_refuses_too():
    with pytest.raises(SymmetrizeError):
        symmetrize(np.array([[1, 2], [2, 3]]))


# --- the identity map: axes already on one scale --------------------------


def test_same_scale_gives_exactly_m_plus_mt_with_the_diagonal_doubled():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 50, size=(30, 30))
    s, d, lost = symmetrize(m)
    assert np.array_equal(s, m + m.T)
    assert np.array_equal(d, np.diag(m))
    assert lost == 0
    assert s.dtype == np.int64
    assert np.array_equal(s, s.T)
    assert int(s.sum()) == 2 * int(m.sum())


# --- rebinning Y onto X's scale ---------------------------------------------


def test_rebin_conserves_counts_and_stays_integer():
    rng = np.random.default_rng(2)
    m = rng.integers(0, 40, size=(64, 64))
    m[60:, :] = 0                                   # nothing near the top edge
    out, lost = rebin_rows(m, a=1.3, b=0.97, seed=7)
    assert out.dtype == np.int64
    assert lost == 0
    assert int(out.sum()) == int(m.sum())
    # Only rows move: every column keeps its total.
    assert np.array_equal(out.sum(axis=0), m.sum(axis=0))


def test_rebin_is_reproducible_by_seed():
    m = np.random.default_rng(3).integers(0, 40, size=(40, 40))
    m[35:, :] = 0
    a1, _ = rebin_rows(m, a=0.4, b=0.93, seed=11)
    a2, _ = rebin_rows(m, a=0.4, b=0.93, seed=11)
    b1, _ = rebin_rows(m, a=0.4, b=0.93, seed=12)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b1), "the seed should matter"


def test_rebin_puts_a_line_where_the_map_says():
    m = np.zeros((300, 300), dtype=np.int64)
    m[200, :] = 1000                                # a line at Y = 200
    out, lost = rebin_rows(m, a=0.0, b=0.5, seed=1)
    rows = out.sum(axis=1)
    assert lost == 0
    # 200 -> 100; bin 200 spans [199.5, 200.5] -> [99.75, 100.25], inside bin 100.
    assert rows[100] == rows.sum()


def test_rebin_splits_by_overlap_fractions_on_average():
    """Each count is assigned at random with the overlap fraction as its
    probability, so the totals follow the fractions within Poisson
    limits."""
    m = np.zeros((10, 1), dtype=np.int64)
    m[4, 0] = 1_000_000
    # b = 1, a = 0.25: bin 4 spans [3.5, 4.5] -> [3.75, 4.75]:
    # 0.75 of it in new bin 4 ([3.5, 4.5]) and 0.25 in new bin 5.
    out, lost = rebin_rows(m, a=0.25, b=1.0, seed=5)
    assert lost == 0
    assert out[4, 0] + out[5, 0] == 1_000_000
    sigma = np.sqrt(1_000_000 * 0.25 * 0.75)
    assert abs(out[5, 0] - 250_000) < 5 * sigma


def test_counts_mapped_off_the_matrix_are_dropped_and_reported():
    m = np.zeros((10, 10), dtype=np.int64)
    m[9, :] = 7                                      # top row
    out, lost = rebin_rows(m, a=5.0, b=1.0, seed=1)  # 9 -> 14: off the edge
    assert lost == 70
    assert int(out.sum()) == 0


@pytest.mark.parametrize("a, b", [(0.0, 0.0), (0.0, -1.0), (float("nan"), 1.0)])
def test_a_degenerate_map_is_refused(a, b):
    with pytest.raises(SymmetrizeError):
        rebin_rows(np.eye(4, dtype=np.int64), a=a, b=b, seed=1)


# --- finding the map from two lines -----------------------------------------


def test_refine_centroid_finds_a_peak_from_a_rough_guess():
    x = np.arange(400)
    spec = 50 + 2000 * np.exp(-0.5 * ((x - 213.4) / 1.4) ** 2)
    assert abs(refine_centroid(spec, 210) - 213.4) < 0.1


def test_map_from_lines_maps_y_lines_onto_x_lines():
    a, b = map_from_lines((148.8, 168.5), (151.9, 172.3))
    assert abs((a + b * 151.9) - 148.8) < 1e-9
    assert abs((a + b * 172.3) - 168.5) < 1e-9


@pytest.mark.parametrize("x_lines, y_lines", [
    ((100, 200), (150, 151)),          # Y lines too close
    ((200, 100), (100, 200)),          # would need a negative gain
])
def test_map_from_lines_refuses_a_useless_pair(x_lines, y_lines):
    with pytest.raises(SymmetrizeError):
        map_from_lines(x_lines, y_lines)


def test_the_co60_map_from_refined_lines():
    m = _co60()
    px, py = compute_projection(m, "x"), compute_projection(m, "y")
    x1, x2 = refine_centroid(px, 149), refine_centroid(px, 169)
    y1, y2 = refine_centroid(py, 152), refine_centroid(py, 172)
    # The Gaussian fits in the investigation gave 148.78/168.54 and 151.87/172.25.
    for got, want in ((x1, 148.78), (x2, 168.54), (y1, 151.87), (y2, 172.25)):
        assert abs(got - want) < 0.3, (got, want)
    a, b = map_from_lines((x1, x2), (y1, y2))
    assert 0.95 < b < 0.99


def _peak_height(p, lo, hi):
    return float(p[lo:hi].max())


def test_matching_first_keeps_the_co60_peaks_sharper_than_a_naive_fold():
    m = _co60()
    px, py = compute_projection(m, "x"), compute_projection(m, "y")
    a, b = map_from_lines(
        (refine_centroid(px, 149), refine_centroid(px, 169)),
        (refine_centroid(py, 152), refine_centroid(py, 172)))
    matched, _d, lost = symmetrize(m, a=a, b=b)
    naive = m + m.T
    assert lost < 0.001 * m.sum()
    pm, pn = compute_projection(matched, "x"), compute_projection(naive, "x")
    # Same total counts within the dropped edge counts, but taller peaks:
    # the two detectors' lines now coincide instead of sitting 3 bins apart.
    for lo, hi in ((140, 160), (160, 182)):
        assert _peak_height(pm, lo, hi) > 1.25 * _peak_height(pn, lo, hi)


# --- the diagonal: self-coincidences and their variance ---------------------


def _self_coincidence_matrix(n_a=200_000, n_b=200_000, seed=42):
    rng = np.random.default_rng(seed)
    n, sig, e, e3 = 512, 1.3, 300.0, 400.0

    def bins(x):
        return np.clip(np.rint(x).astype(int), 0, n - 1)

    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (bins(rng.normal(e, sig, n_a)), bins(rng.normal(e, sig, n_a))), 1)
    a, c = bins(rng.normal(e, sig, n_b)), bins(rng.normal(e3, sig, n_b))
    swap = rng.random(n_b) < 0.5
    np.add.at(m, (np.where(swap, c, a), np.where(swap, a, c)), 1)
    return m


def test_a_self_coincidence_peak_keeps_its_intensity():
    """Gate on E where E-E and E-E3 cascades are equally common: the E
    peak must be twice the E3 peak, because either gamma of an E-E event
    can be the gate. Counting the diagonal once gave 1.788 here."""
    m = _self_coincidence_matrix()
    s, d, _ = symmetrize(m)
    net, _var = compute_cut_with_variance(s, "x", (296, 304), None, diagonal_twice=d)
    ratio = net[290:311].sum() / net[390:411].sum()
    assert abs(ratio - 2.0) < 0.02, ratio


def test_a_pure_diagonal_cell_gets_variance_four_d():
    """The same events sit in a diagonal cell twice, so its variance is
    4*D, not the 2*D that 'variance = counts' would give."""
    m = np.zeros((50, 50), dtype=np.int64)
    m[20, 20] = 100
    m[20, 30] = 7                                   # off the diagonal
    s, d, _ = symmetrize(m)
    net, var = compute_cut_with_variance(s, "x", (20, 20), None, diagonal_twice=d)
    assert net[20] == 200 and var[20] == 400
    assert net[30] == 7 and var[30] == 7            # ordinary cell: Poisson


def test_without_the_correction_the_diagonal_variance_is_underestimated():
    """Control for the test above: the plain call must NOT produce 4*D,
    or the correction is not what makes the difference."""
    m = np.zeros((50, 50), dtype=np.int64)
    m[20, 20] = 100
    m[20, 30] = 7
    s, _d, _ = symmetrize(m)
    _net, var = compute_cut_with_variance(s, "x", (20, 20), None)
    assert var[20] == 200


def test_the_diagonal_correction_follows_the_background_factor():
    m = np.zeros((60, 60), dtype=np.int64)
    m[40, 40] = 50                                  # diagonal, inside the BG gate
    m[10, 30] = 9
    s, d, _ = symmetrize(m)
    # Gate 10..11 (2 lines), background 40..43 (4 lines): factor = 0.5.
    net, var = compute_cut_with_variance(s, "x", (10, 11), [(40, 43)], diagonal_twice=d)
    plain_net, plain_var = compute_cut_with_variance(s, "x", (10, 11), [(40, 43)])
    assert np.array_equal(net, plain_net), "the correction must not move the counts"
    assert var[40] == plain_var[40] + 0.5 ** 2 * 2 * 50
    others = np.ones(60, dtype=bool)
    others[40] = False
    assert np.array_equal(var[others], plain_var[others])


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_correction_works_gating_either_axis(axis):
    m = np.zeros((30, 30), dtype=np.int64)
    m[12, 12] = 10
    m[12, 20] = 3          # without it the matrix is symmetric and refused
    s, d, _ = symmetrize(m)
    _net, var = compute_cut_with_variance(s, axis, (12, 12), None, diagonal_twice=d)
    assert var[12] == 40


# --- the band-wise symmetry check and progress ------------------------------


def test_the_symmetry_check_finds_an_asymmetry_in_a_late_band():
    """The check stops at the first asymmetric band of 256 rows, so it must
    still look at every band: a matrix symmetric everywhere except one
    cell far down must not be refused as 'already symmetric'."""
    rng = np.random.default_rng(9)
    base = rng.integers(0, 5, size=(700, 700))
    sym = base + base.T
    with pytest.raises(SymmetrizeError, match="already symmetric"):
        check_symmetrizable(sym)
    sym[650, 20] += 1                       # third band, below the diagonal
    check_symmetrizable(sym)                # accepted now


def test_rebin_reports_progress_to_completion():
    m = np.random.default_rng(4).integers(0, 9, size=(200, 200))
    m[190:, :] = 0
    calls = []
    rebin_rows(m, a=0.5, b=0.97, seed=1, progress=lambda d, t: calls.append((d, t)))
    assert calls[0] == (0, 200)
    assert calls[-1] == (200, 200)
    assert all(d1 <= d2 for (d1, _), (d2, _) in zip(calls, calls[1:]))
