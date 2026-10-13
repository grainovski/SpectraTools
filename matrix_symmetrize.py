"""Symmetrizing a gamma-gamma matrix, by the user's choice.

    S = M' + M'.T        where M' is M with its Y axis mapped onto X's scale

Matrices are the app's [y][x] (rows are Y). The design, and the
measurements behind each choice, are in
docs/superpowers/specs/2026-10-10-matrix-symmetrization-design.md. The
short version:

- **Gain-match first.** A two-detector matrix has one detector per axis
  with different gains; folding it as it stands puts the same line in two
  places. On the CoMPASS Co-60 run a naive M + M.T widened the peaks by
  about 75%, while matching Y to X first kept them at the detectors' own
  width. The map is linear, x = a + b*y, taken from the same two lines on
  both projections; a = 0, b = 1 when the axes already share a scale.

- **The diagonal stays doubled.** M + M.T counts every (gate gamma,
  partner) pair once, which is what a gate measures. Counting the
  diagonal once instead cost a simulated self-coincidence peak 10.6% of
  its intensity and flattened its top, because with finite resolution
  only about a fifth of such a peak lies exactly on the diagonal. The
  doubled diagonal's one cost -- its cells hold the same events twice,
  so their variance is 4*D rather than 2*D -- is corrected in the cut
  (matrix_cut, `diagonal_twice`), which is why symmetrize() returns D.

- **Integer counts.** Each count of a Y bin is sent to one of the new bins
  it overlaps, at random with the overlap fractions as probabilities, from
  a fixed seed. That keeps the matrix integer and its statistics exactly
  Poisson, where splitting by fractions would make nearly every cell
  fractional; the fixed seed makes the result reproducible.
"""

import math

import numpy as np

#: Fixed so that symmetrizing the same matrix twice gives the same result.
DEFAULT_SEED = 20261010

#: Overlaps smaller than this are treated as no overlap, so a bin edge
#: landing exactly on a boundary does not create a zero-width target.
_EPS = 1e-12


class SymmetrizeError(ValueError):
    """A matrix or a line pair that cannot be used, with a message meant
    for the user."""


def check_symmetrizable(matrix):
    """Raise SymmetrizeError, saying why, unless `matrix` can be folded.

    Refused: non-square (the axes cannot be mirrored), any negative cell
    (background-subtracted or difference data, where neither the fold nor
    the Poisson statistics mean anything), and a matrix that is already
    symmetric (folding it again would double every count).
    """
    m = np.asarray(matrix)
    if m.ndim != 2 or m.size == 0:
        raise SymmetrizeError("The matrix is empty, so there is nothing to symmetrize.")
    if m.shape[0] != m.shape[1]:
        raise SymmetrizeError(
            f"Only a square matrix can be symmetrized; this one is "
            f"{m.shape[1]} x {m.shape[0]} (X x Y).")
    if np.any(m < 0):
        raise SymmetrizeError(
            "The matrix holds negative counts, so it is background-subtracted "
            "or a difference of matrices. Symmetrizing that is not meaningful.")
    if _is_symmetric(m):
        raise SymmetrizeError(
            "The matrix is already symmetric. Symmetrizing it again would "
            "double every count.")


def _is_symmetric(m, block=256):
    """m == m.T, compared a band of rows at a time so that an asymmetric
    matrix -- the usual case -- is settled by its first band. A whole
    8192x8192 comparison took 1.8 s; the first band takes 0.03 s."""
    n = m.shape[0]
    for i in range(0, n, block):
        if not np.array_equal(m[i:i + block, :], m[:, i:i + block].T):
            return False
    return True


def refine_centroid(projection, guess, half_width=6, iterations=6):
    """The background-subtracted centroid of the peak nearest `guess`.

    A straight-line background through the two outermost bins at each end
    of a window of +-`half_width` bins is subtracted, and the centroid of
    what remains is taken; the window is re-centred and the step repeated
    until it settles. Accurate to about a tenth of a bin for an isolated
    peak, which is far finer than the gain match needs.
    """
    p = np.asarray(projection, dtype=float)
    n = p.size
    c = float(guess)
    for _ in range(iterations):
        centre = int(round(c))
        lo, hi = max(0, centre - half_width), min(n - 1, centre + half_width)
        if hi - lo < 4:
            raise SymmetrizeError(f"Channel {guess:g} is too close to the edge to locate a peak.")
        idx = np.arange(lo, hi + 1, dtype=float)
        w = p[lo:hi + 1]
        left, right = w[:2].mean(), w[-2:].mean()
        background = left + (right - left) * (idx - idx[0]) / (idx[-1] - idx[0])
        net = np.clip(w - background, 0.0, None)
        if net.sum() <= 0:
            raise SymmetrizeError(f"No peak found near channel {guess:g}.")
        new = float((idx * net).sum() / net.sum())
        if abs(new - c) < 0.01:
            return new
        c = new
    return c


def map_from_lines(x_lines, y_lines):
    """(a, b) of the map x = a + b*y that carries both Y lines onto their
    X counterparts. Raises SymmetrizeError for a pair that cannot define
    a usable map."""
    (x1, x2), (y1, y2) = x_lines, y_lines
    if abs(y2 - y1) < 2:
        raise SymmetrizeError(
            "The two lines must be at least 2 channels apart on Y to define "
            "the scale between the axes.")
    b = (x2 - x1) / (y2 - y1)
    if not math.isfinite(b) or b <= 0:
        raise SymmetrizeError(
            "The lines are in a different order on X and on Y. Enter the "
            "same two lines on both axes.")
    return x1 - b * y1, b


def rebin_rows(matrix, a, b, seed=DEFAULT_SEED, progress=None):
    """Map every row (Y bin) onto X's scale with x = a + b*y.

    Returns (rebinned int64 matrix of the same shape, counts dropped off
    the edges). Old bin r spans [r - 1/2, r + 1/2]; its image overlaps one
    or more new bins, and each count goes to one of them at random with
    probability equal to the overlap fraction (sequential binomial
    splitting, vectorised along the row).

    `progress(done, total)`, if given, is called every 64 rows and at the
    end -- a full 8192-row rebin takes several seconds.
    """
    if not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
        raise SymmetrizeError("The scale between the axes must be a positive, finite number.")
    m = np.asarray(matrix)
    if np.any(m < 0):
        raise SymmetrizeError("Negative counts cannot be rebinned.")
    n_rows, n_cols = m.shape
    rng = np.random.default_rng(seed)
    out = np.zeros((n_rows, n_cols), dtype=np.int64)
    lost = 0
    for r in range(n_rows):
        if progress is not None and r % 64 == 0:
            progress(r, n_rows)
        row = m[r].astype(np.int64)
        if not row.any():
            continue
        lo, hi = a + b * (r - 0.5), a + b * (r + 0.5)
        targets = []
        for t in range(int(math.floor(lo + 0.5)), int(math.floor(hi + 0.5)) + 1):
            overlap = min(hi, t + 0.5) - max(lo, t - 0.5)
            if overlap > _EPS and 0 <= t < n_rows:
                targets.append((t, overlap / (hi - lo)))
        inside = sum(p for _t, p in targets)
        remaining = row.copy()
        remaining_p = 1.0
        for k, (t, p) in enumerate(targets):
            last = k == len(targets) - 1
            if last and 1.0 - inside < 1e-9:
                # Everything left belongs here; a probability that rounds
                # to 0.99999999 must not strand a stray count.
                take = remaining
            else:
                take = rng.binomial(remaining, min(1.0, p / remaining_p))
            out[t] += take
            remaining = remaining - take
            remaining_p -= p
        lost += int(remaining.sum())
    if progress is not None:
        progress(n_rows, n_rows)
    return out, lost


def symmetrize(matrix, a=0.0, b=1.0, seed=DEFAULT_SEED, progress=None):
    """(S, D, lost): S = M' + M'.T, D = diag(M'), counts dropped by the
    rebin. M' is `matrix` with its Y axis mapped onto X by x = a + b*y;
    with the default identity map it is `matrix` itself.

    D is what matrix_cut's `diagonal_twice` needs to give the doubled
    diagonal its true variance.
    """
    check_symmetrizable(matrix)
    if a == 0.0 and b == 1.0:
        mg, lost = np.asarray(matrix).astype(np.int64, copy=True), 0
    else:
        mg, lost = rebin_rows(matrix, a, b, seed, progress=progress)
    diagonal = np.diag(mg).copy()
    return mg + mg.T, diagonal, lost
