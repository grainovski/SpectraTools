import numpy as np


def _check_axis(axis):
    """Shared validation for both public entry points -- anything
    other than 'x'/'y' (a typo, None, 'Y', 'row', ...) must fail loudly
    rather than being silently treated as 'x' by the ternaries in the
    private helpers below."""
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def _nint(x):
    """Round-half-away-from-zero, matching TV's own NINT macro
    (lib/tv/vsTypes.h) -- Python's built-in round() uses
    round-half-to-even instead, which would disagree with TV exactly
    on a .5 boundary."""
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _region_bounds(matrix, axis, region):
    """Returns (lo, hi) integer indices (inclusive), clamped to the
    matrix's extent along `axis`."""
    lo, hi = region
    size = matrix.shape[0] if axis == "y" else matrix.shape[1]
    lo_idx = max(0, _nint(lo))
    hi_idx = min(size - 1, _nint(hi))
    return lo_idx, hi_idx


def _marked_width(region):
    """Channel count a region MARKS, exactly as TV's MrkRange counts it
    (tv-1.9.13/lib/tv/vsMark.c:29-37): `NINT(x2) - NINT(x1)`, plus one
    per region for inclusive channel counting.

    Deliberately NOT clamped to the matrix. TV clamps the SUM
    (SpcProject, vsSpectra.c:962-981) but computes the gate-width factor
    from the marks themselves (VsCut.c:478 calls MrkRange on
    VcutsGate/VcutsBgGate, which hold the raw marks), and the two are
    genuinely different quantities.

    This function used to clamp, on the stated grounds that it "matches
    TV's own SpcProject clamping behavior on the sum side" -- which
    conflated the two. The consequence was real over-subtraction: a
    background region dragged past the edge of the matrix had its width
    clipped, which shrank the denominator of pos_width/bg_width, inflated
    the factor, and subtracted far too much background. Measured against
    a literal transcription of TV's algorithm, a single background region
    overhanging the low edge produced a net spectrum ~7% of TV's; fully
    in-bounds regions were already bit-identical.
    """
    lo, hi = region
    return _nint(hi) - _nint(lo) + 1


def _region_sum(matrix, axis, region):
    """Raw (unweighted) sum over `region`'s channel range on `axis`,
    returned as a 1D array indexed by the COMPLEMENTARY axis.

    A region that lies entirely outside the matrix contributes nothing,
    matching TV's SpcProject, whose `for (i = l; i < r; i++)` loop simply
    never executes when clamping leaves r <= l
    (tv-1.9.13/lib/tv/vsSpectra.c:962-981).

    The explicit check is load-bearing for a region below channel 0.
    Clamping leaves lo_idx at 0 but hi_idx NEGATIVE, and `matrix[0:-3]`
    is not an empty slice in numpy -- a negative stop counts back from
    the end, so it summed almost the whole matrix and injected millions
    of counts of pure garbage as "background", which was then subtracted.
    Marks past the HIGH edge never had this problem: lo_idx > hi_idx
    there produces a genuinely empty slice.
    """
    lo_idx, hi_idx = _region_bounds(matrix, axis, region)
    if hi_idx < lo_idx:
        complementary = matrix.shape[1] if axis == "y" else matrix.shape[0]
        return np.zeros(complementary, dtype=matrix.dtype)
    if axis == "y":
        return matrix[lo_idx : hi_idx + 1, :].sum(axis=0)
    else:
        return matrix[:, lo_idx : hi_idx + 1].sum(axis=1)


def compute_projection(matrix, axis):
    """axis='x' -> matrix.sum(axis=0), a spectrum indexed by X-channel
    (column). axis='y' -> matrix.sum(axis=1), a spectrum indexed by
    Y-channel (row)."""
    _check_axis(axis)
    if axis == "x":
        return matrix.sum(axis=0)
    else:
        return matrix.sum(axis=1)


def compute_cut(matrix, axis, cut_region, bg_regions):
    """`axis` is which projection was marked -- the axis being gated
    ON, in that projection's own channel units. Implements TV's
    gate-width-weighted background subtraction:
        net[ch] = pos[ch] - (pos_width/bg_width) * bg[ch]
    where pos/bg are raw sums over the marked row-range (axis='y') or
    column-range (axis='x') of `matrix`. Returns a 1D array along the
    COMPLEMENTARY axis (axis='x' input -> Y-indexed output, and vice
    versa). Zero bg_regions => net = pos (no subtraction, not an
    error). Never clamps negative results."""
    _check_axis(axis)
    pos = _region_sum(matrix, axis, cut_region)

    if not bg_regions:
        return pos

    # Widths come from the MARKS, unclamped; the sums above are clamped
    # to the matrix. TV separates these the same way -- see
    # _marked_width. Clamping the widths too is what caused
    # over-subtraction for any region overhanging an edge.
    pos_width = _marked_width(cut_region)
    bg = np.zeros_like(pos)
    bg_width = 0
    for region in bg_regions:
        bg = bg + _region_sum(matrix, axis, region)
        bg_width += _marked_width(region)

    if bg_width <= 0:
        # Unreachable for normally-marked regions: with inclusive
        # counting even a single-channel mark has width 1, and callers
        # hand over (min, max) pairs. Kept as a guard because the
        # alternative is a ZeroDivisionError (or a sign flip, for a
        # negative total) on a caller that passed a reversed region --
        # returning pos unchanged matches the no-background case above.
        return pos

    return pos - (pos_width / bg_width) * bg
