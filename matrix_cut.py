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


def _region_width(matrix, axis, region):
    """Channel count covered by `region` after clamping to the
    matrix's extent. Floored at 0 -- a region entirely past an edge
    (lo_idx clamped up to 0 while hi_idx is still below it, or hi_idx
    clamped down to size-1 while lo_idx is already above it) would
    otherwise yield a negative width, which would silently corrupt
    compute_cut's gate-width ratio (or, in a coincidental case, divide
    by exactly zero) instead of the region contributing nothing,
    matching TV's own SpcProject clamping behavior on the sum side
    (tv-1.9.13/lib/tv/vsSpectra.c:971-980)."""
    lo_idx, hi_idx = _region_bounds(matrix, axis, region)
    return max(0, hi_idx - lo_idx + 1)


def _region_sum(matrix, axis, region):
    """Raw (unweighted) sum over `region`'s channel range on `axis`,
    returned as a 1D array indexed by the COMPLEMENTARY axis."""
    lo_idx, hi_idx = _region_bounds(matrix, axis, region)
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

    pos_width = _region_width(matrix, axis, cut_region)
    bg = np.zeros_like(pos)
    bg_width = 0
    for region in bg_regions:
        bg = bg + _region_sum(matrix, axis, region)
        bg_width += _region_width(matrix, axis, region)

    if bg_width == 0:
        # Every bg region is individually out of the matrix's bounds
        # (each one's floored width is 0) -- the same lo_idx > hi_idx
        # condition that zeroes a region's width also zeroes its sum,
        # so `bg` is guaranteed all-zero here too. Falling through to
        # the division would be a ZeroDivisionError for no benefit:
        # returning pos unchanged is the exact (not approximate)
        # limiting value of the gate-width ratio, and matches the
        # zero-bg_regions case above -- no valid background specified,
        # so no subtraction.
        return pos

    return pos - (pos_width / bg_width) * bg
