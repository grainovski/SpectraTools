import numpy as np


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
    lo_idx, hi_idx = _region_bounds(matrix, axis, region)
    return hi_idx - lo_idx + 1


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
    if axis == "x":
        return matrix.sum(axis=0)
    elif axis == "y":
        return matrix.sum(axis=1)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


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
    pos = _region_sum(matrix, axis, cut_region)

    if not bg_regions:
        return pos

    pos_width = _region_width(matrix, axis, cut_region)
    bg = np.zeros_like(pos)
    bg_width = 0
    for region in bg_regions:
        bg = bg + _region_sum(matrix, axis, region)
        bg_width += _region_width(matrix, axis, region)

    return pos - (pos_width / bg_width) * bg
