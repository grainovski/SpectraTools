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
    on a .5 boundary.

    Kept as TV's convention even though the gate-width weighting no
    longer follows TV (see merge_regions): this is the mark-to-channel
    conversion for .mtx matrices, and .mtx is TV's own format. HDTV's
    mfile backend rounds the other way on an exact half
    (MFMatrix::FindCutBin is `ceil(x - 0.5)`, which sends 2.5 to 2);
    when a ROOT backend arrives, its own conversion belongs with it
    rather than here.
    """
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _axis_size(matrix, axis):
    """Channel count along the axis being gated ON."""
    return matrix.shape[0] if axis == "y" else matrix.shape[1]


def _complementary_size(matrix, axis):
    """Channel count along the axis a cut/projection is indexed BY."""
    return matrix.shape[1] if axis == "y" else matrix.shape[0]


def as_region_list(regions):
    """Normalises `regions` to a list of (lo, hi) tuples.

    Accepts a single bare region -- `(100, 150)` -- as well as a
    sequence of them, so callers that only ever have one gate need not
    wrap it. The two forms are distinguishable because a bare region is
    a length-2 sequence of *numbers*, while a sequence of regions holds
    sequences. None and an empty sequence both mean "no regions".
    """
    if regions is None:
        return []
    regions = list(regions)
    if len(regions) == 2 and all(isinstance(v, (int, float, np.number)) for v in regions):
        return [(regions[0], regions[1])]
    return [(lo, hi) for lo, hi in regions]


def merge_regions(matrix, axis, regions):
    """Marked regions as a sorted list of non-overlapping inclusive
    channel-index ranges, clipped to the matrix, with anything wholly
    outside dropped.

    This is the single place where marks become indices, and it settles
    three things that were previously separate special cases:

    **Clipping.** A region is clipped to [0, size-1] and dropped
    entirely if nothing survives. That covers both a region past the
    last channel and one below channel 0. The latter used to be a real
    bug: clamping left `lo_idx` at 0 with `hi_idx` NEGATIVE, and
    `matrix[0:-3]` is not an empty slice in numpy -- a negative stop
    counts back from the end, so a background region marked entirely
    below channel 0 summed almost the whole matrix and injected
    millions of counts of garbage as "background". Dropping the region
    at conversion time makes that structurally impossible rather than
    something each consumer must remember to check.

    **Merging.** Overlapping regions are merged, so a channel the user
    marked twice is summed once and counted once. Ported from HDTV's
    `VMatrix::AddRegion` (src/mfile-root/VMatrix.cxx:25-61), which
    keeps a sorted boundary list and merges on insert; its sibling
    `PolyBg::AddRegion` does the same for fit backgrounds with the
    comment "If regions overlap, the values covered by two or more
    regions are still only considered once in the fit."

    Not merging biased the background's SHAPE toward the overlap,
    because the doubled channels got double weight in both the sum and
    the width. On a matrix with a linear ramp along the cut axis, a
    nested pair of background marks (0-29 plus 25-29) gave 8821.4
    against a correct 9000 -- 2.0% low -- and an asymmetric overlap
    (0-19 plus 15-29) gave 0.40% low. A *symmetric* overlap cancels
    exactly, which is why any test for this has to use an asymmetric or
    nested pair over a sloped background; a symmetric one passes with
    the bug present.

    TV does not merge either, so this is a divergence from TV as well
    as a fix -- but unlike the weighting question there is no reading in
    which a twice-marked channel should count twice.

    **Reversed regions.** (hi, lo) is normalised to (lo, hi), as HDTV
    does with TMath::Min/Max, so a backwards mark can no longer produce
    a negative width.
    """
    size = _axis_size(matrix, axis)
    clipped = []
    for lo, hi in regions:
        lo_idx = max(0, _nint(min(lo, hi)))
        hi_idx = min(size - 1, _nint(max(lo, hi)))
        if hi_idx < lo_idx:
            continue
        clipped.append((lo_idx, hi_idx))

    clipped.sort()
    merged = []
    for lo_idx, hi_idx in clipped:
        if merged and lo_idx <= merged[-1][1]:
            prev_lo, prev_hi = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi_idx))
        else:
            merged.append((lo_idx, hi_idx))
    return merged


def _sum_and_lines(matrix, axis, merged):
    """(sum over every line in `merged`, number of lines summed).

    The line count is the weighting quantity -- see compute_cut. Both
    come from the same loop over the same ranges, so "counts" and
    "width" cannot disagree about which lines were involved. That
    equivalence is the whole point of the C1 change; keeping them in one
    function is what enforces it.

    Accumulates in float64 regardless of the matrix's dtype, matching
    HDTV's own use of TArrayD. A cut is a weighted difference and is not
    integral in general, so there is nothing to be gained by staying in
    int64 for the no-background case alone.
    """
    total = np.zeros(_complementary_size(matrix, axis), dtype=float)
    lines = 0
    for lo_idx, hi_idx in merged:
        if axis == "y":
            total += matrix[lo_idx : hi_idx + 1, :].sum(axis=0)
        else:
            total += matrix[:, lo_idx : hi_idx + 1].sum(axis=1)
        lines += hi_idx - lo_idx + 1
    return total, lines


def compute_projection(matrix, axis):
    """axis='x' -> matrix.sum(axis=0), a spectrum indexed by X-channel
    (column). axis='y' -> matrix.sum(axis=1), a spectrum indexed by
    Y-channel (row)."""
    _check_axis(axis)
    if axis == "x":
        return matrix.sum(axis=0)
    else:
        return matrix.sum(axis=1)


def _cut_with_variance(matrix, axis, cut_regions, bg_regions):
    """Shared worker for compute_cut/compute_cut_with_variance."""
    _check_axis(axis)
    pos, n_cut = _sum_and_lines(matrix, axis, merge_regions(matrix, axis, as_region_list(cut_regions)))
    bg, n_bg = _sum_and_lines(matrix, axis, merge_regions(matrix, axis, as_region_list(bg_regions)))

    if n_bg == 0:
        # No background at all, or every background region fell outside
        # the matrix -- either way there is nothing to subtract, which is
        # a normal result and not an error.
        return pos, np.maximum(pos, 0.0)

    factor = n_cut / n_bg
    net = pos - factor * bg
    # Poisson variance of a sum of counts is the sum itself, so the two
    # sums we already hold ARE the two variances; no second pass over the
    # matrix is needed. Clamped at zero per channel because a matrix that
    # already holds subtracted data (a Subtract-Spectra result, or
    # random-coincidence subtraction) can carry negative counts, for which
    # the Poisson assumption -- and a variance read off the counts -- does
    # not hold. Zero is the honest floor there rather than a negative
    # variance or a NaN from its square root.
    variance = np.maximum(pos, 0.0) + factor ** 2 * np.maximum(bg, 0.0)
    return net, variance


def compute_cut(matrix, axis, cut_region, bg_regions):
    """`axis` is which projection was marked -- the axis being gated
    ON, in that projection's own channel units. Returns a 1D array
    along the COMPLEMENTARY axis (axis='x' input -> Y-indexed output,
    and vice versa).

        net[ch] = pos[ch] - (n_cut / n_bg) * bg[ch]

    where pos and bg are sums over the marked row-range (axis='y') or
    column-range (axis='x'), and n_cut / n_bg are the numbers of lines
    ACTUALLY SUMMED after clipping and merging.

    `cut_region` may be a single (lo, hi) region or several -- several
    gates are summed into one cut spectrum, with n_cut accumulating
    across them, which is how a cascade is gated on more than one of its
    members at once. Zero background regions means no subtraction, not
    an error. Negative results are never clamped.

    **The weighting deliberately diverges from TV.** TV takes the factor
    from `MrkRange(gate) / MrkRange(bgGate)` (VsCut.c:478), and MrkRange
    (vsMark.c:29-37) works purely on the marks, knowing nothing about
    the matrix -- so TV divides counts drawn from the clipped range by a
    width drawn from the unclipped marks. The two quantities come from
    different domains and the mismatch shows up whenever a mark
    overhangs an edge.

    HDTV's rule is used instead: both come from the lines actually
    summed, making the factor a genuine mean-background-per-line. HDTV
    implements it twice independently -- `VMatrix::Cut`
    (src/mfile-root/VMatrix.cxx:105-114, `bgFac = nCut / nBg` counted
    per line added) and `RHisto2D.ExecuteCut` (hdtv/histogram.py:505,
    `bgFactor = -numFgBins/numBgBins` counted from FindBin results).

    Measured on a flat matrix where every channel holds the same counts,
    so the correct net is exactly zero by construction: a background
    region overhanging the low edge left 250 of 1000 counts per channel
    unsubtracted under TV's rule -- 25% -- where HDTV's gives 0.

    Note the direction. This under-subtracts, the opposite of the
    over-subtraction reported against v3.1.2 and fixed in v3.1.3, so it
    is a separate fault rather than a regression from that fix. It also
    subsumes v3.1.4's special case: a background region wholly outside
    the matrix contributes no lines and therefore no width, so it is
    ignored as a consequence of the uniform rule rather than by an
    explicit skip.
    """
    net, _ = _cut_with_variance(matrix, axis, cut_region, bg_regions)
    return net


def compute_cut_with_variance(matrix, axis, cut_region, bg_regions):
    """compute_cut's result plus its per-channel variance, as
    (net, variance).

    A background-subtracted cut is not Poisson-distributed: its
    variance is `pos + factor**2 * bg`, which exceeds the net counts it
    accompanies, and the net can legitimately go negative where no
    Poisson error exists at all. Fitting such a spectrum with sqrt(N)
    weights therefore misweights every channel and reports optimistic
    parameter errors and chi-square.

    HDTV avoids this by taking every projection with ROOT's "e" option
    and folding the background in with `TH1::Add(tmp, bgFactor)`, which
    propagates variance as `e1**2 + f**2 * e2**2` -- the same
    expression. Its cut histograms therefore arrive with usable errors
    and everything downstream is weighted by them.

    Provided as a separate entry point so compute_cut's return type
    stays a bare array for the callers that do not need this.
    """
    return _cut_with_variance(matrix, axis, cut_region, bg_regions)
