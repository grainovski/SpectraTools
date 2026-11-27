"""Sorting CoMPASS list-mode events into singles, a time-difference
spectrum and random-subtracted gamma-gamma matrices.

No GUI here: this is the arithmetic the v7.0.0 Sort List-Mode dialogs
drive. The plan, and the measurements behind each choice, are in
docs/superpowers/plans/2026-11-27-v7-listmode-gg-sorting-plan.md. The
short version:

- **Input is duck-typed.** Events arrive as an iterable of chunks, each
  with `.channel`, `.timestamp` (int64 picoseconds, non-decreasing over
  the whole stream), `.energy` (integer ADC) and `.flags`. That is
  listmode_io.EventChunk, but nothing here imports it, so the sorting can
  be tested on synthetic streams and fed from any reader.

- **All combinations.** Every Y-channel event is paired with every
  X-channel event within the maximum |dt|, not just the nearest. With two
  detectors that is the standard rule; nearest-partner pairing would bias
  the random floor, because a random partner competes with the true one.

- **dt = t(X) - t(Y)**, i.e. t(CH1) - t(CH0) by default, the same sign
  as CoMPASS's own TOF spectrum with CH1 as the stop.

- **Window weights count inclusive dt bins actually summed.** That is
  TV's MrkRange convention (Sum(x2 - x1) + number of marks, vsMark.c:29),
  with the clipping rule matrix_cut takes from HDTV: a window is first
  clipped to the sorted range, and only the bins left are counted.

- **Matrices are sparse while sorting.** A dense 16384 x 16384 float64 or
  int64 array is 2 GiB; the pair lists are a few MB. Cells are flat
  indices row * ncols + col with counts (`SparseMatrix`). Index
  convention is the app's [y][x]: rows are Y (CH0 energy), columns are
  X (CH1 energy), so a pair (x = a, y = b) lands at row b, column a.

- **Subtraction is exact; integer output is stochastically rounded.**
  S = P - w R is kept in float64. For the integer-only .mtx format each
  cell becomes floor(S) + Bernoulli(S - floor(S)) from a fixed seed, so
  E[cell] = S. Plain rounding would send every -1/3 to 0 and
  systematically under-subtract.
"""

import math
from typing import NamedTuple

import numpy as np

from matrix_symmetrize import DEFAULT_SEED

#: Full resolution of the DT5780's 14-bit ADC.
NCHAN = 16384


class SortError(ValueError):
    """A sorting parameter or an input stream that cannot be used, with a
    message meant for the user."""


# ----------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------

def _chunk_arrays(chunk, exclude_flags):
    """(channel, timestamp, energy) of one chunk with excluded events
    removed, as plain numpy arrays.

    Exclusion happens here, before pairing, so an excluded event can
    neither form a pair nor occupy a slot in the carried-over window.

    `exclude_flags` is either an int bit MASK (drop an event when
    ``flags & mask`` is non-zero) or a set/frozenset/list of flag VALUES
    (drop an event whose flags value is one of them). The dialog works in
    values, and the two are not interchangeable: OR-ing the default
    excluded values {0x80, 0x480, 0xC80, 0x8000, 0x8800} gives a mask that
    contains 0x800 and would also drop every good 0x800 event.
    """
    channel = np.asarray(chunk.channel)
    timestamp = np.asarray(chunk.timestamp, dtype=np.int64)
    energy = np.asarray(chunk.energy).astype(np.int64, copy=False)
    if not (len(channel) == len(timestamp) == len(energy)):
        raise SortError("An event chunk has branches of different lengths.")
    if isinstance(exclude_flags, (set, frozenset, list, tuple)):
        if exclude_flags:
            flags = np.asarray(chunk.flags).astype(np.uint64, copy=False)
            values = np.fromiter((int(v) for v in exclude_flags), dtype=np.uint64)
            keep = ~np.isin(flags, values)
            channel, timestamp, energy = channel[keep], timestamp[keep], energy[keep]
    elif exclude_flags:
        flags = np.asarray(chunk.flags).astype(np.uint64, copy=False)
        keep = (flags & np.uint64(exclude_flags)) == 0
        channel, timestamp, energy = channel[keep], timestamp[keep], energy[keep]
    return channel, timestamp, energy


# ----------------------------------------------------------------------
# Singles
# ----------------------------------------------------------------------

class Singles(dict):
    """channel -> int64 histogram of energy, plus what did not fit.

    A dict so callers can index it by channel directly; `overflow` maps
    each channel to the number of its events whose energy fell outside
    0..nchan-1. Those are reported, never clipped into the last bin,
    where they would masquerade as a peak at the top of the range.
    `excluded` is the number of events (all channels) skipped by
    `exclude_flags`.
    """

    def __init__(self, *args, overflow=None, excluded=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.overflow = dict(overflow or {})
        self.excluded = int(excluded)


def sort_singles(chunks, channels=(0, 1), nchan=NCHAN, exclude_flags=0):
    """Singles spectra of `channels` at `nchan` bins (bin = ADC value).

    Events on channels not listed are ignored. Returns a `Singles` dict.
    """
    if nchan < 1:
        raise SortError("The number of channels must be at least 1.")
    channels = tuple(int(c) for c in channels)
    hists = {c: np.zeros(nchan, dtype=np.int64) for c in channels}
    overflow = {c: 0 for c in channels}
    excluded = 0
    for chunk in chunks:
        n_all = len(np.asarray(chunk.channel))
        channel, _t, energy = _chunk_arrays(chunk, exclude_flags)
        excluded += n_all - len(channel)
        for c in channels:
            e = energy[channel == c]
            inside = (e >= 0) & (e < nchan)
            overflow[c] += int(np.count_nonzero(~inside))
            hists[c] += np.bincount(e[inside], minlength=nchan)
    return Singles(hists, overflow=overflow, excluded=excluded)


# ----------------------------------------------------------------------
# Pairing
# ----------------------------------------------------------------------

class Pairs(NamedTuple):
    """All-combination pairs: dt = t(x) - t(y) in ps, and the two
    energies. Element i of each array belongs to the same pair."""
    dt: np.ndarray
    ex: np.ndarray
    ey: np.ndarray


def _pairs_in_buffer(t, ch, e, n_old, max_dt_ps, x_channel, y_channel):
    """Pairs in one buffer of events where at least one member is new.

    The buffer is the carried-over events (the first `n_old`) followed by
    the new chunk. Pairs whose members are both old were counted when
    those events were new, so they are excluded here -- that, and nothing
    about timestamps, is what makes a pair straddling a chunk edge count
    exactly once, ties included.

    Vectorised with searchsorted: for each Y event, the X events within
    [t_y - max, t_y + max] are one contiguous slice of the (time-sorted)
    X list. A new Y event takes the whole slice; an old Y event only the
    part made of new X events, which starts at the first new X index.
    """
    is_x = ch == x_channel
    is_y = ch == y_channel
    idx = np.arange(len(t))
    xi, yi = idx[is_x], idx[is_y]
    tx, ex = t[xi], e[xi]
    ty, ey = t[yi], e[yi]
    first_new_x = int(np.searchsorted(xi, n_old))
    y_new = yi >= n_old

    lo = np.searchsorted(tx, ty - max_dt_ps, side="left")
    hi = np.searchsorted(tx, ty + max_dt_ps, side="right")
    lo = np.where(y_new, lo, np.maximum(lo, first_new_x))
    counts = np.maximum(hi - lo, 0)
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    y_of_pair = np.repeat(np.arange(len(ty)), counts)
    # Position of each pair within its Y event's slice: 0, 1, ..., n-1.
    starts = np.cumsum(counts) - counts
    offset = np.arange(total) - np.repeat(starts, counts)
    x_of_pair = np.repeat(lo, counts) + offset
    dt = tx[x_of_pair] - ty[y_of_pair]
    return dt, ex[x_of_pair], ey[y_of_pair]


def pair_events(chunks, max_dt_ps, x_channel=1, y_channel=0, exclude_flags=0):
    """All-combination pairs with |t(x) - t(y)| <= max_dt_ps.

    Chunks are processed one at a time with the trailing `max_dt_ps` of
    events carried into the next, so a pair is never lost at a boundary
    and memory stays bounded by the chunk size plus that window. The
    stream must be time-sorted; a timestamp that goes backwards is
    refused rather than silently mis-paired (searchsorted needs sorted
    input, and a backwards step would drop or double pairs).
    """
    max_dt_ps = int(max_dt_ps)
    if max_dt_ps < 0:
        raise SortError("The maximum time difference must not be negative.")
    if x_channel == y_channel:
        raise SortError("The X and Y detectors must be different channels.")
    out_dt, out_ex, out_ey = [], [], []
    carry_t = np.zeros(0, dtype=np.int64)
    carry_ch = np.zeros(0, dtype=np.int64)
    carry_e = np.zeros(0, dtype=np.int64)
    last_t = None
    for chunk in chunks:
        ch, t, e = _chunk_arrays(chunk, exclude_flags)
        if len(t) == 0:
            continue
        if np.any(np.diff(t) < 0) or (last_t is not None and t[0] < last_t):
            raise SortError(
                "The events are not in time order. Sort the list-mode data "
                "by timestamp before pairing.")
        last_t = int(t[-1])
        sel = (ch == x_channel) | (ch == y_channel)
        ch, t, e = ch[sel].astype(np.int64), t[sel], e[sel]
        n_old = len(carry_t)
        bt = np.concatenate([carry_t, t])
        bch = np.concatenate([carry_ch, ch])
        be = np.concatenate([carry_e, e])
        dt, ex, ey = _pairs_in_buffer(bt, bch, be, n_old, max_dt_ps,
                                      x_channel, y_channel)
        if len(dt):
            out_dt.append(dt)
            out_ex.append(ex)
            out_ey.append(ey)
        if len(bt):
            # Later events have t >= bt[-1], so only events within
            # max_dt of it can still find a partner.
            keep_from = int(np.searchsorted(bt, bt[-1] - max_dt_ps, side="left"))
            carry_t, carry_ch, carry_e = bt[keep_from:], bch[keep_from:], be[keep_from:]
    if not out_dt:
        empty = np.zeros(0, dtype=np.int64)
        return Pairs(empty, empty.copy(), empty.copy())
    return Pairs(np.concatenate(out_dt), np.concatenate(out_ex),
                 np.concatenate(out_ey))


# ----------------------------------------------------------------------
# Time-difference spectrum and windows
# ----------------------------------------------------------------------

def _check_dt_axis(granularity_ps, max_dt_ps):
    if not granularity_ps or granularity_ps <= 0 or int(granularity_ps) != granularity_ps:
        raise SortError("The time granularity must be a positive whole number of picoseconds.")
    if max_dt_ps < 0:
        raise SortError("The maximum time difference must not be negative.")
    return int(granularity_ps), int(max_dt_ps)


def _n_half_bins(granularity_ps, max_dt_ps):
    """K, so that the dt bins are -K..K (centres k * granularity)."""
    return int(dt_bin_index(np.array([max_dt_ps]), granularity_ps)[0])


def dt_bin_index(dt, granularity_ps):
    """The dt bin each value falls in: bin k is centred on k * granularity.

    Rounded half away from zero (matrix_cut._nint, TV's NINT), so the
    binning is symmetric about dt = 0: +g/2 and -g/2 go to bins +1 and -1.
    With 10 ns timestamps every dt is an exact multiple of the
    granularity and the bin is simply dt / g.
    """
    g = int(granularity_ps)
    dt = np.asarray(dt, dtype=np.int64)
    mag = (2 * np.abs(dt) + g) // (2 * g)
    return np.where(dt < 0, -mag, mag)


def dt_histogram(dt, granularity_ps, max_dt_ps):
    """(bin centres in ps, int64 counts) over -max..+max.

    Bins are `granularity_ps` wide and centred on its multiples, so with
    10 ns timestamps each bin holds exactly one tick value. Pairs beyond
    the range (which pair_events with the same max never produces) are
    left out.
    """
    g, m = _check_dt_axis(granularity_ps, max_dt_ps)
    k = _n_half_bins(g, m)
    centres = np.arange(-k, k + 1, dtype=np.int64) * g
    b = dt_bin_index(dt, g)
    inside = (b >= -k) & (b <= k)
    counts = np.bincount(b[inside] + k, minlength=2 * k + 1).astype(np.int64)
    return centres, counts


def _as_windows(windows):
    """A single (lo, hi) window, or a sequence of them, as a list."""
    if windows is None:
        return []
    windows = list(windows)
    if len(windows) == 2 and all(isinstance(v, (int, float, np.number)) for v in windows):
        return [(windows[0], windows[1])]
    return [(lo, hi) for lo, hi in windows]


def window_bins(window, granularity_ps, max_dt_ps=None):
    """Inclusive bin-index range (ilo, ihi) of the dt bins whose centres
    lie inside the window (lo_ps, hi_ps), clipped to the sorted range
    when `max_dt_ps` is given.

    Returns None if no bin is inside -- either the window is narrower than
    a bin and holds no bin centre, or it lies wholly outside the range.
    Reversed windows are normalised, as matrix_cut does.
    """
    g = int(granularity_ps)
    lo, hi = min(window), max(window)
    ilo = math.ceil(lo / g)
    ihi = math.floor(hi / g)
    if max_dt_ps is not None:
        k = _n_half_bins(g, int(max_dt_ps))
        ilo, ihi = max(ilo, -k), min(ihi, k)
    if ihi < ilo:
        return None
    return ilo, ihi


def _resolved_bins(window, granularity_ps, max_dt_ps, label):
    g = int(granularity_ps)
    if window_bins(window, g) is None:
        raise SortError(
            f"The {label} window {_fmt_window(window)} contains no dt bin; it "
            f"must span at least one multiple of {g / 1000:g} ns.")
    bins = window_bins(window, g, max_dt_ps)
    if bins is None:
        raise SortError(
            f"The {label} window {_fmt_window(window)} lies outside the sorted "
            f"range of +/-{max_dt_ps / 1000:g} ns.")
    return bins


def _fmt_window(window):
    lo, hi = min(window), max(window)
    return f"[{lo / 1000:g}, {hi / 1000:g}] ns"


def window_weight(prompt, randoms, granularity_ps, max_dt_ps):
    """(w, n_prompt_bins, n_random_bins), w = n_prompt_bins / n_random_bins.

    Widths are inclusive counts of the dt bins actually summed, after
    clipping each window to the sorted range -- the same quantity that
    window_mask selects, so counts and widths cannot disagree about which
    bins were involved (matrix_cut's HDTV rule). Several random windows
    are summed, as TV sums several background marks.

    Refused, with a message for the user: a window holding no bin, a
    window wholly outside the range, and any two windows sharing a bin
    (prompt with random, or random with random). An overlap would count
    the shared bins as both signal and background, or as background
    twice.
    """
    g, m = _check_dt_axis(granularity_ps, max_dt_ps)
    prompt_list = _as_windows(prompt)
    if len(prompt_list) != 1:
        raise SortError("Exactly one prompt window is needed.")
    random_list = _as_windows(randoms)
    if not random_list:
        raise SortError("At least one random window is needed.")
    p = _resolved_bins(prompt_list[0], g, m, "prompt")
    rs = [_resolved_bins(r, g, m, "random") for r in random_list]
    for i, r in enumerate(rs):
        if r[0] <= p[1] and p[0] <= r[1]:
            raise SortError(
                f"The random window {_fmt_window(random_list[i])} overlaps the "
                f"prompt window {_fmt_window(prompt_list[0])}.")
        for j in range(i):
            q = rs[j]
            if r[0] <= q[1] and q[0] <= r[1]:
                raise SortError(
                    f"The random windows {_fmt_window(random_list[j])} and "
                    f"{_fmt_window(random_list[i])} overlap.")
    n_prompt = p[1] - p[0] + 1
    n_random = sum(r[1] - r[0] + 1 for r in rs)
    return n_prompt / n_random, n_prompt, n_random


def window_mask(dt, windows, granularity_ps=1):
    """Boolean mask of the dt values whose bin lies in any of `windows`.

    Selection is by dt bin (dt_bin_index), not by raw time, so that what
    is summed is exactly the bins window_weight counted. The default
    granularity of 1 ps makes each integer dt its own bin, which is plain
    lo <= dt <= hi.
    """
    g = int(granularity_ps)
    b = dt_bin_index(dt, g)
    mask = np.zeros(len(b), dtype=bool)
    for w in _as_windows(windows):
        bins = window_bins(w, g)
        if bins is not None:
            mask |= (b >= bins[0]) & (b <= bins[1])
    return mask


# ----------------------------------------------------------------------
# Sparse matrices
# ----------------------------------------------------------------------

class SparseMatrix(NamedTuple):
    """A [y][x] matrix as sorted unique flat indices (row * ncols + col)
    and their values. Cells not listed are zero."""
    shape: tuple
    index: np.ndarray
    values: np.ndarray

    @property
    def rows(self):
        return self.index // self.shape[1]

    @property
    def cols(self):
        return self.index % self.shape[1]

    def total(self):
        return self.values.sum()

    def x_projection(self):
        """Sum over rows: a spectrum indexed by X (column)."""
        return np.bincount(self.cols, weights=self.values, minlength=self.shape[1])

    def y_projection(self):
        """Sum over columns: a spectrum indexed by Y (row)."""
        return np.bincount(self.rows, weights=self.values, minlength=self.shape[0])


def _accumulate(shape, index, values):
    """SparseMatrix with duplicate flat indices summed and zeros dropped."""
    if len(index) == 0:
        return SparseMatrix(tuple(shape), np.zeros(0, dtype=np.int64),
                            np.zeros(0, dtype=np.asarray(values).dtype))
    uniq, inv = np.unique(index, return_inverse=True)
    values = np.asarray(values)
    summed = np.bincount(inv, weights=values.astype(np.float64), minlength=len(uniq))
    if values.dtype.kind in "iu":
        # float64 sums of integers are exact below 2**53 counts per cell.
        summed = np.rint(summed).astype(np.int64)
    keep = summed != 0
    return SparseMatrix(tuple(shape), uniq[keep], summed[keep])


class MatrixSort(NamedTuple):
    """sort_matrix's result: the counts and how many pairs fell off the
    matrix (energy outside the shape, before or after x_map)."""
    matrix: SparseMatrix
    dropped: int


def sort_matrix(ex, ey, mask=None, shape=(NCHAN, NCHAN), x_map=None):
    """Sparse int64 counts of the pairs selected by `mask`, at row ey,
    column ex (the app's [y][x]; X = CH1, Y = CH0 by default).

    `x_map`, if given, maps the selected X energies to integer channels
    -- gain matching CH1 onto CH0 plugs in here, with its dithering. It is
    applied to the selected pairs only. Entries outside the shape are
    dropped and counted rather than clipped onto the edge, where they
    would pile up as a false ridge. Never builds a dense array.
    """
    ex = np.asarray(ex)
    ey = np.asarray(ey)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        ex, ey = ex[mask], ey[mask]
    if x_map is not None:
        ex = np.asarray(x_map(ex))
    if ex.dtype.kind == "f":
        if np.any(ex != np.floor(ex)):
            raise SortError("The X energy map must return whole channel numbers.")
    ex = ex.astype(np.int64)
    ey = ey.astype(np.int64)
    nrows, ncols = int(shape[0]), int(shape[1])
    inside = (ex >= 0) & (ex < ncols) & (ey >= 0) & (ey < nrows)
    dropped = int(np.count_nonzero(~inside))
    flat = ey[inside] * ncols + ex[inside]
    if len(flat):
        uniq, counts = np.unique(flat, return_counts=True)
    else:
        uniq, counts = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return MatrixSort(SparseMatrix((nrows, ncols), uniq.astype(np.int64),
                                   counts.astype(np.int64)), dropped)


def subtract(prompt, random, w):
    """S = P - w * R in float64 over the union of P's and R's cells.

    Exact: no rounding happens here (JSON stores these floats). Cells that
    come out exactly zero are dropped.
    """
    if tuple(prompt.shape) != tuple(random.shape):
        raise SortError("The prompt and random matrices have different shapes.")
    index = np.concatenate([prompt.index, random.index])
    values = np.concatenate([prompt.values.astype(np.float64),
                             -float(w) * random.values.astype(np.float64)])
    return _accumulate(prompt.shape, index, values.astype(np.float64))


def stochastic_round(sparse, seed=DEFAULT_SEED):
    """int64 SparseMatrix: floor(S) + Bernoulli(S - floor(S)) per cell.

    E[cell] = S exactly, so the total and every projection keep their
    expected values, where plain rounding is biased (every -1/3 rounds to
    0 and the matrix is under-subtracted). Seeded, so writing the same
    result twice gives the same file.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(sparse.values, dtype=np.float64)
    fl = np.floor(v)
    up = rng.random(len(v)) < (v - fl)
    out = fl.astype(np.int64) + up.astype(np.int64)
    keep = out != 0
    return SparseMatrix(tuple(sparse.shape), sparse.index[keep], out[keep])


def to_dense(sparse, shape=None, dtype=None):
    """The dense [y][x] array, for consumers that need one (the matrix
    panel, dense JSON). Only call this at a resolution that fits."""
    shape = tuple(sparse.shape if shape is None else shape)
    if tuple(shape) != tuple(sparse.shape):
        raise SortError("to_dense: the requested shape differs from the matrix's.")
    dtype = sparse.values.dtype if dtype is None else dtype
    out = np.zeros(shape[0] * shape[1], dtype=dtype)
    out[sparse.index] = sparse.values
    return out.reshape(shape)


def compress(sparse, factor):
    """Sum factor x factor blocks: cell (r, c) goes to (r // f, c // f).

    The new shape is ceil(n / factor) per axis, so a size that is not a
    multiple of the factor keeps its last, partial block rather than
    losing those counts.
    """
    f = int(factor)
    if f < 1 or f != factor:
        raise SortError("The compression factor must be a positive whole number.")
    nrows, ncols = sparse.shape
    new_shape = (-(-nrows // f), -(-ncols // f))
    flat = (sparse.rows // f) * new_shape[1] + (sparse.cols // f)
    return _accumulate(new_shape, flat, sparse.values)
