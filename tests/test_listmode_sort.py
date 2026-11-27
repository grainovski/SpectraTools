"""Sorting list-mode events into singles, dt spectra and random-subtracted
gamma-gamma matrices (listmode_sort).

The synthetic generator below builds a time-sorted two-channel stream
with a known number of true coincidences and a known flat random rate, so
the weight, the recovered net and the random-only control all have exact
or statistically sharp expectations. Real-data cross-checks run on the
Na-22 run when its untracked folder is present and skip by name
otherwise (see the plan's Fixtures section).
"""

import json
import math
import os
from typing import NamedTuple

import numpy as np
import pytest

from listmode_sort import (
    SortError,
    SparseMatrix,
    compress,
    dt_bin_index,
    dt_histogram,
    pair_events,
    sort_matrix,
    sort_singles,
    stochastic_round,
    subtract,
    to_dense,
    window_mask,
    window_weight,
)
from matrix_symmetrize import DEFAULT_SEED

NS = 1000  # ps
G = 10 * NS  # the DT5780 run's timestamp granularity


class Chunk(NamedTuple):
    channel: np.ndarray
    timestamp: np.ndarray
    energy: np.ndarray
    flags: np.ndarray


def make_chunks(channel, timestamp, energy, flags=None, chunk_size=None):
    channel = np.asarray(channel, dtype=np.uint16)
    timestamp = np.asarray(timestamp, dtype=np.int64)
    energy = np.asarray(energy, dtype=np.int64)
    flags = np.zeros(len(channel), dtype=np.uint32) if flags is None else np.asarray(flags, dtype=np.uint32)
    n = len(channel)
    size = n if chunk_size is None else chunk_size
    return [Chunk(channel[i:i + size], timestamp[i:i + size], energy[i:i + size], flags[i:i + size])
            for i in range(0, max(n, 1), max(size, 1))]


def synthetic_stream(seed, n_true, rate_hz, duration_s, line_y=1000, line_x=1500):
    """(channel, timestamp, energy) arrays, time-sorted, of:

    - n_true true coincidences: a CH0 event (energy line_y) and a CH1
      event (energy line_x) at dt = t(CH1) - t(CH0) in {-G, 0, +G} with
      probabilities 1/4, 1/2, 1/4 -- the prompt peak spans the -10/0/+10
      ns ticks, like the real run;
    - uncorrelated singles in each channel at rate_hz, uniform in time,
      with energies uniform over 100..4000 (the randoms).

    Timestamps are multiples of G.
    """
    rng = np.random.default_rng(seed)
    n_ticks = int(duration_s * 1e12) // G
    n_single = rng.poisson(rate_hz * duration_s, size=2)
    t_true = rng.integers(2, n_ticks - 2, size=n_true)
    jitter = rng.choice([-1, 0, 1], size=n_true, p=[0.25, 0.5, 0.25])
    ch = np.concatenate([
        np.zeros(n_true), np.ones(n_true),
        np.zeros(n_single[0]), np.ones(n_single[1])]).astype(np.uint16)
    ticks = np.concatenate([
        t_true, t_true + jitter,
        rng.integers(0, n_ticks, size=n_single[0]),
        rng.integers(0, n_ticks, size=n_single[1])])
    energy = np.concatenate([
        np.full(n_true, line_y), np.full(n_true, line_x),
        rng.integers(100, 4000, size=n_single.sum())])
    order = np.argsort(ticks, kind="stable")
    return ch[order], ticks[order].astype(np.int64) * G, energy[order].astype(np.int64)


def brute_force_pairs(channel, timestamp, energy, max_dt, x_channel=1, y_channel=0):
    """Reference all-combination pairing, O(n_x * n_y)."""
    xs = np.flatnonzero(channel == x_channel)
    ys = np.flatnonzero(channel == y_channel)
    dt = timestamp[xs][None, :] - timestamp[ys][:, None]
    yi, xi = np.nonzero(np.abs(dt) <= max_dt)
    return dt[yi, xi], energy[xs][xi], energy[ys][yi]


def canonical(dt, ex, ey):
    order = np.lexsort((ey, ex, dt))
    return np.stack([dt[order], ex[order], ey[order]])


PROMPT = (-10 * NS, 10 * NS)
RANDOMS = [(-1500 * NS, -500 * NS), (500 * NS, 1500 * NS)]
MAX_DT = 2000 * NS


def net_and_sigma(pairs, w):
    """Net prompt total and its standard deviation sqrt(P + w^2 R)."""
    p = sort_matrix(pairs.ex, pairs.ey, window_mask(pairs.dt, PROMPT, G)).matrix
    r = sort_matrix(pairs.ex, pairs.ey, window_mask(pairs.dt, RANDOMS, G)).matrix
    s = subtract(p, r, w)
    np_, nr = p.total(), r.total()
    return s.total(), math.sqrt(np_ + w * w * nr), p, r, s


# ----------------------------------------------------------------------
# Singles
# ----------------------------------------------------------------------

def test_sort_singles_counts_overflow_and_excluded_flags():
    chunks = make_chunks(
        channel=[0, 0, 1, 1, 1, 2],
        timestamp=[0, 1, 2, 3, 4, 5],
        energy=[5, 16384, 7, 7, 3, 9],
        flags=[0, 0, 0, 0, 0x8, 0],
        chunk_size=4)
    s = sort_singles(chunks, exclude_flags=0x8)
    assert s[0][5] == 1 and s[0].sum() == 1
    assert s.overflow == {0: 1, 1: 0}
    assert s[1][7] == 2 and s[1].sum() == 2  # the flagged energy-3 event is skipped
    assert s.excluded == 1
    assert 2 not in s


# ----------------------------------------------------------------------
# Pairing
# ----------------------------------------------------------------------

def test_all_combinations_three_x_events_in_one_window():
    chunks = make_chunks(channel=[1, 0, 1, 1], timestamp=[95, 100, 104, 108],
                         energy=[11, 50, 12, 13])
    p = pair_events(chunks, max_dt_ps=10)
    assert len(p.dt) == 3
    assert sorted(p.dt.tolist()) == [-5, 4, 8]
    assert sorted(p.ex.tolist()) == [11, 12, 13]
    assert p.ey.tolist() == [50, 50, 50]


def test_dt_sign_is_t_x_minus_t_y():
    p = pair_events(make_chunks([0, 1], [100, 130], [1, 2]), max_dt_ps=50)
    assert p.dt.tolist() == [30]
    p = pair_events(make_chunks([0, 1], [100, 130], [1, 2]), max_dt_ps=50,
                    x_channel=0, y_channel=1)
    assert p.dt.tolist() == [-30]


def test_pair_straddling_chunk_edge_counted_once():
    chunks = make_chunks(channel=[0, 1], timestamp=[100, 105], energy=[1, 2], chunk_size=1)
    p = pair_events(chunks, max_dt_ps=10)
    assert p.dt.tolist() == [5]


def test_equal_timestamps_across_chunk_edge_counted_once():
    # y(100) | x(100), y(100): x pairs with both y's, each exactly once.
    chunks = make_chunks(channel=[0, 1, 0], timestamp=[100, 100, 100], energy=[1, 2, 3],
                         chunk_size=1)
    p = pair_events(chunks, max_dt_ps=0)
    assert sorted(p.ey.tolist()) == [1, 3]


def test_chunk_size_does_not_change_pairs_and_matches_brute_force():
    ch, t, e = synthetic_stream(seed=7, n_true=40, rate_hz=2e5, duration_s=2e-3)
    assert len(ch) > 600
    max_dt = 50 * NS
    ref = canonical(*brute_force_pairs(ch, t, e, max_dt))
    assert ref.shape[1] > 40
    for size in (1, 7, None):
        p = pair_events(make_chunks(ch, t, e, chunk_size=size), max_dt_ps=max_dt)
        np.testing.assert_array_equal(canonical(p.dt, p.ex, p.ey), ref)


def test_unsorted_input_refused():
    with pytest.raises(SortError, match="time order"):
        pair_events(make_chunks([0, 1], [100, 50], [1, 2]), max_dt_ps=10)
    with pytest.raises(SortError, match="time order"):
        pair_events(make_chunks([0, 1], [100, 50], [1, 2], chunk_size=1), max_dt_ps=10)


def test_excluded_flags_never_pair():
    chunks = make_chunks([0, 1, 1], [100, 101, 102], [1, 2, 3], flags=[0, 4, 0])
    p = pair_events(chunks, max_dt_ps=10, exclude_flags=4)
    assert p.ex.tolist() == [3]


def test_pairing_one_million_events_is_fast():
    import time
    ch, t, e = synthetic_stream(seed=3, n_true=20000, rate_hz=5e5, duration_s=1.0)
    assert len(ch) > 1_000_000
    start = time.perf_counter()
    p = pair_events(make_chunks(ch, t, e, chunk_size=200_000), max_dt_ps=MAX_DT)
    elapsed = time.perf_counter() - start
    assert len(p.dt) > 20000
    assert elapsed < 20.0, f"pairing 1e6 events took {elapsed:.1f} s"


# ----------------------------------------------------------------------
# dt spectrum and windows
# ----------------------------------------------------------------------

def test_dt_histogram_bins_centred_on_multiples_of_granularity():
    dt = np.array([-20, -10, 0, 0, 10, 4, -4, 5, -5]) * NS
    centres, counts = dt_histogram(dt, G, 20 * NS)
    assert centres.tolist() == [-20 * NS, -10 * NS, 0, 10 * NS, 20 * NS]
    # +/-5 ns (half a bin) rounds away from zero, symmetric about 0.
    assert counts.tolist() == [1, 2, 4, 2, 0]
    assert dt_bin_index(np.array([5 * NS, -5 * NS]), G).tolist() == [1, -1]


def test_window_weight_inclusive_bin_count():
    w, n_p, n_r = window_weight(PROMPT, RANDOMS, G, MAX_DT)
    assert (n_p, n_r) == (3, 202)  # [-10, 10] ns at 10 ns = 3 bins; 101 + 101
    assert w == 3 / 202


def test_window_weight_single_random_window_and_bin_centre_rule():
    # [-15, 15] ns holds the centres -10, 0, 10: still 3 bins.
    w, n_p, n_r = window_weight((-15 * NS, 15 * NS), (100 * NS, 199 * NS), G, MAX_DT)
    assert (n_p, n_r) == (3, 10)


def test_window_weight_clips_to_sorted_range():
    # Random window overhangs +2 us: only the bins 1000..2000 ns count.
    w, n_p, n_r = window_weight(PROMPT, (1000 * NS, 5000 * NS), G, MAX_DT)
    assert n_r == 101
    assert w == 3 / 101


@pytest.mark.parametrize("prompt, randoms, message", [
    (PROMPT, (0, 50 * NS), "overlaps the prompt"),
    (PROMPT, [(100 * NS, 300 * NS), (300 * NS, 500 * NS)], "overlap"),
    ((2 * NS, 8 * NS), RANDOMS, "contains no dt bin"),
    (PROMPT, (3000 * NS, 4000 * NS), "outside the sorted range"),
    (PROMPT, [], "At least one random"),
])
def test_window_weight_refusals(prompt, randoms, message):
    with pytest.raises(SortError, match=message):
        window_weight(prompt, randoms, G, MAX_DT)


def test_window_mask_selects_the_bins_window_weight_counts():
    dt = np.arange(-30, 31, 1) * NS
    mask = window_mask(dt, PROMPT, G)
    # dt in [-15 ns, 15 ns) round into bins -1..1; +15 ns rounds up to bin 2.
    assert dt[mask].min() == -14 * NS and dt[mask].max() == 14 * NS
    assert window_mask(np.array([-10, 0, 10, 11]), (-10, 10)).tolist() == [True, True, True, False]


# ----------------------------------------------------------------------
# Matrices
# ----------------------------------------------------------------------

def test_orientation_x_is_column_y_is_row():
    m = sort_matrix(np.array([7]), np.array([3]), shape=(10, 12)).matrix
    d = to_dense(m)
    assert d.shape == (10, 12)
    assert d[3, 7] == 1 and d.sum() == 1


def test_sort_matrix_x_map_and_dropped():
    ex = np.array([1, 2, 3, 9])
    ey = np.array([0, 0, 0, 0])
    res = sort_matrix(ex, ey, shape=(4, 8), x_map=lambda e: 2 * e)
    d = to_dense(res.matrix)
    assert d[0, 2] == 1 and d[0, 4] == 1 and d[0, 6] == 1
    assert res.dropped == 1  # 2 * 9 = 18 is off the 8-column edge
    # A mask selects before the map; an out-of-range Y is dropped too.
    res = sort_matrix(ex, np.array([0, 5, 0, 0]), mask=np.array([True, True, False, False]),
                      shape=(4, 8), x_map=lambda e: 2 * e)
    assert res.matrix.total() == 1 and res.dropped == 1


def test_subtract_union_and_stochastic_round_and_compress():
    shape = (4, 4)
    p = SparseMatrix(shape, np.array([0, 5]), np.array([3, 1]))
    r = SparseMatrix(shape, np.array([5, 15]), np.array([2, 4]))
    s = subtract(p, r, 0.5)
    np.testing.assert_array_equal(s.index, [0, 15])  # cell 5: 1 - 0.5*2 = 0, dropped
    np.testing.assert_allclose(s.values, [3.0, -2.0])
    c = compress(SparseMatrix(shape, np.array([0, 1, 4, 5, 15]), np.array([1, 2, 3, 4, 5])), 2)
    np.testing.assert_array_equal(to_dense(c), [[10, 0], [0, 5]])
    c3 = compress(SparseMatrix(shape, np.array([15]), np.array([1])), 3)
    assert c3.shape == (2, 2) and to_dense(c3)[1, 1] == 1


def test_stochastic_round_preserves_mean_where_plain_rounding_does_not():
    n = 1_000_000
    shape = (1000, 1000)
    s = SparseMatrix(shape, np.arange(n), np.full(n, -1.0 / 3.0))
    rounded = stochastic_round(s, seed=DEFAULT_SEED)
    assert rounded.values.dtype == np.int64
    mean = rounded.total() / n
    sigma = math.sqrt((1 / 3) * (2 / 3) / n)  # Bernoulli spread of the fraction
    assert abs(mean - (-1 / 3)) < 3 * sigma
    # Plain rounding sends every -1/3 to 0: the mean is off by 1/3, hundreds of sigma.
    plain = np.rint(s.values)
    assert abs(plain.mean() - (-1 / 3)) > 100 * sigma
    # Reproducible from the seed.
    again = stochastic_round(s, seed=DEFAULT_SEED)
    np.testing.assert_array_equal(again.index, rounded.index)
    # Mixed values: total within statistics, every cell floor or ceil.
    rng = np.random.default_rng(1)
    v = rng.uniform(-5, 5, size=n)
    s2 = SparseMatrix(shape, np.arange(n), v)
    r2 = stochastic_round(s2)
    d = to_dense(r2, dtype=np.int64).ravel()
    assert np.all((d == np.floor(v)) | (d == np.ceil(v)))
    frac = v - np.floor(v)
    assert abs(d.sum() - v.sum()) < 3 * math.sqrt(np.sum(frac * (1 - frac)))


# ----------------------------------------------------------------------
# Whole chain on the synthetic generator
# ----------------------------------------------------------------------

def test_recovered_net_prompt_matches_injected_true_coincidences():
    n_true = 2000
    ch, t, e = synthetic_stream(seed=11, n_true=n_true, rate_hz=2e4, duration_s=1.0)
    pairs = pair_events(make_chunks(ch, t, e, chunk_size=5000), max_dt_ps=MAX_DT)
    w, _, _ = window_weight(PROMPT, RANDOMS, G, MAX_DT)
    net, sigma, p, r, s = net_and_sigma(pairs, w)
    assert r.total() > 500  # the random windows measured something
    assert abs(net - n_true) < 3 * sigma, (net, sigma)
    # The true line sits at row line_y, column line_x and survives subtraction.
    d = to_dense(s)
    assert d[1000, 1500] == pytest.approx(n_true, abs=3 * math.sqrt(n_true))
    # Rounded for .mtx, the total is kept within statistics.
    rounded = stochastic_round(s)
    assert abs(rounded.total() - net) < 3 * math.sqrt(len(s.values))


def test_flat_random_only_control_subtracts_to_zero_and_wrong_w_is_caught():
    ch, t, e = synthetic_stream(seed=5, n_true=0, rate_hz=1e5, duration_s=1.0)
    pairs = pair_events(make_chunks(ch, t, e, chunk_size=50_000), max_dt_ps=MAX_DT)
    w, n_p, n_r = window_weight(PROMPT, RANDOMS, G, MAX_DT)
    net, sigma, p, r, _ = net_and_sigma(pairs, w)
    print(f"control: P={p.total()} R={r.total()} w={w:.6f} net={net:.1f} sigma={sigma:.1f}")
    assert p.total() > 200 and r.total() > 15000  # denominators, not an empty check
    assert abs(net) < 3 * sigma
    # Must-fail: exclusive widths (x2 - x1 without the +1 per mark) give
    # w = 2/200, which leaves ~100 random counts unsubtracted.
    wrong = (n_p - 1) / (n_r - 2)
    net_wrong, sigma_wrong, *_ = net_and_sigma(pairs, wrong)
    print(f"wrong w={wrong:.6f}: net={net_wrong:.1f} sigma={sigma_wrong:.1f}")
    assert abs(net_wrong) > 3 * sigma_wrong


# ----------------------------------------------------------------------
# Real data (untracked; skips by name when absent)
# ----------------------------------------------------------------------

RUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "Na22_261126_ANI_STEF_180")
FILTERED_ROOT = os.path.join(RUN, "FILTERED", "SDataF_Na22_261126_ANI_STEF.root")
UNFILTERED_ROOT = os.path.join(RUN, "UNFILTERED", "SData_Na22_261126_ANI_STEF.root")
EE_JSON = os.path.join(
    RUN, "FILTERED", "CH1@DT5780SCP_58363_E_EspectrumF_Na22_261126_ANI_STEF_20261126_115129.json")

needs_run = pytest.mark.skipif(
    not os.path.isdir(RUN), reason="Na22_261126_ANI_STEF_180/ (untracked real run) is absent")


def read_root_chunks(path, tree, chunk_size=100_000):
    uproot = pytest.importorskip("uproot")
    with uproot.open(path) as f:
        a = f[tree].arrays(["Channel", "Timestamp", "Energy", "Flags"], library="np")
    return make_chunks(a["Channel"], a["Timestamp"].astype(np.int64), a["Energy"],
                       a["Flags"], chunk_size=chunk_size)


@needs_run
def test_real_filtered_exact_time_pairs_equal_compass_ee_json():
    pairs = pair_events(read_root_chunks(FILTERED_ROOT, "Data_F"), max_dt_ps=0)
    with open(EE_JSON) as f:
        data = np.asarray(json.load(f)["data"], dtype=np.int64)  # data[x][y], X = CH1
    assert data.shape == (1024, 1024)
    assert len(pairs.dt) == 39914 == int(data.sum())

    prompt = sort_matrix(pairs.ex, pairs.ey).matrix
    # CoMPASS's 2-D grid: 1023 bins over [0, 16383] plus an overflow bin.
    xb = np.minimum(np.arange(16384) * 1023 // 16383, 1023)
    x_proj = np.bincount(xb, weights=prompt.x_projection(), minlength=1024)
    y_proj = np.bincount(xb, weights=prompt.y_projection(), minlength=1024)
    np.testing.assert_array_equal(x_proj, data.sum(axis=1))  # X = first index = CH1
    np.testing.assert_array_equal(y_proj, data.sum(axis=0))
    # Cell for cell: row = Y bin, column = X bin, so the JSON is its transpose.
    rows, cols = xb[prompt.rows], xb[prompt.cols]
    grid = np.zeros((1024, 1024), dtype=np.int64)
    np.add.at(grid, (rows, cols), prompt.values)
    np.testing.assert_array_equal(grid, data.T)


@needs_run
def test_real_unfiltered_all_combinations_at_least_nearest_partner():
    pairs = pair_events(read_root_chunks(UNFILTERED_ROOT, "Data"), max_dt_ps=10 * NS)
    print(f"UNFILTERED all-combination pairs |dt| <= 10 ns: {len(pairs.dt)}; "
          f"dt == 0: {int(np.count_nonzero(pairs.dt == 0))}")
    assert len(pairs.dt) >= 43387


def test_exclude_by_flag_values_keeps_values_a_mask_would_drop():
    # The default excluded VALUES OR together into a mask containing 0x800,
    # so a mask would also drop the good 0x800 event; a value set must not.
    flags = [0x0, 0x800, 0x80, 0x480, 0xC80, 0x8000, 0x8800]
    n = len(flags)
    chunks = make_chunks([0] * n, np.arange(n) * G, np.arange(n) + 10, flags)
    excluded = {0x80, 0x480, 0xC80, 0x8000, 0x8800}
    by_value = sort_singles(chunks, channels=(0,), nchan=64, exclude_flags=excluded)
    assert np.flatnonzero(by_value[0]).tolist() == [10, 11]  # 0x0 and 0x800 kept
    assert by_value.excluded == 5
    mask = 0
    for v in excluded:
        mask |= v
    by_mask = sort_singles(chunks, channels=(0,), nchan=64, exclude_flags=mask)
    assert np.flatnonzero(by_mask[0]).tolist() == [10]  # the mask loses 0x800


@needs_run
def test_real_raw_minus_default_flag_values_equals_unfiltered_singles():
    raw_root = os.path.join(RUN, "RAW", "SDataR_Na22_261126_ANI_STEF.root")
    excluded = {0x80, 0x480, 0xC80, 0x8000, 0x8800}
    raw = sort_singles(read_root_chunks(raw_root, "Data_R"), exclude_flags=excluded)
    unf = sort_singles(read_root_chunks(UNFILTERED_ROOT, "Data"))
    for c in (0, 1):
        np.testing.assert_array_equal(raw[c], unf[c])
    assert raw.excluded == 831
