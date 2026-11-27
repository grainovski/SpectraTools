"""save_json_matrix and the SpectraTools JSON layouts (7.0.0, option (c)).

The dense layout's orientation is pinned against load_json_matrix, whose
own orientation is proven cell for cell against CoMPASS's ROOT file in
tests/test_json_matrix_io.py -- so writing data[x][y] here and reading it
back there closes the loop to CoMPASS itself.
"""

import json
import tracemalloc

import numpy as np
import pytest

from channel_axis import ChannelAxis
from histogram_io import ParseError
from json_matrix_io import (
    DENSE_MAX, axis_for_compression, compress_matrix, load_json_matrix,
    load_json_matrix_with_meta, save_json_matrix,
)

META = {
    "compression": 1,
    "source_run": "Na22_261126_ANI_STEF_180/RAW/SDataR_run.root",
    "windows": {"prompt": [[-20, 20]], "random": [[100, 300]]},
    "w": 0.2,
    "seed": 20261010,
    "gain_coefficients": [0.5, 1.01, 2e-7],
}


def _meta(**changes):
    m = dict(META)
    m.update(changes)
    return m


def _sparse(m):
    flat = np.flatnonzero(m)
    return (m.shape, flat, m.ravel()[flat])


# --- sparse layout -------------------------------------------------------


def test_sparse_round_trip_keeps_exact_floats_and_negatives(tmp_path):
    m = np.zeros((4, 6))
    m[0, 5] = -1.0 / 3.0
    m[3, 0] = 0.1
    m[2, 2] = -7.0
    m[1, 4] = 1234.5678901234567
    m[3, 5] = 5e-300
    path = str(tmp_path / "s.json")
    save_json_matrix(path, m, META)
    back, meta, axes = load_json_matrix_with_meta(path)
    assert back.dtype == np.float64
    assert back.shape == (4, 6)
    assert np.array_equal(back, m)               # bit-exact, not approx
    assert meta["seed"] == 20261010 and meta["axes"] == {"x": "CH1", "y": "CH0"}
    assert axes == (None, None)                  # full resolution


def test_sparse_file_layout_names_x_and_y(tmp_path):
    m = np.zeros((5, 3), dtype=np.int64)         # 5 Y rows, 3 X columns
    m[4, 2] = 9
    path = tmp_path / "s.json"
    save_json_matrix(str(path), m, META)
    doc = json.loads(path.read_text())
    assert doc["format"] == "sparse" and doc["shape"] == [5, 3]
    assert (doc["x"], doc["y"], doc["v"]) == ([2], [4], [9])
    assert doc["spectratools"]["compression"] == 1


def test_sparse_integer_matrix_reads_back_as_int64(tmp_path):
    m = np.array([[0, -3], [4, 0]], dtype=np.int64)
    path = str(tmp_path / "s.json")
    save_json_matrix(path, _sparse(m), META)
    back = load_json_matrix(path)
    assert back.dtype == np.int64 and back.tolist() == [[0, -3], [4, 0]]


def test_sparse_16k_writes_without_a_dense_array(tmp_path):
    n = 16384
    rng = np.random.default_rng(3)
    flat = rng.choice(n * n, size=100_000, replace=False).astype(np.int64)
    vals = rng.normal(0, 5, size=flat.size)
    path = tmp_path / "big.json"
    tracemalloc.start()
    try:
        save_json_matrix(str(path), ((n, n), flat, vals), META)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64 * 2**20, f"peak {peak / 2**20:.1f} MiB"   # dense float64: 2 GiB
    doc = json.loads(path.read_text())
    assert doc["shape"] == [n, n]
    order = np.argsort(flat)
    assert np.array_equal(np.array(doc["y"]) * n + np.array(doc["x"]), flat[order])
    assert np.array_equal(np.array(doc["v"]), vals[order])      # exact floats


# --- dense CoMPASS layout ------------------------------------------------


def test_dense_layout_is_compass_data_x_y(tmp_path):
    m = np.zeros((5, 3), dtype=np.int64)         # [y][x]
    m[4, 2] = 7
    m[0, 1] = -2
    path = tmp_path / "d.json"
    save_json_matrix(str(path), m, _meta(compression=4), layout="dense")
    doc = json.loads(path.read_text())
    assert len(doc["data"]) == 3 and len(doc["data"][0]) == 5   # rows are X
    assert doc["data"][2][4] == 7 and doc["data"][1][0] == -2
    assert doc["xaxis"]["nBins"] == 3 and doc["yaxis"]["nBins"] == 5
    assert "title" in doc
    back, meta, axes = load_json_matrix_with_meta(str(path))
    assert back.shape == (5, 3) and np.array_equal(back, m)
    assert meta["compression"] == 4
    assert axes == (ChannelAxis(1.5, 4.0), ChannelAxis(1.5, 4.0))


def test_dense_layout_from_sparse_input(tmp_path):
    m = np.arange(12, dtype=np.float64).reshape(3, 4) / 4.0
    path = str(tmp_path / "d.json")
    save_json_matrix(path, _sparse(m), META, layout="dense")
    assert np.array_equal(load_json_matrix(path), m)


def test_dense_layout_refused_above_4096(tmp_path):
    big = ((DENSE_MAX + 1, 8), np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match='layout="sparse"'):
        save_json_matrix(str(tmp_path / "d.json"), big, META, layout="dense")
    # The sparse layout takes the same matrix.
    save_json_matrix(str(tmp_path / "s.json"), big, META)


# --- strictness: only SpectraTools files may be signed / fractional -------


def test_compass_file_without_block_keeps_the_strict_check(tmp_path):
    m = np.array([[1, -2], [3, 4]])
    ours = str(tmp_path / "ours.json")
    save_json_matrix(ours, m, META, layout="dense")
    doc = json.loads(open(ours).read())
    del doc["spectratools"]
    theirs = tmp_path / "theirs.json"
    theirs.write_text(json.dumps(doc))
    assert load_json_matrix(ours).tolist() == m.tolist()
    with pytest.raises(ParseError, match="negative"):
        load_json_matrix(str(theirs))


def test_compass_file_reports_no_meta_and_identity_axes(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"data": [[1, 2], [3, 4]]}))
    m, meta, axes = load_json_matrix_with_meta(str(path))
    assert m.dtype == np.int64 and meta is None and axes == (None, None)


def test_bad_block_or_sparse_body_is_a_parse_error(tmp_path):
    cases = [
        {"data": [[1]], "spectratools": [1]},
        {"data": [[1]], "spectratools": {"compression": 0}},
        {"format": "sparse", "shape": [2, 2], "x": [0], "y": [5], "v": [1],
         "spectratools": {"compression": 1}},
        {"format": "sparse", "shape": [2, 2], "x": [0, 1], "y": [0], "v": [1],
         "spectratools": {"compression": 1}},
        {"format": "sparse", "shape": [2, 2], "x": [0], "y": [0], "v": [True],
         "spectratools": {"compression": 1}},
        {"format": "sparse", "spectratools": {"compression": 1}},
    ]
    for i, doc in enumerate(cases):
        path = tmp_path / f"b{i}.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(ParseError) as err:
            load_json_matrix(str(path))
        assert str(path) in str(err.value)


# --- writer refusals -----------------------------------------------------


def test_writer_refusals(tmp_path):
    p = str(tmp_path / "x.json")
    m = np.ones((2, 2))
    incomplete = {k: v for k, v in META.items() if k != "seed"}
    with pytest.raises(ValueError, match="seed"):
        save_json_matrix(p, m, incomplete)
    with pytest.raises(ValueError, match="compression"):
        save_json_matrix(p, m, _meta(compression=0))
    with pytest.raises(ValueError, match="JSON"):
        save_json_matrix(p, m, _meta(source_run=object()))
    with pytest.raises(ValueError, match="finite"):
        save_json_matrix(p, np.array([[np.nan, 1.0]]), META)
    with pytest.raises(ValueError, match="layout"):
        save_json_matrix(p, m, META, layout="csv")


# --- compression and channel axes ----------------------------------------


def test_axis_for_compression():
    assert axis_for_compression(1) is None
    assert axis_for_compression(4) == ChannelAxis(1.5, 4.0)
    # Bin b holds ADC f*b .. f*b+f-1: its channel is their mean.
    assert axis_for_compression(16).to_channel(3) == np.mean(np.arange(48, 64))
    with pytest.raises(ValueError):
        axis_for_compression(0)


def test_compress_matrix_dense_and_sparse_agree():
    rng = np.random.default_rng(5)
    m = rng.integers(-3, 9, (8, 12))
    a = compress_matrix(m, 4)
    b = compress_matrix(_sparse(m), 4)
    assert a.shape == (2, 3) and a.dtype == np.int64
    assert np.array_equal(a, b)
    assert a[1, 2] == m[4:8, 8:12].sum()
    with pytest.raises(ValueError, match="divide"):
        compress_matrix(m, 5)


def test_a_compressed_file_puts_a_peak_at_its_full_resolution_channel(tmp_path):
    """Full-resolution 16384^2 events with a peak at ADC (x, y) =
    (6001.3, 9874.6); compressed by 16 (CoMPASS's own 1024-bin size, and
    a dense file small enough for a unit test), saved dense, reopened: the
    peak's channel through the file's axis must be within f/2 of the
    full-res centroid -- whereas the bare bin index is off by a factor
    of 16.

    The peak is wide (sigma 40 channels, many bins) so the centroid is
    not limited by where a narrow peak happens to fall inside one
    16-channel bin; that lets the check be 0.5 channel instead of f/2,
    tight enough that an axis without the (f-1)/2 offset fails it."""
    n, f = 16384, 16
    rng = np.random.default_rng(11)
    cx, cy = 6001.3, 9874.6
    xs = np.rint(rng.normal(cx, 40.0, 200_000)).astype(np.int64)
    ys = np.rint(rng.normal(cy, 40.0, 200_000)).astype(np.int64)
    flat = ys * n + xs
    full_x = np.average(xs.astype(float))
    full_y = np.average(ys.astype(float))

    compressed = compress_matrix(((n, n), flat, np.ones_like(flat)), f)
    path = str(tmp_path / "c.json")
    save_json_matrix(path, compressed, _meta(compression=f), layout="dense")
    m, _meta_back, (x_axis, y_axis) = load_json_matrix_with_meta(path)
    assert m.shape == (n // f, n // f)

    px, py = m.sum(axis=0), m.sum(axis=1)
    bins = np.arange(n // f)
    x_bin = np.average(bins, weights=px)
    y_bin = np.average(bins, weights=py)
    assert abs(x_axis.to_channel(x_bin) - full_x) <= 0.5
    assert abs(y_axis.to_channel(y_bin) - full_y) <= 0.5
    # Controls: the offset matters, and so does the width.
    assert abs(ChannelAxis(0.0, f).to_channel(x_bin) - full_x) > 5
    # Control: without the axis the same peak reads ~5600 channels low.
    assert abs(x_bin - full_x) > 1000
