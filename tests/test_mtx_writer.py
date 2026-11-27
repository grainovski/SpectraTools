"""save_mtx: the lc-format (.mtx) matrix writer added in 7.0.0.

The strongest check here is byte equality with a file written by libmfile
itself. tests/fixtures/libmfile_written_7x9.mtx was produced on
2026-11-27 by building the vendored libmfile-1.0.7 from source in WSL
(Ubuntu-24.04; CRLFs stripped, then its own ./configure && make) and
running a ten-line C program that opens a new file with mopen(..., "w"),
sets MAT_LC version 2, 1 level, 7 lines, 9 columns with msetinfo, puts
each line once in order with a full-width mputint, and calls mclose. The
values are _REF below. The same harness compared save_mtx against
libmfile on 40 further random matrices (shapes up to 59 x 699; Poisson,
signed, sparse, full int32 range, and tag-boundary values), dense and
sparse input: all 41 files byte-identical.
"""

import os
import struct
import tracemalloc

import numpy as np
import pytest

from lc_codec import INT32_MAX, INT32_MIN, decode_row, encode_row
from mtx_io import MAGIC_LC, coalesce_sparse, load_mtx, save_mtx

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
LIBMFILE_REF = os.path.join(FIXTURES, "libmfile_written_7x9.mtx")

_REF = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],                               # all-zero row
    [5, -2, 0, 1, 1, 1, 1, 1, 40000],
    [INT32_MAX, INT32_MIN, INT32_MAX, 0, INT32_MIN, -1, 7, 7, 7],
    [-3, -3, -3, -3, -3, 2, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [123456789, -987654321, 0, 0, 0, 0, 0, 0, 1],
], dtype=np.int64)


def _sparse(m):
    flat = np.flatnonzero(m)
    return (m.shape, flat, m.ravel()[flat])


def _header(blob):
    return struct.unpack("<11I", blob[:44])


# --- byte compatibility with libmfile ------------------------------------


def test_dense_output_is_byte_identical_to_libmfile(tmp_path):
    path = tmp_path / "py.mtx"
    save_mtx(str(path), _REF)
    with open(LIBMFILE_REF, "rb") as f:
        reference = f.read()
    assert path.read_bytes() == reference


def test_sparse_output_is_byte_identical_to_libmfile(tmp_path):
    path = tmp_path / "py.mtx"
    save_mtx(str(path), _sparse(_REF))
    with open(LIBMFILE_REF, "rb") as f:
        reference = f.read()
    assert path.read_bytes() == reference


def test_byte_comparison_fails_when_one_cell_differs(tmp_path):
    """Must-fail control: the equality above is not vacuous."""
    changed = _REF.copy()
    changed[3, 4] += 1
    path = tmp_path / "py.mtx"
    save_mtx(str(path), changed)
    with open(LIBMFILE_REF, "rb") as f:
        assert path.read_bytes() != f.read()


def test_libmfile_written_file_reads_back_through_load_mtx():
    """Includes INT32_MAX next to INT32_MIN, which libmfile stores as a
    WRAPPED delta of +1 -- load_mtx must wrap it back as TV does."""
    assert np.array_equal(load_mtx(LIBMFILE_REF), _REF)


def test_header_fields_match_libmfile_new_file(tmp_path):
    path = tmp_path / "m.mtx"
    save_mtx(str(path), _REF)
    blob = path.read_bytes()
    (magic, version, levels, lines, columns, poslentablepos,
     freepos, freelistpos, used, free, status) = _header(blob)
    assert magic == MAGIC_LC == 0x80FFFF10
    assert (version, levels, lines, columns) == (2, 1, 7, 9)
    assert poslentablepos == 44                       # lc_minfo.c:107
    assert freepos == len(blob)                       # end of the last row
    assert (freelistpos, used, free, status) == (0, 0, 0, 0)  # lc_minfo.c:109, :204-206
    table = struct.unpack("<14I", blob[44:44 + 7 * 8])
    pos = 44 + 7 * 8                                  # rows start after the table
    for row in range(7):
        assert table[2 * row] == pos
        pos += table[2 * row + 1]
    assert pos == freepos


def test_encoder_reproduces_every_sampled_row_of_the_real_matrices():
    """The real gg.mtx / gpff.mtx fixtures were written by TV's libmfile:
    decode a sample of rows from both levels and re-encode them with the
    wrap32 encoder save_mtx uses -- the bytes must be the file's own."""
    checked = 0
    for name in ("gg.mtx", "gpff.mtx"):
        with open(os.path.join(FIXTURES, name), "rb") as f:
            blob = f.read()
        _m, _v, levels, lines, columns, tablepos = _header(blob)[:6]
        n = levels * lines
        table = struct.unpack(f"<{2 * n}I", blob[tablepos:tablepos + 8 * n])
        for row in range(0, n, 97):
            pos, length = table[2 * row], table[2 * row + 1]
            raw = blob[pos:pos + length]
            values = decode_row(raw, columns, name, kind="lc matrix file")
            assert encode_row(np.array(values), wrap32=True) == raw, (name, row)
            checked += 1
    assert checked == 2 * len(range(0, 2 * 8192, 97))


# --- round trips ---------------------------------------------------------


@pytest.mark.parametrize("form", ["dense", "sparse"])
def test_round_trip_non_square_pins_orientation(tmp_path, form):
    """5 lines (Y) x 3 columns (X): a transposed write would change the
    shape and move the marker."""
    m = np.zeros((5, 3), dtype=np.int64)
    m[4, 2] = 7
    m[1, 0] = -3
    path = str(tmp_path / "m.mtx")
    save_mtx(path, m if form == "dense" else _sparse(m))
    back = load_mtx(path)
    assert back.shape == (5, 3)
    assert back[4, 2] == 7 and back[1, 0] == -3
    assert _header(open(path, "rb").read())[3:5] == (5, 3)  # lines, columns


@pytest.mark.parametrize("form", ["dense", "sparse"])
def test_round_trip_random_signed_matrix(tmp_path, form):
    rng = np.random.default_rng(7)
    m = (rng.poisson(2, (40, 130)) - rng.poisson(2, (40, 130))).astype(np.int64)
    m[11] = 0                                           # an all-zero row
    m[0, 0], m[0, 1] = INT32_MIN, INT32_MAX             # extremes, adjacent
    path = str(tmp_path / "m.mtx")
    save_mtx(path, m if form == "dense" else _sparse(m))
    assert np.array_equal(load_mtx(path), m)


def test_all_zero_row_is_encoded_not_left_unwritten(tmp_path):
    """libmfile leaves len 0 only for a line nobody put; a put all-zero
    line is compressed like any other (writeline, lc_getput.c:111)."""
    path = tmp_path / "m.mtx"
    save_mtx(str(path), _sparse(_REF))
    table = struct.unpack("<14I", path.read_bytes()[44:44 + 56])
    assert table[1] > 0                                 # row 0 is all zero
    assert load_mtx(str(path))[0].tolist() == [0] * 9


def test_whole_matrix_of_zeros(tmp_path):
    path = str(tmp_path / "m.mtx")
    save_mtx(path, ((3, 70000 // 2), np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
    assert not load_mtx(path).any()


def test_unsigned_and_narrow_integer_dtypes_are_accepted(tmp_path):
    m = np.array([[1, 2], [3, 65535]], dtype=np.uint16)
    path = str(tmp_path / "m.mtx")
    save_mtx(path, m)
    assert load_mtx(path).tolist() == [[1, 2], [3, 65535]]


def test_sparse_duplicates_are_summed(tmp_path):
    path = str(tmp_path / "m.mtx")
    save_mtx(path, ((2, 3), np.array([5, 0, 5, 5]), np.array([1, 4, 2, -1])))
    assert load_mtx(path).tolist() == [[4, 0, 0], [0, 0, 2]]


def test_progress_reports_rows(tmp_path):
    calls = []
    save_mtx(str(tmp_path / "m.mtx"), np.zeros((300, 4), dtype=np.int32),
             progress=lambda done, total: calls.append((done, total)))
    assert calls == [(0, 300), (256, 300), (300, 300)]


# --- refusals ------------------------------------------------------------


def test_refuses_float_input(tmp_path):
    with pytest.raises(ValueError, match="integers"):
        save_mtx(str(tmp_path / "m.mtx"), np.array([[1.0, 2.5]]))
    with pytest.raises(ValueError, match="integers"):
        save_mtx(str(tmp_path / "m.mtx"), ((1, 2), np.array([0]), np.array([1.0])))


@pytest.mark.parametrize("value", [INT32_MAX + 1, INT32_MIN - 1, 2**40])
def test_refuses_values_outside_int32(tmp_path, value):
    with pytest.raises(ValueError, match="32-bit"):
        save_mtx(str(tmp_path / "m.mtx"), np.array([[0, value]], dtype=np.int64))
    with pytest.raises(ValueError, match="32-bit"):
        save_mtx(str(tmp_path / "s.mtx"), ((1, 2), np.array([1]), np.array([value])))


def test_refuses_sum_of_duplicates_outside_int32(tmp_path):
    with pytest.raises(ValueError, match="32-bit"):
        save_mtx(str(tmp_path / "m.mtx"),
                 ((1, 1), np.array([0, 0]), np.array([INT32_MAX, 1])))


def test_refuses_bad_shapes(tmp_path):
    p = str(tmp_path / "m.mtx")
    with pytest.raises(ValueError, match="2-D"):
        save_mtx(p, np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError, match="65536"):
        save_mtx(p, ((1, 65537), np.array([0]), np.array([1])))
    with pytest.raises(ValueError):
        save_mtx(p, np.zeros((0, 4), dtype=np.int64))


def test_refuses_bad_sparse_input(tmp_path):
    p = str(tmp_path / "m.mtx")
    with pytest.raises(ValueError, match="out of range"):
        save_mtx(p, ((2, 2), np.array([4]), np.array([1])))
    with pytest.raises(ValueError, match="same length"):
        save_mtx(p, ((2, 2), np.array([0, 1]), np.array([1])))
    with pytest.raises(ValueError, match="shape"):
        save_mtx(p, ((2,), np.array([0]), np.array([1])))


def test_coalesce_sparse_sorts_sums_and_drops_zeros():
    ny, nx, idx, vals = coalesce_sparse((2, 3), [4, 1, 4, 2], [1, 5, -1, 0])
    assert (ny, nx) == (2, 3)
    assert idx.tolist() == [1] and vals.tolist() == [5]


# --- full-resolution sparse matrix ---------------------------------------


def test_sparse_16k_matrix_writes_without_a_dense_array(tmp_path):
    """16384 x 16384 with 10**5 entries. A dense int64 matrix would be
    2 GiB and even an int8 one 256 MiB; the writer's peak must stay a
    small fraction of the smallest of those."""
    n = 16384
    rng = np.random.default_rng(20261127)
    flat = rng.choice(n * n, size=100_000, replace=False).astype(np.int64)
    vals = rng.integers(-40, 2000, size=flat.size)
    vals[vals == 0] = 1
    path = str(tmp_path / "big.mtx")

    tracemalloc.start()
    try:
        save_mtx(path, ((n, n), flat, vals))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 32 * 2**20, f"peak {peak / 2**20:.1f} MiB"
    assert peak < n * n // 8                     # far below even 1 bit per cell

    # Read back WITHOUT load_mtx (which would build the 2 GiB array):
    # decode the rows that hold entries from a sample, check the rest are
    # the compressed all-zero row, and compare at the stored cells.
    with open(path, "rb") as f:
        blob = f.read()
    hdr = _header(blob)
    assert hdr[3:5] == (n, n) and hdr[6] == len(blob)
    table = np.frombuffer(blob, dtype="<u4", count=2 * n, offset=44).reshape(n, 2)
    zero_row = encode_row([0] * n, wrap32=True)
    rows = flat // n
    occupied = set(rows.tolist())
    for row in range(n):
        if row not in occupied:
            pos, length = table[row]
            assert blob[pos:pos + length] == zero_row
    sample_rows = sorted(occupied)[::40]
    for row in sample_rows:
        pos, length = (int(v) for v in table[row])
        decoded = np.array(decode_row(blob[pos:pos + length], n, path, kind="lc matrix file"))
        expected = np.zeros(n, dtype=np.int64)
        sel = rows == row
        expected[flat[sel] % n] = vals[sel]
        assert np.array_equal(decoded, expected), row
