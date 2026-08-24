import functools
import os
import struct
import time

import numpy as np
import pytest

from histogram_io import ParseError
from lc_codec import decode_row, zigzag_decode
from mtx_io import MAGIC_LC, load_mtx

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@functools.lru_cache(maxsize=None)
def _load_cached(filename):
    """Loads and decodes a real fixture file once per test run and
    reuses the result across every test that needs it. decode_row (in
    lc_codec.py) is a pure-Python per-value loop over up to 8192x8192
    tag bytes (~7.5s for one load of gg.mtx) -- without this cache, the
    24 real-file loads below would re-decode the same two files from
    scratch every time, which used to make this file alone take ~316s."""
    return load_mtx(os.path.join(FIXTURES, filename))


# Oracle values obtained by compiling libmfile-1.0.7 from source (under
# WSL, bypassing its CRLF-broken autotools configure by invoking gcc
# directly on the .c files) and calling its real mgetint() against the
# actual fixture files -- not hand-computed. checksum = sum(v*(i+1) for
# i, v in enumerate(row)), a cheap way to catch a decode error anywhere
# in a full 8192-value row, not just its first few values.
_ORACLE_ROWS = {
    ("gpff.mtx", 0): {"sum": 301, "min": -5, "max": 6, "checksum": 278695},
    ("gpff.mtx", 1): {"sum": 57, "min": -2, "max": 3, "checksum": 17951},
    ("gpff.mtx", 2): {"sum": 181, "min": -3, "max": 106, "checksum": 65609},
    ("gpff.mtx", 100): {"sum": 74, "min": -2, "max": 10, "checksum": 34558},
    ("gpff.mtx", 500): {"sum": 1045, "min": -4, "max": 8, "checksum": 948725},
    ("gpff.mtx", 1000): {"sum": 9140, "min": -9, "max": 52, "checksum": 6801414},
    ("gpff.mtx", 4096): {"sum": 3446, "min": -16, "max": 43, "checksum": 2809152},
    ("gpff.mtx", 8000): {"sum": 153, "min": -2, "max": 3, "checksum": 101229},
    ("gpff.mtx", 8191): {"sum": 108, "min": -2, "max": 2, "checksum": 93110},
    ("gg.mtx", 0): {"sum": 331252, "min": -6, "max": 206242, "checksum": 83310174},
    ("gg.mtx", 1): {"sum": 30165, "min": -3, "max": 9669, "checksum": 14434711},
    ("gg.mtx", 2): {"sum": 3964, "min": -3, "max": 955, "checksum": 2078348},
    ("gg.mtx", 100): {"sum": 1059634, "min": -1, "max": 2788, "checksum": 813141814},
    ("gg.mtx", 500): {"sum": 402288, "min": -2, "max": 1121, "checksum": 315223325},
    ("gg.mtx", 1000): {"sum": 189809, "min": -2, "max": 403, "checksum": 144945290},
    ("gg.mtx", 4096): {"sum": 14828, "min": -3, "max": 145, "checksum": 10284507},
    ("gg.mtx", 8000): {"sum": 0, "min": 0, "max": 0, "checksum": 0},
    ("gg.mtx", 8191): {"sum": 0, "min": 0, "max": 0, "checksum": 0},
}


def _row_checksum(row):
    return int(sum(int(v) * (i + 1) for i, v in enumerate(row)))


@pytest.mark.parametrize("filename", ["gpff.mtx", "gg.mtx"])
def test_load_mtx_real_file_shape_and_dtype(filename):
    data = _load_cached(filename)
    assert data.shape == (8192, 8192)
    assert data.dtype == np.int64


@pytest.mark.parametrize("filename,row", list(_ORACLE_ROWS.keys()))
def test_load_mtx_real_file_row_matches_oracle(filename, row):
    data = _load_cached(filename)
    expected = _ORACLE_ROWS[(filename, row)]
    actual_row = data[row, :]
    assert int(actual_row.sum()) == expected["sum"]
    assert int(actual_row.min()) == expected["min"]
    assert int(actual_row.max()) == expected["max"]
    assert _row_checksum(actual_row) == expected["checksum"]


def test_load_mtx_real_file_spot_check_values():
    gpff = _load_cached("gpff.mtx")
    assert list(gpff[0, :20]) == [
        6, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1,
    ]
    assert list(gpff[1, :20]) == [
        3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    gg = _load_cached("gg.mtx")
    assert list(gg[0, :20]) == [
        206242, 9669, 955, 234, 137, 82, 78, 64, 49, 39, 36, 49, 52, 57, 61, 53, 65, 76, 116, 132,
    ]
    assert list(gg[1, :20]) == [
        9669, 690, 107, 21, 14, 8, 3, 6, 5, 5, 5, 4, 4, 11, 18, 15, 14, 11, 9, 7,
    ]


def test_load_mtx_real_file_negative_values_preserved():
    # gpff.mtx level 0 is real random-coincidence-subtracted data,
    # confirmed by the user -- negative values must survive unclamped.
    gpff = _load_cached("gpff.mtx")
    assert (gpff < 0).any()


def test_load_mtx_symmetric_file_is_actually_symmetric():
    gg = _load_cached("gg.mtx")
    assert gg[10, 2000] == gg[2000, 10] == 1


@functools.lru_cache(maxsize=None)
def _compressed_rows(filename, count):
    """The raw per-row byte strings lc_codec.decode_row actually
    consumes, straight out of a real fixture's row table, plus that
    matrix's column count. Lets a benchmark drive the decoder directly
    instead of going through load_mtx() -- which would also time file
    I/O and numpy assembly (~30% of its runtime), diluting exactly the
    thing under test. Cached because the fixture is 31 MB."""
    with open(os.path.join(FIXTURES, filename), "rb") as f:
        blob = f.read()
    (_magic, _version, _levels, lines, columns, poslentablepos,
     _freepos, _freelistpos, _used, _free, _status) = struct.unpack("<11I", blob[:44])
    table = struct.unpack(f"<{lines * 2}I", blob[poslentablepos:poslentablepos + lines * 8])
    rows = []
    for row in range(lines):
        pos, length = table[row * 2], table[row * 2 + 1]
        if length == 0:
            continue  # never written -- no tag stream to decode
        rows.append(blob[pos:pos + length])
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise AssertionError(
            f"{filename} yielded only {len(rows)} non-empty rows, needed {count} -- "
            f"timing a decoder over too few rows measures scheduler noise, not "
            f"decode speed, and would make the ratio below meaningless"
        )
    return tuple(rows), columns


def _unoptimized_decode_row(data, num_values, path, *, kind):
    """A FROZEN copy of lc_codec.decode_row exactly as it stood before
    commit c9ffee6 optimized it -- retrieved verbatim from
    `git show c9ffee6^:lc_codec.py`, not reconstructed from memory.

    This is the baseline the speed guard below measures against, so it
    must keep behaving like the OLD code forever. Do NOT "clean this
    up", dedupe it against lc_codec.decode_row, or apply the same
    optimizations to it -- any of those silently turns the guard into a
    comparison of the current implementation against itself, which can
    never fail. It is duplicated logic on purpose."""
    values = []
    last = 0
    pos = 0
    nleft = num_values
    try:
        while nleft > 0:
            t = data[pos]
            pos += 1

            if t & 0x80:
                n = t & 0x3F
                if n > 59:
                    bytes_extra = n - 59
                    n = 59
                    for i in range(bytes_extra):
                        b = data[pos]
                        pos += 1
                        n += (b + 1) << (i * 8)

                if t & 0x40:
                    diff = n & 1
                    same = (n >> 1) + 3
                    values.append(last + diff)
                    nleft -= same
                    if nleft <= 0:
                        raise ParseError(f"{kind}: same-run tag overruns row: {path}")
                    values.extend([last] * same)
                else:
                    last = last + zigzag_decode(n)
                    values.append(last)
                nleft -= 1

            elif t & 0x40:
                nleft -= 2
                if nleft < 0:
                    raise ParseError(f"{kind}: 2-value pack overruns row: {path}")
                a = t & 0x7
                b = (t >> 3) & 0x7
                values.append(last + zigzag_decode(a))
                last = last + zigzag_decode(b)
                values.append(last)

            else:
                nleft -= 3
                if nleft < 0:
                    raise ParseError(f"{kind}: 3-value pack overruns row: {path}")
                a = t & 0x3
                b = (t >> 2) & 0x3
                c = (t >> 4) & 0x3
                values.append(last + zigzag_decode(a))
                values.append(last + zigzag_decode(b))
                last = last + zigzag_decode(c)
                values.append(last)
    except IndexError as exc:
        raise ParseError(f"{kind}: row data ends mid-tag: {path}") from exc

    return values


_BENCHMARK_ROWS = 100
# Ratio of (current decode) / (pre-c9ffee6 decode) that must be beaten.
# Measured on the machine this was written on: 0.842-0.846 across six
# fresh processes (spread 0.004). A wholesale revert to the old decoder
# puts it at ~1.0 by construction. 0.95 sits between the two with ~26x
# the observed spread as headroom on the passing side, and still fails a
# revert by a clear 0.05.
_MAX_DECODE_RATIO = 0.95


def _best_decode_time(decoder, rows, columns, repeats=3):
    """Fastest of `repeats` passes over the same rows. Best-of, not
    mean: scheduler noise and other load can only ever ADD time, so the
    minimum is the closest available estimate of the real cost and the
    most reproducible thing to compare."""
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        for row_bytes in rows:
            decoder(row_bytes, columns, "benchmark", kind="lc matrix file")
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return best


def test_decode_row_is_faster_than_the_pre_optimization_implementation():
    """Timing regression guard for lc_codec.decode_row, expressed as a
    RATIO against a frozen copy of the pre-optimization decoder rather
    than an absolute wall-clock bound.

    The previous version of this test asserted `elapsed < 4.4` seconds
    for a full load_mtx("gg.mtx"). That number was calibrated on one
    machine and made the test a property of the hardware: it failed on
    a ~25% slower box with the optimization verifiably present, and no
    threshold could fix that -- anything loose enough to pass there
    would also let a genuine regression pass here, destroying the guard
    on the machine where it worked.

    Timing both decoders in the same process on the same rows cancels
    machine speed out entirely: a slow machine slows both sides equally
    and the ratio holds."""
    rows, columns = _compressed_rows("gg.mtx", _BENCHMARK_ROWS)

    # Same input, same output -- otherwise the two sides aren't
    # comparable and the ratio would be meaningless. This also catches
    # the frozen reference drifting away from the real decoder's
    # semantics if the format ever gains a tag kind.
    for row_bytes in rows:
        assert decode_row(row_bytes, columns, "x", kind="lc matrix file") == \
            _unoptimized_decode_row(row_bytes, columns, "x", kind="lc matrix file")

    current = _best_decode_time(decode_row, rows, columns)
    baseline = _best_decode_time(_unoptimized_decode_row, rows, columns)

    assert current < baseline * _MAX_DECODE_RATIO, (
        f"decode_row is not meaningfully faster than the pre-c9ffee6 "
        f"implementation: {current:.4f}s vs {baseline:.4f}s "
        f"(ratio {current / baseline:.3f}, must be < {_MAX_DECODE_RATIO})"
    )


def test_decode_speed_guard_would_catch_a_reverted_optimization():
    """Control for the test above: the harness must be able to FAIL.

    Runs the frozen pre-optimization decoder against itself. That is
    exactly what the measurement would see if someone reverted
    lc_codec.decode_row, so the ratio has to land near 1.0 and miss the
    threshold. Without this, a benchmark that silently measured nothing
    (a cached result, an empty row list, both sides accidentally bound
    to the same function) would report a passing ratio and the real
    guard above would be worthless -- a broken check reads as a result."""
    rows, columns = _compressed_rows("gg.mtx", _BENCHMARK_ROWS)

    first = _best_decode_time(_unoptimized_decode_row, rows, columns)
    second = _best_decode_time(_unoptimized_decode_row, rows, columns)

    assert not (first < second * _MAX_DECODE_RATIO), (
        f"the speed guard cannot distinguish a decoder from itself "
        f"({first:.4f}s vs {second:.4f}s, ratio {first / second:.3f}) -- "
        f"it would not catch a reverted optimization"
    )


def _build_lc_header(levels, lines, columns, poslentablepos, version=2):
    return struct.pack(
        "<11I", MAGIC_LC, version, levels, lines, columns, poslentablepos, 0, 0, 0, 0, 0
    )


def _build_minimal_lc_file(tmp_path, rows_bytes, columns, name="test.mtx"):
    """rows_bytes: list of compressed-row byte strings, one per row,
    level 0 (levels=1). An empty bytes object for a row means "never
    written" -- table entry (0, 0), matching the real format's
    convention for a row libmfile's readline() short-circuits to zero
    without decoding."""
    lines = len(rows_bytes)
    header_size = 44
    poslentablepos = header_size
    header = _build_lc_header(1, lines, columns, poslentablepos)

    table = b""
    data = b""
    pos = header_size + lines * 8
    for row in rows_bytes:
        table += struct.pack("<II", pos if row else 0, len(row))
        data += row
        pos += len(row)

    path = tmp_path / name
    path.write_bytes(header + table + data)
    return str(path)


def test_load_mtx_single_fallback_values(tmp_path):
    # Hand-encoded against lc2_uncompress's traced algorithm: last=0;
    # tag 0x8A (n=10, plain, <=59) -> zigzag_decode(10)=5, last=5;
    # tag 0x8D (n=13) -> zigzag_decode(13)=-7, last=-2;
    # tag 0x84 (n=4) -> zigzag_decode(4)=2, last=0.
    row0 = bytes([0x8A, 0x8D, 0x84])
    path = _build_minimal_lc_file(tmp_path, [row0], columns=3)
    data = load_mtx(path)
    assert data.shape == (1, 3)
    assert list(data[0]) == [5, -2, 0]


def test_load_mtx_zero_length_row_is_all_zero(tmp_path):
    row0 = bytes([0x8A, 0x8D, 0x84])
    path = _build_minimal_lc_file(tmp_path, [row0, b""], columns=3)
    data = load_mtx(path)
    assert list(data[1]) == [0, 0, 0]


def test_load_mtx_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.mtx"
    path.write_bytes(struct.pack("<11I", 0xDEADBEEF, 2, 1, 1, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_truncated_header_raises(tmp_path):
    path = tmp_path / "short.mtx"
    path.write_bytes(b"\x00" * 10)
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_unsupported_version_raises(tmp_path):
    path = tmp_path / "v1.mtx"
    path.write_bytes(struct.pack("<11I", MAGIC_LC, 1, 1, 1, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_zero_levels_raises(tmp_path):
    path = tmp_path / "zerolev.mtx"
    path.write_bytes(struct.pack("<11I", MAGIC_LC, 2, 0, 1, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_oversized_columns_raises_parse_error_not_memory_error(tmp_path):
    # Regression guard: lines/columns come straight from the file header
    # with no bound check before sizing np.zeros((lines, columns)). A
    # corrupted columns field (e.g. 0xFFFFFFFF) would otherwise attempt a
    # multi-GB allocation and raise MemoryError -- not a subclass of
    # OSError, so it would propagate uncaught past load_mtx's own `except
    # OSError`, past main_window._open_matrix_dialog's `except ParseError`,
    # and crash the app instead of showing "Could not open matrix."
    path = tmp_path / "hugecols.mtx"
    path.write_bytes(struct.pack("<11I", MAGIC_LC, 2, 1, 1, 0xFFFFFFFF, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_oversized_lines_raises_parse_error(tmp_path):
    path = tmp_path / "hugelines.mtx"
    path.write_bytes(struct.pack("<11I", MAGIC_LC, 2, 1, 0xFFFFFFFF, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_truncated_row_table_raises(tmp_path):
    # Header claims 5 lines (needs a 40-byte table) but the file ends
    # right after the header.
    path = tmp_path / "notable.mtx"
    path.write_bytes(_build_lc_header(1, 5, 3, 44))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_truncated_row_data_raises(tmp_path):
    # Table claims row 0 is 3 bytes long at offset 52, but only 1 byte
    # actually follows the table in the file.
    header = _build_lc_header(1, 1, 3, 44)
    table = struct.pack("<II", 52, 3)
    path = tmp_path / "trunc.mtx"
    path.write_bytes(header + table + bytes([0x8A]))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_same_run_and_packed_tags(tmp_path):
    # Exercises the 2-pack and same-run branches together (the
    # real-file tests above already exercise every branch on real
    # data; this is a small from-first-principles cross-check). Both
    # 2-pack values are computed against the SAME original `last`, not
    # chained -- this is the single easiest part of the format to get
    # wrong, per the design research, so it's spelled out in full:
    #   0x69 (2-pack): a=1->decode(1)=-1 -> value1 = last(0)+(-1) = -1
    #                  (last is NOT updated by this first value);
    #                  b=5->decode(5)=-3 -> last = 0+(-3) = -3,
    #                  value2 = -3 (last IS updated by the second value)
    #   0xC6 (same-run, n=6): diff=6&1=0, same=(6>>1)+3=6 ->
    #                  value = last(-3)+diff(0) = -3, then 6 more
    #                  copies of -3 (last stays -3 throughout,
    #                  unchanged by a same-run tag) -- 7 values total
    # 2 (2-pack) + 7 (same-run) = 9 values.
    row0 = bytes([0x69, 0xC6])
    path = _build_minimal_lc_file(tmp_path, [row0], columns=9)
    data = load_mtx(path)
    assert list(data[0]) == [-1, -3, -3, -3, -3, -3, -3, -3, -3]


def test_load_mtx_reads_the_file_without_a_syscall_per_row(monkeypatch):
    """The row loop used to seek+read once per line -- 8194 read() calls
    for a real 8192-line matrix. That is invisible on a local disk but
    dominates on a network or virtualised filesystem, where per-call
    latency is what costs (measured: 1.63 s on WSL's 9p mount vs 0.02 s
    for the same file on native ext4, doubling the whole load).

    Asserts the CALL COUNT, not elapsed time, so it stays meaningful on
    any machine -- unlike a wall-clock threshold, which is calibrated to
    one CPU.
    """
    import builtins

    real_open = builtins.open
    reads = []

    class _CountingFile:
        def __init__(self, handle):
            self._handle = handle

        def read(self, *args):
            reads.append(1)
            return self._handle.read(*args)

        def seek(self, *args):
            return self._handle.seek(*args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

    def counting_open(path, *args, **kwargs):
        return _CountingFile(real_open(path, *args, **kwargs))

    monkeypatch.setattr(builtins, "open", counting_open)
    data = load_mtx(os.path.join(FIXTURES, "gg.mtx"))
    monkeypatch.undo()

    assert data.shape == (8192, 8192)
    assert len(reads) <= 2, (
        f"expected the file to be read in one bulk call, got {len(reads)} read() calls "
        "-- has the per-row seek+read crept back in?"
    )
