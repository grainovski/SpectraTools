import os
import struct

import numpy as np
import pytest

from histogram_io import ParseError
from mtx_io import MAGIC_LC, load_mtx

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

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
    data = load_mtx(os.path.join(FIXTURES, filename))
    assert data.shape == (8192, 8192)
    assert data.dtype == np.int64


@pytest.mark.parametrize("filename,row", list(_ORACLE_ROWS.keys()))
def test_load_mtx_real_file_row_matches_oracle(filename, row):
    data = load_mtx(os.path.join(FIXTURES, filename))
    expected = _ORACLE_ROWS[(filename, row)]
    actual_row = data[row, :]
    assert int(actual_row.sum()) == expected["sum"]
    assert int(actual_row.min()) == expected["min"]
    assert int(actual_row.max()) == expected["max"]
    assert _row_checksum(actual_row) == expected["checksum"]


def test_load_mtx_real_file_spot_check_values():
    gpff = load_mtx(os.path.join(FIXTURES, "gpff.mtx"))
    assert list(gpff[0, :20]) == [
        6, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1,
    ]
    assert list(gpff[1, :20]) == [
        3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    gg = load_mtx(os.path.join(FIXTURES, "gg.mtx"))
    assert list(gg[0, :20]) == [
        206242, 9669, 955, 234, 137, 82, 78, 64, 49, 39, 36, 49, 52, 57, 61, 53, 65, 76, 116, 132,
    ]
    assert list(gg[1, :20]) == [
        9669, 690, 107, 21, 14, 8, 3, 6, 5, 5, 5, 4, 4, 11, 18, 15, 14, 11, 9, 7,
    ]


def test_load_mtx_real_file_negative_values_preserved():
    # gpff.mtx level 0 is real random-coincidence-subtracted data,
    # confirmed by the user -- negative values must survive unclamped.
    gpff = load_mtx(os.path.join(FIXTURES, "gpff.mtx"))
    assert (gpff < 0).any()


def test_load_mtx_symmetric_file_is_actually_symmetric():
    gg = load_mtx(os.path.join(FIXTURES, "gg.mtx"))
    assert gg[10, 2000] == gg[2000, 10] == 1


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
