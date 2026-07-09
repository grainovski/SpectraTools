import struct
from pathlib import Path

import pytest

from histogram_io import ParseError
from spk_io import MAT_COLMAX, load_spk

FIXTURES = Path(__file__).parent / "fixtures"


def _lc_header(version, levels, lines, columns, poslentablepos):
    return struct.pack(
        "<11I",
        0x80FFFF10, version, levels, lines, columns, poslentablepos,
        0, 0, 0, 0, 0,
    )


def test_parses_real_demo_spk():
    # demo.spk is a real tv/Mfile MAT_LC file (magic 0x80FFFF10, version 2,
    # 4096 channels) -- verified byte-for-byte during design against the
    # libmfile-1.0.7 source, and the decompressed result was independently
    # sanity-checked to have zero negative values and a clear multi-peak
    # gamma-spectrum shape (see design spec).
    data = load_spk(str(FIXTURES / "demo.spk"))
    assert len(data) == 4096
    assert list(data[:20]) == [
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0,
    ]
    assert int(data[724]) == 41580
    assert all(v >= 0 for v in data)
    assert int(data.sum()) == 13069569


def test_lc2_3value_tag(tmp_path):
    # tag byte 0x24 = 0b00_10_00_01... bits: a=0(->+0), b=1(->-1), c=2(->+1)
    # (3-value pack: tag bits 0-1 of the byte's top 2 bits are both 0)
    header = _lc_header(version=2, levels=1, lines=1, columns=3, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_3value.spk"
    file_path.write_bytes(header + poslen + bytes([0x24]))

    data = load_spk(str(file_path))

    assert list(data) == [0, -1, 1]


def test_lc2_2value_tag(tmp_path):
    # tag byte 0x53 = 0x40 + a(3) + (b(2)<<3): a=3(->-2), b=2(->+1)
    header = _lc_header(version=2, levels=1, lines=1, columns=2, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_2value.spk"
    file_path.write_bytes(header + poslen + bytes([0x53]))

    data = load_spk(str(file_path))

    assert list(data) == [-2, 1]


def test_lc2_1value_tag(tmp_path):
    # tag byte 0x85 = 0x80 + 5: n=5 -> zigzag decode -3
    header = _lc_header(version=2, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_1value.spk"
    file_path.write_bytes(header + poslen + bytes([0x85]))

    data = load_spk(str(file_path))

    assert list(data) == [-3]


def test_lc2_same_run_tag(tmp_path):
    # tag byte 0xC7 = 0xC0 + 7: n=7 -> diff=1, same=(7>>1)+3=6.
    # Produces [last+diff] followed by 6 copies of the *unchanged* last (0).
    header = _lc_header(version=2, levels=1, lines=1, columns=7, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_same_run.spk"
    file_path.write_bytes(header + poslen + bytes([0xC7]))

    data = load_spk(str(file_path))

    assert list(data) == [1, 0, 0, 0, 0, 0, 0]


def test_lc2_extended_single_value_tag(tmp_path):
    # tag byte 0xBD: low 6 bits = 61 (0x3D) -> 2 extension bytes needed.
    # extension bytes [0x00, 0x00] -> n = 59 + (0+1) + (0+1)*256 = 316
    # -> zigzag decode = +158
    header = _lc_header(version=2, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 3)
    file_path = tmp_path / "lc2_extended.spk"
    file_path.write_bytes(header + poslen + bytes([0xBD, 0x00, 0x00]))

    data = load_spk(str(file_path))

    assert list(data) == [158]


def test_lc1_3value_tag(tmp_path):
    # tag byte 0x39: a=1,b=2,c=3 (each 2 bits) -> cumulative deltas -1,+1,-2
    header = _lc_header(version=1, levels=1, lines=1, columns=3, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc1_3value.spk"
    file_path.write_bytes(header + poslen + bytes([0x39]))

    data = load_spk(str(file_path))

    assert list(data) == [-1, 0, -2]


def test_lc1_2value_tag(tmp_path):
    # tag byte 0x6B = 0x40 + a(3) + (b(5)<<3) -> cumulative deltas -2,-3
    header = _lc_header(version=1, levels=1, lines=1, columns=2, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc1_2value.spk"
    file_path.write_bytes(header + poslen + bytes([0x6B]))

    data = load_spk(str(file_path))

    assert list(data) == [-2, -5]


def test_lc1_1value_tag(tmp_path):
    # tag byte 0x8A = 0x80 + 10: n=10 -> zigzag decode +5
    header = _lc_header(version=1, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc1_1value.spk"
    file_path.write_bytes(header + poslen + bytes([0x8A]))

    data = load_spk(str(file_path))

    assert list(data) == [5]


def test_lc1_extended_tag(tmp_path):
    # tag byte 0xD4: initial 6 bits = 20, one continuation byte 0x03
    # (bit7 clear = last continuation byte) -> i = 20 + (3<<6) = 212
    # -> zigzag decode +106
    header = _lc_header(version=1, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 2)
    file_path = tmp_path / "lc1_extended.spk"
    file_path.write_bytes(header + poslen + bytes([0xD4, 0x03]))

    data = load_spk(str(file_path))

    assert list(data) == [106]


def test_lc_empty_line_is_all_zero(tmp_path):
    header = _lc_header(version=2, levels=1, lines=1, columns=5, poslentablepos=44)
    poslen = struct.pack("<2I", 0, 0)  # len == 0 means "empty line" per source
    file_path = tmp_path / "lc_empty.spk"
    file_path.write_bytes(header + poslen)

    data = load_spk(str(file_path))

    assert list(data) == [0, 0, 0, 0, 0]


def test_lc_rejects_unsupported_version(tmp_path):
    header = _lc_header(version=3, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc_badversion.spk"
    file_path.write_bytes(header + poslen + bytes([0x80]))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_2d_matrix(tmp_path):
    # levels/lines checked immediately after the header is parsed, before
    # the position table is even read -- so a bare 44-byte header is enough
    # to exercise this rejection.
    header = _lc_header(version=2, levels=1, lines=2, columns=1, poslentablepos=44)
    file_path = tmp_path / "lc_2d.spk"
    file_path.write_bytes(header)

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_truncated_compressed_data(tmp_path):
    header = _lc_header(version=2, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 5)  # claims 5 bytes but none follow
    file_path = tmp_path / "lc_truncated.spk"
    file_path.write_bytes(header + poslen)

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_excessive_channel_count(tmp_path):
    header = _lc_header(version=2, levels=1, lines=1, columns=MAT_COLMAX + 1, poslentablepos=44)
    file_path = tmp_path / "lc_too_many_channels.spk"
    file_path.write_bytes(header)

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_stream_that_runs_out_of_bytes(tmp_path):
    # A 2-value tag only ever produces 2 values, but 3 channels are
    # declared -- decoding must run out of compressed bytes trying to
    # satisfy the third and raise, rather than reading past the buffer.
    header = _lc_header(version=2, levels=1, lines=1, columns=3, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc_short_stream.spk"
    file_path.write_bytes(header + poslen + bytes([0x53]))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_raises_parse_error_on_unrecognized_file(tmp_path):
    file_path = tmp_path / "not_a_spk_file.spk"
    file_path.write_bytes(b"this is not a valid tv/Mfile spk file at all" + b"\x00" * 40)

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_parses_real_pg_25um_f_spk():
    # pg_25um_f.spk is a real tv/Mfile oldmat/MAT_LF4 file: 16384 raw
    # little-endian float32 channel values, no header, identified by a
    # 64-byte "\nMatFmt: 16k.lf4:2\n" trailer at EOF -- verified during
    # design against the libmfile-1.0.7 source and the file's own bytes.
    data = load_spk(str(FIXTURES / "pg_25um_f.spk"))
    assert len(data) == 16384

    with open(FIXTURES / "pg_25um_f.spk", "rb") as f:
        raw = f.read()
    expected_first5 = struct.unpack_from("<5f", raw, 0)
    assert list(data[:5]) == [round(v) for v in expected_first5]
    expected_last = struct.unpack_from("<f", raw, 4 * 16383)[0]
    assert int(data[16383]) == round(expected_last)


def _oldmat_bytes(fmt_string, payload):
    trailer = b"\nMatFmt: " + fmt_string.encode("ascii") + b"\n"
    trailer = trailer.ljust(64, b"\x00")
    return payload + trailer


def test_oldmat_le4(tmp_path):
    values = [10, -5, 2000]
    payload = struct.pack("<3i", *values)
    file_path = tmp_path / "oldmat_le4.spk"
    file_path.write_bytes(_oldmat_bytes("3.le4", payload))

    data = load_spk(str(file_path))

    assert list(data) == values


def test_oldmat_lf4(tmp_path):
    values = [1.4, 2.6, -3.2]
    payload = struct.pack("<3f", *values)
    file_path = tmp_path / "oldmat_lf4.spk"
    file_path.write_bytes(_oldmat_bytes("3.lf4:2", payload))

    data = load_spk(str(file_path))

    assert list(data) == [round(v) for v in values]


def test_oldmat_he2(tmp_path):
    values = [1, 60000, 3]
    payload = struct.pack(">3H", *values)
    file_path = tmp_path / "oldmat_he2.spk"
    file_path.write_bytes(_oldmat_bytes("3.he2", payload))

    data = load_spk(str(file_path))

    assert list(data) == values


def test_oldmat_rejects_unsupported_fmtname(tmp_path):
    payload = struct.pack("<3i", 1, 2, 3)
    file_path = tmp_path / "oldmat_bad_fmt.spk"
    file_path.write_bytes(_oldmat_bytes("3.xyz", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_2d_matrix(tmp_path):
    # "2.3.le4" parses as lines=2, columns=3 (levels defaults to 1) --
    # lines != 1 must still be rejected as a 2-D matrix.
    payload = struct.pack("<6i", 1, 2, 3, 4, 5, 6)
    file_path = tmp_path / "oldmat_2d.spk"
    file_path.write_bytes(_oldmat_bytes("2.3.le4", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_size_mismatch(tmp_path):
    payload = struct.pack("<3i", 1, 2, 3)
    file_path = tmp_path / "oldmat_mismatch.spk"
    # declares 5 channels but the payload only has 3
    file_path.write_bytes(_oldmat_bytes("5.le4", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_malformed_trailer(tmp_path):
    payload = struct.pack("<3i", 1, 2, 3)
    # magic prefix present but no terminating newline after the fmt string
    trailer = (b"\nMatFmt: 3.le4" + b"\x00" * 50)[:64]
    file_path = tmp_path / "oldmat_malformed.spk"
    file_path.write_bytes(payload + trailer)

    with pytest.raises(ParseError):
        load_spk(str(file_path))
