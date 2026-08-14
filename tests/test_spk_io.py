import struct
from pathlib import Path

import numpy as np
import pytest

from histogram_io import ParseError
from spk_io import (
    LC_HEADER_SIZE, LC_MAGIC, MAT_COLMAX, _lc2_compress, _lc2_uncompress, _put_tag_n,
    _zigzag_decode, _zigzag_encode, load_spk, save_spk,
)

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


def test_lc_rejects_value_overflowing_int64(tmp_path):
    # LC1's extended-tag continuation loop has no cap on the number of
    # continuation bytes, so a handful of crafted bytes can build a
    # decoded value whose magnitude vastly exceeds int64 range. This must
    # raise ParseError (via the try/except OverflowError in _load_lc),
    # not crash with an unhandled OverflowError.
    header = _lc_header(version=1, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 11)
    # tag byte 0xFF (extended, initial 6 bits = 0x3F = 63), followed by
    # 9 continuation bytes of 0xFF (each contributes 127 << s, s growing
    # by 7 each time -- by the 9th byte alone the running total is far
    # past 2**63) and a final stop byte (0x00, bit7 clear).
    compressed = bytes([0xFF] + [0xFF] * 9 + [0x00])
    file_path = tmp_path / "lc_overflow.spk"
    file_path.write_bytes(header + poslen + compressed)

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


def test_oldmat_rejects_2d_matrix_via_levels(tmp_path):
    # "2.1.3.le4" (three numbers) parses as levels=2, lines=1, columns=3 --
    # levels != 1 must be rejected too, not just lines != 1.
    payload = struct.pack("<6i", 1, 2, 3, 4, 5, 6)
    file_path = tmp_path / "oldmat_2d_levels.spk"
    file_path.write_bytes(_oldmat_bytes("2.1.3.le4", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_too_many_dimensions(tmp_path):
    # A 4th dot-separated leading number is not a valid MatFmt grammar --
    # the format only allows up to levels.lines.columns (3 numbers).
    payload = struct.pack("<3i", 1, 2, 3)
    file_path = tmp_path / "oldmat_too_many_dims.spk"
    file_path.write_bytes(_oldmat_bytes("1.1.1.3.le4", payload))

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


def test_load_spk_raises_parse_error_not_silent_corruption_on_extreme_oldmat_float(tmp_path):
    # oldmat's floating-point (lf4/hf4) path does
    # np.round(channels).astype(np.int64) with no range check, same bug
    # class as test_lc_rejects_value_overflowing_int64 above but on the
    # oldmat path instead of LC. .astype() doesn't raise on overflow --
    # it silently casts to the int64 sentinel (-9223372036854775808)
    # with only an invisible RuntimeWarning. Must raise ParseError
    # instead. 1e30 is well within float32 range (max ~3.4e38) so it
    # survives the file round-trip unchanged, but is far outside int64
    # range (~9.2e18).
    payload = struct.pack("<1f", 1e30)
    file_path = tmp_path / "oldmat_extreme.spk"
    file_path.write_bytes(_oldmat_bytes("1.lf4:2", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_zigzag_encode_matches_decode_for_various_values():
    for value in [-1000, -3, -2, -1, 0, 1, 2, 3, 1000, 158]:
        assert _zigzag_decode(_zigzag_encode(value)) == value


def test_zigzag_encode_known_values():
    assert _zigzag_encode(0) == 0
    assert _zigzag_encode(-1) == 1
    assert _zigzag_encode(1) == 2
    assert _zigzag_encode(-2) == 3
    assert _zigzag_encode(158) == 316


def test_put_tag_n_single_byte_for_small_values():
    assert _put_tag_n(0x80, 5) == bytes([0x85])
    assert _put_tag_n(0x80, 0) == bytes([0x80])
    assert _put_tag_n(0x80, 59) == bytes([0xBB])


def test_put_tag_n_extended_encoding_matches_known_fixture():
    # Matches test_lc2_extended_single_value_tag above: the zigzag code
    # for 158 is 316, which must encode as tag 0xBD + [0x00, 0x00].
    assert _put_tag_n(0x80, 316) == bytes([0xBD, 0x00, 0x00])


def test_put_tag_n_same_diff_base():
    assert _put_tag_n(0xC0, 7) == bytes([0xC7])


def test_lc2_compress_3value_tag():
    assert _lc2_compress([0, -1, 1]) == bytes([0x24])


def test_lc2_compress_2value_tag():
    assert _lc2_compress([-2, 1]) == bytes([0x53])


def test_lc2_compress_1value_tag():
    assert _lc2_compress([-3]) == bytes([0x85])


def test_lc2_compress_same_run_tag():
    assert _lc2_compress([1, 0, 0, 0, 0, 0, 0]) == bytes([0xC7])


def test_lc2_compress_extended_single_value_tag():
    assert _lc2_compress([158]) == bytes([0xBD, 0x00, 0x00])


def test_lc2_compress_empty_input():
    assert _lc2_compress([]) == b""


def test_lc2_compress_round_trips_through_uncompress_for_varied_data():
    samples = [
        [0] * 20,
        list(range(50)),
        [5, 5, 5, 5, 5, 5, 5, 5, -100, 200, 0, 0, 0],
        [1000, -1000, 500, -500, 0, 0, 0, 0, 0, 1],
        [7],
    ]
    for values in samples:
        compressed = _lc2_compress(values)
        decoded = _lc2_uncompress(compressed, len(values), "test")
        assert decoded == values


def test_lc2_compress_recompresses_demo_spk_to_the_identical_bytes():
    # The strongest possible check: decode demo.spk's real compressed
    # payload, recompress the decoded values, and confirm the output
    # matches the original file's bytes exactly -- not just that it
    # round-trips through our own decoder.
    with open(FIXTURES / "demo.spk", "rb") as f:
        raw = f.read()
    fields = struct.unpack_from("<11I", raw, 0)
    _magic, _version, _levels, _lines, columns, poslentablepos = fields[:6]
    pos, length = struct.unpack_from("<2I", raw, poslentablepos)
    original_compressed = raw[pos:pos + length]

    decoded = _lc2_uncompress(original_compressed, columns, "demo.spk")
    recompressed = _lc2_compress(decoded)

    assert recompressed == original_compressed


def test_put_tag_n_succeeds_at_max_valid_value():
    # LC2's tag byte is tag_base + 60 + extra, and the format only allows
    # extra in [0,3] (1-4 extension bytes -- matching _lc2_uncompress's
    # own "extra_bytes = v - 59" for v in [60,63]). The largest value
    # representable with the maximum 4 extension bytes (all 0xFF) is
    # 59 + sum((0xFF + 1) << (8*i) for i in range(4)) == 4311810363,
    # independently confirmed by binary search against this function and
    # cross-checked against _lc2_uncompress's own reconstruction formula.
    assert _put_tag_n(0x80, 4311810363) == bytes([0xBF, 0xFF, 0xFF, 0xFF, 0xFF])


def test_put_tag_n_rejects_value_one_past_max_valid():
    # One more than the boundary above -- would require a 5th extension
    # byte, which the LC2 tag format has no room to represent.
    with pytest.raises(ValueError):
        _put_tag_n(0x80, 4311810364)


def test_lc2_compress_raises_value_error_instead_of_corrupting_or_crashing():
    # Regression coverage for a real failure class: Python ints don't
    # wrap like the C reference's 32-bit int, so nothing previously
    # bounded how large a delta could be. Oversized deltas used to
    # either silently corrupt (wrong tag byte, still "successfully"
    # decoded to different values) or crash deep inside _put_tag_n with
    # an unhelpful "bytes must be in range(0, 256)" ValueError. They must
    # now raise a clear, purposeful ValueError instead.
    with pytest.raises(ValueError):
        _lc2_compress([2**40])
    with pytest.raises(ValueError):
        _lc2_compress([2**600])


def test_lc2_compress_same_run_boundary_exactly_4_uses_run_tag():
    assert _lc2_compress([0, 0, 0, 0]) == bytes([0xC0])


def test_lc2_compress_same_run_boundary_exactly_3_falls_through_to_pack():
    assert _lc2_compress([0, 0, 0]) == bytes([0x00])


def test_save_spk_round_trips_through_load_spk(tmp_path):
    data = np.array([0, 0, 1, 5, 100, 0, 41580, 2])
    path = tmp_path / "out.spk"

    save_spk(str(path), data)
    result = load_spk(str(path))

    assert list(result) == list(data)


def test_save_spk_round_trips_the_real_demo_spk_data(tmp_path):
    original = load_spk(str(FIXTURES / "demo.spk"))
    path = tmp_path / "roundtrip.spk"

    save_spk(str(path), original)
    result = load_spk(str(path))

    assert list(result) == list(original)


def test_save_spk_writes_version_2_header(tmp_path):
    data = np.array([1, 2, 3])
    path = tmp_path / "out.spk"

    save_spk(str(path), data)

    with open(path, "rb") as f:
        raw = f.read()
    fields = struct.unpack_from("<11I", raw, 0)
    magic, version, levels, lines, columns, poslentablepos = fields[:6]
    assert magic == LC_MAGIC
    assert version == 2
    assert levels == 1
    assert lines == 1
    assert columns == 3
    assert poslentablepos == LC_HEADER_SIZE


def test_save_spk_round_trips_all_zero_spectrum(tmp_path):
    data = np.zeros(20, dtype=np.int64)
    path = tmp_path / "out.spk"

    save_spk(str(path), data)
    result = load_spk(str(path))

    assert list(result) == [0] * 20


def test_save_spk_round_trips_negative_values(tmp_path):
    # Real spectra are non-negative, but the codec itself is signed --
    # this exercises that the round-trip holds regardless.
    data = np.array([0, 1000000, 41580, -5])
    path = tmp_path / "out.spk"

    save_spk(str(path), data)
    result = load_spk(str(path))

    assert list(result) == list(data)


def test_save_spk_rejects_a_channel_count_too_large_for_the_format():
    with pytest.raises(ValueError):
        save_spk("unused.spk", [0] * (MAT_COLMAX + 1))
