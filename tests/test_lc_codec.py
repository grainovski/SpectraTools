import pytest

from lc_codec import zigzag_decode, decode_row, DIM_MAX
from histogram_io import ParseError


def test_zigzag_decode_matches_known_values():
    assert zigzag_decode(0) == 0
    assert zigzag_decode(1) == -1
    assert zigzag_decode(2) == 1
    assert zigzag_decode(3) == -2


def test_decode_row_raises_parse_error_on_truncated_data():
    with pytest.raises(ParseError):
        decode_row(b"", 5, "fake_path", kind="lc matrix file")


def test_dim_max_is_65536():
    assert DIM_MAX == 1 << 16


def test_decode_row_requires_kind_as_a_keyword_argument():
    # kind has no default and is keyword-only -- a caller can't
    # silently inherit another format's error wording by omission,
    # which is exactly how spk_io.py ended up describing a corrupt
    # .spk file as a "matrix file" the first time this was extracted.
    with pytest.raises(TypeError):
        decode_row(b"", 5, "fake_path")


def test_decode_row_error_message_uses_the_given_kind():
    with pytest.raises(ParseError, match=r"^widget: "):
        decode_row(b"", 5, "fake_path", kind="widget")


def test_decode_row_returns_a_plain_list():
    # Both callers (mtx_io.py's `data[row, :] = values`, and
    # spk_io.py's _lc2_uncompress whose result feeds both
    # np.array(values, ...) and direct `== a_list` equality checks in
    # tests) depend on this exact return type. Locking it in here:
    # decode_row's performance optimization must never change it to,
    # say, a numpy array, without every caller being updated in step.
    result = decode_row(bytes([0x8A, 0x8D, 0x84]), 3, "fake_path", kind="lc matrix file")
    assert type(result) is list
    assert all(type(v) is int for v in result)
    assert result == [5, -2, 0]


def test_decode_row_single_value_tag_at_zigzag_table_boundary():
    # decode_row's performance optimization looks up small deltas in a
    # precomputed 64-entry zigzag table and falls back to calling
    # zigzag_decode() directly above that -- this pins down the exact
    # boundary (n=63 is the table's last entry, n=64 is the first
    # value that must fall back) so a future off-by-one in that split
    # can't silently corrupt just the values right at the edge.
    # Tag byte 0xBC: hi-bit set, bit6 clear (single-value tag), low 6
    # bits = 60 (>59) -> needs 1 extension byte; resolved n = 59 +
    # (extension_byte + 1).
    in_table = decode_row(bytes([0xBC, 0x03]), 1, "fake_path", kind="lc matrix file")  # n=63
    just_past = decode_row(bytes([0xBC, 0x04]), 1, "fake_path", kind="lc matrix file")  # n=64
    assert in_table == [zigzag_decode(63)] == [-32]
    assert just_past == [zigzag_decode(64)] == [32]


# --- encoder (moved here from spk_io in 7.0.0) ----------------------------


def _naive_lc2_compress(values, wrap32=False):
    """Frozen element-by-element port of lc2_compress (lc_c2.c:63-126),
    the shape spk_io._lc2_compress had before 7.0.0: no run lookup. The
    reference encode_row's bisect shortcut must agree with byte for byte."""
    from lc_codec import put_tag_n, wrap_int32, zigzag_encode

    def delta(v, last):
        d = v - last
        return wrap_int32(d) if wrap32 else d

    out = bytearray()
    last = 0
    n = len(values)
    idx = 0
    while idx < n:
        remaining = n - idx
        d = delta(values[idx], last)
        i = 1
        if 0 <= d < 2:
            while i < remaining and values[idx + i] == last:
                i += 1
        same = i - 1
        if same >= 3:
            out += put_tag_n(0xC0, ((same - 3) << 1) + d)
            idx += i
            continue
        s0 = values[idx]
        a = zigzag_encode(d)
        if (a >> 3) == 0 and remaining >= 2:
            s1 = values[idx + 1]
            b = zigzag_encode(delta(s1, last))
            if ((a | b) >> 2) == 0 and remaining >= 3:
                s2 = values[idx + 2]
                c = zigzag_encode(delta(s2, last))
                if (c >> 2) == 0:
                    out.append(a + (b << 2) + (c << 4))
                    idx += 3
                    last = s2
                    continue
            if (b >> 3) == 0:
                out.append(0x40 + a + (b << 3))
                idx += 2
                last = s1
                continue
        out += put_tag_n(0x80, a)
        idx += 1
        last = s0
    return bytes(out)


def test_encode_row_matches_the_naive_encoder_on_a_randomised_sweep():
    """Rows built to exercise the long-run lookup: runs of every length
    around _RUN_SCAN_LIMIT, of 0, of 1 and of other values, mixed with
    singles, small and large deltas and int32 extremes."""
    import numpy as np
    from lc_codec import INT32_MAX, INT32_MIN, encode_row

    rng = np.random.default_rng(20261127)
    pool = [0, 0, 1, -1, 2, 5, 300, -70000, INT32_MIN, INT32_MAX]
    checked = 0
    for _ in range(400):
        row = []
        while len(row) < int(rng.integers(1, 400)):
            v = pool[int(rng.integers(len(pool)))]
            row += [v] * int(rng.choice([1, 2, 3, 4, 15, 16, 17, 18, 40, 200]))
        arr = np.array(row, dtype=np.int64)
        expected = _naive_lc2_compress(row, wrap32=True)
        assert encode_row(row, wrap32=True) == expected
        assert encode_row(arr, wrap32=True) == expected
        assert decode_row(expected, len(row), "sweep", kind="test") is not None
        checked += 1
    assert checked == 400


def test_encode_row_wraps_int32_deltas_like_the_c():
    """INT32_MAX followed by INT32_MIN: C computes the delta in int, which
    wraps to +1 (lc_c2.c:85), so the pair costs two bytes, not ten."""
    from lc_codec import INT32_MAX, INT32_MIN, encode_row, wrap_int32

    assert wrap_int32(INT32_MIN - INT32_MAX) == 1
    wrapped = encode_row([INT32_MAX, INT32_MIN], wrap32=True)
    assert wrapped == bytes([0xBF, 0xC2, 0xFE, 0xFE, 0xFE, 0x82])
    # Unwrapped (save_spk's behaviour) the same delta is out of range.
    with pytest.raises(ValueError):
        encode_row([INT32_MAX, INT32_MIN])
