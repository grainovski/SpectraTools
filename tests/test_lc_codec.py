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
