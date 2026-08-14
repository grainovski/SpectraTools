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
