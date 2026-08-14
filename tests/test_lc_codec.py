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
        decode_row(b"", 5, "fake_path")


def test_dim_max_is_65536():
    assert DIM_MAX == 1 << 16
