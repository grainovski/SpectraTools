"""Tests for the `.sou` source-description reader and the line matcher.

The format is four whitespace-separated numbers per line -- energy, its
error, intensity, its error -- with no header. Every sample file in the
working directory is structurally identical in this respect, so unlike
.lzs there is no real-file trap to pin and synthetic content covers the
format completely. Deliberately no hard-coded file or line count here:
the working directory gains sources over time (am241.sou appeared after
this reader was written), and a count in a docstring only rots.
"""

import math

import pytest

from histogram_io import ParseError
from sou_io import SourceLine, load_sou, match_line


def _write(tmp_path, text, name="src.sou"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- reading -----------------------------------------------------------


def test_reads_the_real_column_layout(tmp_path):
    """Exactly how ba133.sou is laid out: right-aligned, leading-dot
    fractions, trailing-dot integers."""
    path = _write(tmp_path, (
        "         53.1560       .0050        348.          7.\n"
        "        356.0140       .0090      10000.         30.\n"
    ))
    lines = load_sou(path)
    assert lines == [
        SourceLine(53.156, 0.005, 348.0, 7.0),
        SourceLine(356.014, 0.009, 10000.0, 30.0),
    ]


def test_file_order_is_preserved(tmp_path):
    """The reader reports what the file says, it does not sort."""
    path = _write(tmp_path, "200 .1 1 1\n100 .1 1 1\n")
    assert [line.energy for line in load_sou(path)] == [200.0, 100.0]


def test_blank_lines_are_skipped(tmp_path):
    path = _write(tmp_path, "\n100 .1 1 1\n\n   \n200 .1 1 1\n\n")
    assert [line.energy for line in load_sou(path)] == [100.0, 200.0]


def test_empty_file_is_refused(tmp_path):
    with pytest.raises(ParseError, match="no source lines"):
        load_sou(_write(tmp_path, ""))


def test_blank_only_file_is_refused(tmp_path):
    with pytest.raises(ParseError, match="no source lines"):
        load_sou(_write(tmp_path, "\n   \n"))


def test_wrong_column_count_names_the_line(tmp_path):
    path = _write(tmp_path, "100 .1 1 1\n200 .1 1\n")
    with pytest.raises(ParseError, match="line 2.*4 columns.*found 3"):
        load_sou(path)


def test_non_numeric_token_names_the_line(tmp_path):
    path = _write(tmp_path, "100 .1 1 1\nabc .1 1 1\n")
    with pytest.raises(ParseError, match="line 2.*not a number"):
        load_sou(path)


def test_non_finite_energy_is_refused(tmp_path):
    """float() accepts 'nan' and 'inf'; a matcher fed either would
    compare garbage silently."""
    path = _write(tmp_path, "nan .1 1 1\n")
    with pytest.raises(ParseError, match="line 1.*finite"):
        load_sou(path)


def test_non_positive_energy_is_refused(tmp_path):
    """A gamma line at 0 or negative keV is not a line."""
    path = _write(tmp_path, "0 .1 1 1\n")
    with pytest.raises(ParseError, match="line 1.*positive"):
        load_sou(path)


def test_non_utf8_bytes_are_refused(tmp_path):
    path = tmp_path / "bad.sou"
    path.write_bytes(b"100 .1 1 1\n\xff\xfe 2 3 4\n")
    with pytest.raises(ParseError, match="not valid UTF-8"):
        load_sou(str(path))


def test_missing_file_is_a_parse_error(tmp_path):
    with pytest.raises(ParseError):
        load_sou(str(tmp_path / "missing.sou"))


# --- matching ----------------------------------------------------------


def _lines(*energies):
    return [SourceLine(e, 0.01, 1000.0, 10.0) for e in energies]


def test_nearest_line_is_suggested_when_unambiguous():
    lines = _lines(356.014, 383.859)
    assert match_line(350.0, lines) == lines[0]


def test_declines_near_the_midpoint():
    """Half-way between two lines the nearest is not CLEARLY nearest."""
    lines = _lines(356.0, 384.0)
    assert match_line(370.0, lines) is None
    # Just off the midpoint is still ambiguous under the half-distance rule.
    assert match_line(366.0, lines) is None


def test_the_half_distance_rule_exactly():
    """Suggest only when the nearest is closer than half the distance to
    the runner-up. Boundary: d1 == d2 / 2 is NOT good enough."""
    lines = _lines(100.0, 130.0)
    # 110 -> d1=10, d2=20: 10 < 10 is False -> decline.
    assert match_line(110.0, lines) is None
    # 109 -> d1=9, d2=21: 9 < 10.5 -> suggest.
    assert match_line(109.0, lines) == lines[0]


def test_a_single_line_is_always_unambiguous():
    lines = _lines(1836.062)
    assert match_line(200.0, lines) == lines[0]


def test_duplicate_energies_are_ambiguous():
    lines = _lines(511.0, 511.0)
    assert match_line(511.0, lines) is None


def test_exact_hit_beats_any_runner_up():
    lines = _lines(356.014, 356.1)
    assert match_line(356.014, lines) == lines[0]


def test_no_lines_gives_no_match():
    assert match_line(100.0, []) is None


def test_non_finite_energy_never_matches():
    assert match_line(math.nan, _lines(100.0)) is None
    assert match_line(math.inf, _lines(100.0)) is None
