"""Tests for the CalEnEff seven-column export.

Format (from CalEnEff's own ra226_gui.py:18, "ABSOLUTE uncertainties"):
    ch  delta_ch  N  delta_N  E[keV]  I[%]  delta_I[%]
"""

import numpy as np
import pytest

from caleneff_export import ExportError, build_rows, write_caleneff
from sou_io import SourceLine


def _lines():
    # Strongest is 10000 at 344.276, so the scale factor is 100/10000.
    return [
        SourceLine(121.783, 0.002, 5000.0, 50.0),
        SourceLine(344.276, 0.004, 10000.0, 80.0),
        SourceLine(1408.011, 0.010, 2000.0, 30.0),
    ]


def test_intensity_is_normalised_so_the_strongest_line_is_100():
    rows, skipped = build_rows(
        [(100.0, 0.01, 5000.0, 70.0, 344.276)], _lines()
    )
    assert skipped == 0
    assert rows[0].intensity_pct == pytest.approx(100.0)
    assert rows[0].intensity_pct_err == pytest.approx(0.8)


def test_other_lines_scale_by_the_same_factor():
    rows, _ = build_rows(
        [(50.0, 0.01, 900.0, 30.0, 121.783),
         (400.0, 0.02, 300.0, 20.0, 1408.011)],
        _lines(),
    )
    assert rows[0].intensity_pct == pytest.approx(50.0)
    assert rows[0].intensity_pct_err == pytest.approx(0.5)
    assert rows[1].intensity_pct == pytest.approx(20.0)
    assert rows[1].intensity_pct_err == pytest.approx(0.3)


def test_fit_quantities_pass_through_unchanged():
    rows, _ = build_rows(
        [(352.72172, 0.00014, 15787.6, 12.3, 344.276)], _lines()
    )
    row = rows[0]
    assert row.channel == pytest.approx(352.72172)
    assert row.channel_err == pytest.approx(0.00014)
    assert row.area == pytest.approx(15787.6)
    assert row.area_err == pytest.approx(12.3)
    assert row.energy == pytest.approx(344.276)


def test_an_energy_with_no_matching_source_line_is_skipped_and_counted():
    """A hand-typed energy has no intensity, so it cannot be exported."""
    rows, skipped = build_rows(
        [(100.0, 0.01, 900.0, 30.0, 344.276),
         (200.0, 0.01, 500.0, 20.0, 999.999)],
        _lines(),
    )
    assert len(rows) == 1
    assert skipped == 1
    assert rows[0].energy == pytest.approx(344.276)


def test_no_source_lines_skips_everything():
    rows, skipped = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], [])
    assert rows == []
    assert skipped == 1


def test_matching_tolerates_float_representation():
    """The energy stored on an assignment came from the same file, so it
    must match even after a round-trip through text.

    The offset is deliberately non-zero and inside the tolerance. The
    previous version passed float("344.276"), which is bit-identical to
    344.276, so it would have passed at a tolerance of zero and proved
    nothing about the slack its own docstring claims to exercise.
    """
    inside = 344.276 + 5e-7
    assert inside != 344.276, "the offset must survive float rounding"
    rows, skipped = build_rows([(100.0, 0.01, 900.0, 30.0, inside)], _lines())
    assert skipped == 0
    assert len(rows) == 1


def test_an_energy_beyond_the_tolerance_does_not_match():
    """Control for the above: without this, a tolerance wide enough to
    match anything would satisfy that test just as well."""
    outside = 344.276 + 1e-4
    rows, skipped = build_rows([(100.0, 0.01, 900.0, 30.0, outside)], _lines())
    assert rows == []
    assert skipped == 1


def test_written_file_reads_back_with_loadtxt(tmp_path):
    """CalEnEff parses with np.loadtxt, so that is what the test uses."""
    rows, _ = build_rows(
        [(352.72172, 0.00014, 15787.6, 12.3, 344.276),
         (500.0, 0.002, 4000.0, 9.0, 121.783)],
        _lines(),
    )
    path = tmp_path / "out.txt"
    write_caleneff(str(path), rows)

    data = np.loadtxt(str(path), ndmin=2)
    assert data.shape == (2, 7)
    assert data[0, 0] == pytest.approx(352.72172)
    assert data[0, 1] == pytest.approx(0.00014)
    assert data[0, 2] == pytest.approx(15787.6)
    assert data[0, 3] == pytest.approx(12.3)
    assert data[0, 4] == pytest.approx(344.276)
    assert data[0, 5] == pytest.approx(100.0)
    assert data[0, 6] == pytest.approx(0.8)


def test_written_file_has_no_header(tmp_path):
    rows, _ = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], _lines())
    path = tmp_path / "out.txt"
    write_caleneff(str(path), rows)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert not first.lstrip().startswith("#")
    assert len(first.split()) == 7


def test_writing_no_rows_is_refused(tmp_path):
    """Better than an empty file CalEnEff would fail to load."""
    path = tmp_path / "out.txt"
    with pytest.raises(ExportError, match="no rows"):
        write_caleneff(str(path), [])
    assert not path.exists()


def test_unwritable_path_raises_export_error(tmp_path):
    rows, _ = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], _lines())
    with pytest.raises(ExportError):
        write_caleneff(str(tmp_path / "nope" / "out.txt"), rows)


def test_zero_max_intensity_is_refused():
    """Every line at zero intensity gives no scale to normalise by."""
    lines = [SourceLine(100.0, 0.1, 0.0, 0.0)]
    with pytest.raises(ExportError, match="intensity"):
        build_rows([(10.0, 0.1, 500.0, 20.0, 100.0)], lines)


# --- the rule CalEnEff enforces, now enforced here too -----------------


def test_a_row_with_no_uncertainty_at_all_is_skipped():
    """CalEnEff refuses rows where dN and dI are both zero: eff = N / I_pct
    would carry no uncertainty, so the point silently anchors the whole
    efficiency curve. build_rows cited that rule without applying it."""
    lines = [SourceLine(344.276, 0.004, 10000.0, 0.0)]
    rows, skipped = build_rows([(100.0, 0.01, 900.0, 0.0, 344.276)], lines)
    assert rows == []
    assert skipped == 1


def test_a_row_keeps_its_place_when_only_one_uncertainty_is_zero():
    """Control: the rule is about BOTH being zero. A zero area error with a
    real intensity error still yields a usable efficiency uncertainty."""
    lines = [SourceLine(344.276, 0.004, 10000.0, 80.0)]
    rows, skipped = build_rows([(100.0, 0.01, 900.0, 0.0, 344.276)], lines)
    assert len(rows) == 1
    assert skipped == 0


def test_a_non_finite_value_is_refused_rather_than_written_as_nan(tmp_path):
    """A NaN area formatted straight into the file as the text "nan", which
    np.loadtxt reads back as a float and CalEnEff then computes with."""
    rows, skipped = build_rows(
        [(100.0, 0.01, float("nan"), 30.0, 344.276)], _lines()
    )
    assert len(rows) == 1 and skipped == 0        # build_rows still makes it
    with pytest.raises(ExportError) as excinfo:
        write_caleneff(str(tmp_path / "out.txt"), rows)
    assert "non-finite" in str(excinfo.value)
    assert "area" in str(excinfo.value)
    assert not (tmp_path / "out.txt").exists()


def test_an_ordinary_row_still_writes(tmp_path):
    """Control for the guard above: it must not refuse healthy rows."""
    rows, _ = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], _lines())
    path = tmp_path / "out.txt"
    write_caleneff(str(path), rows)
    assert "nan" not in path.read_text()
    assert len(path.read_text().strip().splitlines()) == 1
