"""X8: CSV and LaTeX exports, and X6: reloading a spectrum from disk."""

import math
import os

import numpy as np
import pytest

import fit_export
from calibration import Calibration
from fit_mode import _export_writer_for
from peak_fit import FitResult, IntegrationResult, PeakResult


def _peak(position=100.0, area=1234.5, area_err=45.6, fwhm=5.0):
    return PeakResult(
        position=position, position_err=0.12,
        fwhm=fwhm, fwhm_err=0.25,
        area=area, area_err=area_err,
        amplitude=200.0, sigma=fwhm / 2.3548,
        full_area=area + 300.0, full_area_err=area_err + 5.0,
    )


def _fit(peaks=None):
    return FitResult(
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
        peaks=peaks if peaks is not None else [_peak()],
        reduced_chi2=1.07,
    )


# --- which writer gets picked ------------------------------------------


def test_the_extension_decides_the_format_not_the_filter():
    """A user who types "results.csv" while the Text filter happens to be
    selected asked for CSV, not for a text file wearing a .csv name."""
    assert _export_writer_for("x.csv", "Text files (*.txt)") is fit_export.write_csv
    assert _export_writer_for("x.tex", "Text files (*.txt)") is fit_export.write_latex
    assert _export_writer_for("x.txt", "CSV files (*.csv)") is fit_export.write_text_report


def test_the_filter_breaks_the_tie_when_there_is_no_extension():
    assert _export_writer_for("results", "CSV files (*.csv)") is fit_export.write_csv
    assert _export_writer_for("results", "LaTeX table (*.tex)") is fit_export.write_latex
    assert _export_writer_for("results", "All files (*)") is fit_export.write_text_report


# --- CSV ---------------------------------------------------------------


def test_csv_writes_one_row_per_peak(tmp_path):
    path = tmp_path / "out.csv"
    result = _fit([_peak(position=100.0), _peak(position=140.0)])
    fit_export.write_csv(path, [(1, result)], "spec.txt")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# spectrum:")
    assert lines[1].split(",")[:4] == ["fit", "peak", "position", "position_err"]
    assert len(lines) == 4  # comment + header + two peaks
    assert lines[2].split(",")[:3] == ["1", "1", "100"]
    assert lines[3].split(",")[:3] == ["1", "2", "140"]


def test_csv_leaves_an_undetermined_uncertainty_empty(tmp_path):
    """Not "nan" or "n/a": a spreadsheet reads an empty cell as missing but
    a text cell as text, which turns the whole column into strings and
    breaks any average taken over it."""
    path = tmp_path / "out.csv"
    fit_export.write_csv(path, [(1, _fit([_peak(area_err=float("nan"))]))], "s.txt")

    row = path.read_text(encoding="utf-8").splitlines()[2].split(",")
    header = path.read_text(encoding="utf-8").splitlines()[1].split(",")
    assert row[header.index("area_err")] == ""
    assert "nan" not in path.read_text(encoding="utf-8").lower()


def test_csv_converts_positions_to_kev_when_calibrated(tmp_path):
    path = tmp_path / "cal.csv"
    calibration = Calibration(kind="linear", a=10.0, b=2.0)
    fit_export.write_csv(path, [(1, _fit([_peak(position=100.0)]))], "s.txt", calibration)

    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[1].split(",")
    row = text.splitlines()[2].split(",")
    # Headers say which unit, so a reader cannot mistake one for the other.
    assert "position_keV" in header
    assert "fwhm_keV" in header
    assert float(row[header.index("position_keV")]) == pytest.approx(210.0)
    # FWHM scales by the local slope: 5 channels * 2 keV/channel.
    assert float(row[header.index("fwhm_keV")]) == pytest.approx(10.0)
    # Area is a count and has no energy equivalent -- it must NOT be scaled.
    assert float(row[header.index("area")]) == pytest.approx(1234.5)


def test_integration_results_are_skipped_by_the_tabular_writers(tmp_path):
    """An integration has no peaks; its gross/background/net breakdown is a
    different shape and forcing it into these columns would produce rows
    whose numbers mean something other than the header says."""
    # A real IntegrationResult rather than a stand-in, so this keeps
    # testing the actual shape if that class grows fields.
    from peak_fit import integrate_region

    x = np.arange(40, dtype=float)
    y = np.full(40, 12.0)
    y[18:23] += [10, 40, 90, 35, 8]
    integration = integrate_region(x, y, (2.0, 8.0), (32.0, 38.0), (14.0, 26.0))

    path = tmp_path / "mixed.csv"
    fit_export.write_csv(path, [(1, integration), (2, _fit())], "s.txt")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # comment + header + the ONE real fit
    assert lines[2].startswith("2,1,")


# --- LaTeX -------------------------------------------------------------


def test_latex_emits_a_bare_table_not_a_whole_document(tmp_path):
    """It has to drop into an existing paper, so a preamble would have to
    be stripped before use."""
    path = tmp_path / "out.tex"
    fit_export.write_latex(path, [(1, _fit())], "spec.txt")
    text = path.read_text(encoding="utf-8")

    assert "\\begin{table}" in text
    assert "\\begin{tabular}" in text
    assert "\\documentclass" not in text
    assert "\\begin{document}" not in text


def test_latex_escapes_characters_that_are_syntax(tmp_path):
    """A spectrum path is very likely to contain underscores, which are
    subscript operators in LaTeX and break the build."""
    path = tmp_path / "esc.tex"
    fit_export.write_latex(path, [(1, _fit())], "my_run_01&2.txt")
    text = path.read_text(encoding="utf-8")

    assert r"my\_run\_01\&2.txt" in text
    # The bare form must not survive anywhere in the comment line.
    comment = next(line for line in text.splitlines() if line.startswith("%"))
    assert "my_run" not in comment


def test_latex_escape_handles_backslashes_and_braces_together():
    """The old sequential-replace version escaped the backslash FIRST,
    then the brace pass mangled its own insertion into
    \\textbackslash\\{\\} -- the classic ordering bug, latent because no
    current caller feeds it a backslash. A single pass never rescans its
    own output. A Windows path is exactly where a backslash would come
    from."""
    from fit_export import _latex_escape

    assert _latex_escape("a\\b") == r"a\textbackslash{}b"
    assert _latex_escape("{x}") == r"\{x\}"
    assert _latex_escape("C:\\runs\\{gg}_1.mtx") == (
        r"C:\textbackslash{}runs\textbackslash{}\{gg\}\_1.mtx"
    )


def test_latex_omits_an_undetermined_uncertainty_rather_than_printing_nan(tmp_path):
    path = tmp_path / "nan.tex"
    fit_export.write_latex(path, [(1, _fit([_peak(area_err=float("nan"))]))], "s.txt")
    text = path.read_text(encoding="utf-8")

    assert "nan" not in text.lower()
    # The value itself is still there, just without a +/- term.
    assert "1234" in text


def test_latex_labels_the_unit_it_is_reporting(tmp_path):
    plain = tmp_path / "ch.tex"
    fit_export.write_latex(plain, [(1, _fit())], "s.txt")
    assert "Position (ch)" in plain.read_text(encoding="utf-8")

    calibrated = tmp_path / "kev.tex"
    fit_export.write_latex(
        calibrated, [(1, _fit())], "s.txt", Calibration(kind="linear", a=0.0, b=1.0)
    )
    assert "Position (keV)" in calibrated.read_text(encoding="utf-8")


def test_latex_writes_one_row_per_peak(tmp_path):
    path = tmp_path / "rows.tex"
    result = _fit([_peak(position=100.0), _peak(position=140.0), _peak(position=180.0)])
    fit_export.write_latex(path, [(1, result)], "s.txt")

    body = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("1 &")
    ]
    assert len(body) == 3


# --- X6: reload from disk ----------------------------------------------


def _write_histogram(path, counts):
    path.write_text("\n".join(str(int(c)) for c in counts), encoding="utf-8")


def test_reload_picks_up_new_counts_and_keeps_the_fits(qapp, tmp_path):
    """The point of the feature: watch an acquisition that is still
    counting without losing the analysis set up around it."""
    from main_window import MainWindow

    path = tmp_path / "run.txt"
    _write_histogram(path, np.full(200, 10))

    main_window = MainWindow()
    main_window._load_files([str(path)])
    spectrum = main_window.spectra[0]
    spectrum.fits.append(_fit())
    assert int(spectrum.data.sum()) == 2000

    _write_histogram(path, np.full(200, 25))
    main_window._reload_active_spectrum()

    assert int(spectrum.data.sum()) == 5000
    assert len(spectrum.fits) == 1, "fits must survive a same-length reload"


def test_reload_clears_fits_when_the_channel_count_changes(qapp, tmp_path):
    """Marks and fits are anchored to channel numbers. If the spectrum
    changed length those positions may no longer mean the same thing, and
    keeping them would leave the user reading fits that do not match the
    data under them."""
    from main_window import MainWindow

    path = tmp_path / "grow.txt"
    # The text loader rounds the channel count up to a standard bucket, so
    # the file has to grow past one for its length to change at all --
    # 100 and 150 values both land in the same bucket.
    _write_histogram(path, np.full(100, 5))

    main_window = MainWindow()
    main_window._load_files([str(path)])
    spectrum = main_window.spectra[0]
    spectrum.fits.append(_fit())
    original_length = len(spectrum.data)

    _write_histogram(path, np.full(original_length + 1000, 5))
    main_window._reload_active_spectrum()

    assert len(spectrum.data) > original_length
    assert spectrum.fits == []


def test_reload_is_disabled_for_a_spectrum_with_no_file_behind_it(qapp, tmp_path):
    """An Add/Subtract result or a matrix cut exists only in memory, so
    offering Reload for it would be an action that can only fail."""
    from main_window import MainWindow

    path = tmp_path / "real.txt"
    _write_histogram(path, np.full(50, 3))

    main_window = MainWindow()
    main_window._load_files([str(path)])
    main_window._update_operations_availability()
    assert main_window.reload_spectrum_action.isEnabled() is True

    main_window._add_combined_spectrum(str(tmp_path / "A + B"), np.full(50, 6))
    main_window._update_operations_availability()
    assert main_window.reload_spectrum_action.isEnabled() is False


def test_reload_reports_a_file_that_has_gone_away(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from main_window import MainWindow

    path = tmp_path / "vanishing.txt"
    _write_histogram(path, np.full(60, 2))

    main_window = MainWindow()
    main_window._load_files([str(path)])
    before = main_window.spectra[0].data.copy()
    os.remove(path)

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    main_window._reload_active_spectrum()

    assert warned and "no longer on disk" in warned[0]
    # The spectrum itself is left intact rather than emptied.
    assert np.array_equal(main_window.spectra[0].data, before)
