"""The rearranged Fit Results panel: # / Position / Volume / FWHM / chi^2,
with compact uncertainty notation and no fit-region column."""

import numpy as np
import pytest

from calibration import Calibration
from main_window import MainWindow
from peak_fit import fit_peaks


def _window_with_one_fit(tmp_path):
    x = np.arange(600, dtype=float)
    clean = np.full(600, 30.0) + 900.0 * np.exp(-((x - 150.0) ** 2) / (2 * 4.0 ** 2))
    y = np.random.default_rng(3).poisson(clean).astype(float)
    path = tmp_path / "s.txt"
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")

    window = MainWindow()
    window._load_files([str(path)])
    active = window.spectra[0]
    axis = np.arange(len(active.data), dtype=float)
    active.fits.append(
        fit_peaks(axis, active.data, (60, 100), (200, 240), (120, 180), [150.0])
    )
    window.fit_controller.update_results_list()
    return window


def test_columns_are_in_the_new_order(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    table = window.fit_controller.results_table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["#", "Position", "Volume", "FWHM", "chi^2"]


def test_the_index_cell_holds_no_fit_region(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, 0).text()
    assert text.strip() == "1"
    assert "[" not in text


def test_the_fit_region_is_still_in_the_tooltip(qapp, tmp_path):
    """Dropped from the column, not lost."""
    window = _window_with_one_fit(tmp_path)
    tooltip = window.fit_controller.results_table.item(0, 0).toolTip()
    assert "region" in tooltip.lower()
    assert "120" in tooltip and "180" in tooltip


def test_position_uses_compact_notation(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, 1).text()
    assert "±" not in text
    assert text.endswith(")")


def test_fwhm_uses_compact_notation(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, 3).text()
    assert "±" not in text


def test_headers_switch_to_kev_when_calibrated(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    window._apply_calibration_change(Calibration(kind="linear", a=0.0, b=2.0), True)
    window.fit_controller.update_results_list()
    table = window.fit_controller.results_table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["#", "Position (keV)", "Volume", "FWHM (keV)", "chi^2"]
