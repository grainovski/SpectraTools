"""The Fit Results panel: Position / Volume / FWHM / chi^2, with compact
uncertainty notation capped at two decimals, no row-number column, and no
fit-region column.

Column indices are named rather than inlined. When the '#' column was
dropped every index shifted by one, and two tests here kept passing while
reading the wrong cells -- 'Position uses compact notation' was reading
Volume, which is also compact, and 'FWHM has no plus-minus' was reading
chi^2, which never had one.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QHeaderView

from calibration import Calibration
from main_window import MainWindow
from peak_fit import fit_peaks

POSITION, VOLUME, FWHM, CHI2 = 0, 1, 2, 3


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


def _headers(table):
    return [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]


def _decimals(text):
    """Digits after the point in the value, ignoring any (nn) suffix."""
    value = text.split("(")[0]
    return len(value.split(".")[1]) if "." in value else 0


# ------------------------------------------------------------------ columns


def test_columns_are_in_the_new_order(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    table = window.fit_controller.results_table
    assert _headers(table) == ["Position", "Volume", "FWHM", "chi^2"]


def test_there_is_no_row_number_column(qapp, tmp_path):
    """Dropped outright: the row's identity is its position in the table,
    and the number was costing a column of width to repeat it."""
    window = _window_with_one_fit(tmp_path)
    table = window.fit_controller.results_table
    assert table.columnCount() == 4
    assert "#" not in _headers(table)


def test_headers_switch_to_kev_when_calibrated(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    window._apply_calibration_change(Calibration(kind="linear", a=0.0, b=2.0), True)
    window.fit_controller.update_results_list()
    table = window.fit_controller.results_table
    assert _headers(table) == ["Position (keV)", "Volume", "FWHM (keV)", "chi^2"]


def test_columns_are_sized_to_their_contents(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    header = window.fit_controller.results_table.horizontalHeader()
    for column in range(window.fit_controller.results_table.columnCount()):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.ResizeToContents


# ------------------------------------------------------------------ tooltip


def test_the_fit_region_is_still_in_the_tooltip(qapp, tmp_path):
    """Dropped from the columns, not lost."""
    window = _window_with_one_fit(tmp_path)
    tooltip = window.fit_controller.results_table.item(0, POSITION).toolTip()
    assert "region" in tooltip.lower()
    assert "120" in tooltip and "180" in tooltip


def test_every_cell_carries_the_tooltip(qapp, tmp_path):
    """It used to sit on the '#' cell alone. With that column gone,
    anchoring it to Position only would make the detail reachable from
    one narrow cell instead of anywhere on the row."""
    window = _window_with_one_fit(tmp_path)
    table = window.fit_controller.results_table
    for column in range(table.columnCount()):
        assert "region" in table.item(0, column).toolTip().lower()


# ----------------------------------------------------------------- notation


def test_position_is_capped_at_two_decimals(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, POSITION).text()
    assert "±" not in text
    assert _decimals(text) <= 2, text


def test_fwhm_is_capped_at_two_decimals(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, FWHM).text()
    assert "±" not in text
    assert _decimals(text) <= 2, text


def test_volume_is_not_capped(qapp, tmp_path):
    """The cap was asked for on the energy columns. Volume is a count,
    where the uncertainty routinely exceeds 1 and the parenthesised
    digits are the only thing carrying it."""
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, VOLUME).text()
    assert "±" not in text


def test_a_precise_position_drops_the_uncertainty_entirely(qapp, tmp_path):
    """An uncertainty below 0.01 cannot be written in two decimals;
    showing '(0)' would claim a perfect measurement."""
    window = _window_with_one_fit(tmp_path)
    window._apply_calibration_change(
        # b = 1e-4 keV/channel makes every energy uncertainty tiny.
        Calibration(kind="linear", a=0.0, b=1e-4), True
    )
    window.fit_controller.update_results_list()
    text = window.fit_controller.results_table.item(0, POSITION).text()
    assert "(" not in text, text
    assert _decimals(text) == 2, text
