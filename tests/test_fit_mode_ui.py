import json
import os
import tempfile

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QTableWidgetItem

import fit_mode
from main_window import MainWindow
from peak_fit import FWHM_FACTOR, FitResult, IntegrationResult, PeakResult, hypermet_left_tail
from spectrum import LoadedSpectrum

_QT_KEY = {"b": Qt.Key.Key_B, "r": Qt.Key.Key_R, "p": Qt.Key.Key_P}


def _click(main_window, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process("button_press_event", event)


def _dispatch(main_window, name, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent(name, main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process(name, event)


def _held_key_click(main_window, key_char, xdata, ydata=10.0):
    """Directly sets the controller's held-key state (bypassing real Qt
    key events) then simulates a mouse click -- the fast, simple way
    most tests exercise "given this key is held, what does a click do."
    test_key_event_filter_tracks_held_key below separately verifies the
    actual Qt event-filter wiring that sets this state in real usage."""
    main_window.fit_controller._held_key = key_char
    _click(main_window, xdata, ydata)
    main_window.fit_controller._held_key = None


def _make_active_spectrum(main_window, path=None):
    """`path` should point at a real (or plausible) location on disk --
    run_fit() writes an auto-log file next to it on every successful
    fit (see fit_export.append_auto_log). Defaults to a fresh temp
    directory per call so the many tests that don't care about export
    behavior never write into the repo; tests that DO care pass an
    explicit tmp_path-based path."""
    if path is None:
        path = os.path.join(tempfile.mkdtemp(), "synthetic.txt")
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (
        500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))
    ).astype(np.int64)
    spectrum = LoadedSpectrum(path, y, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()
    return spectrum


def test_key_event_filter_tracks_held_key(qapp):
    main_window = MainWindow()
    canvas = main_window.canvas

    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, press)
    assert main_window.fit_controller._held_key == "b"

    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, release)
    assert main_window.fit_controller._held_key is None


def test_windows_scan_code_fallback_detects_b_when_layout_remaps_the_key(qapp, monkeypatch):
    monkeypatch.setattr(fit_mode.sys, "platform", "win32")
    main_window = MainWindow()
    canvas = main_window.canvas

    # Simulates a non-Latin Windows keyboard layout (e.g. Bulgarian): the
    # physical B key produces a Qt.Key value Qt doesn't recognize as "b",
    # but its hardware scan code (0x30, layout-independent) still
    # identifies which physical key was actually pressed.
    press = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_unknown, Qt.KeyboardModifier.NoModifier,
        0x30, 0, 0, "",
    )
    QApplication.sendEvent(canvas, press)
    assert main_window.fit_controller._held_key == "b"

    release = QKeyEvent(
        QEvent.Type.KeyRelease, Qt.Key.Key_unknown, Qt.KeyboardModifier.NoModifier,
        0x30, 0, 0, "",
    )
    QApplication.sendEvent(canvas, release)
    assert main_window.fit_controller._held_key is None


def test_scan_code_fallback_does_not_apply_outside_windows(qapp, monkeypatch):
    monkeypatch.setattr(fit_mode.sys, "platform", "linux")
    main_window = MainWindow()
    canvas = main_window.canvas

    press = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_unknown, Qt.KeyboardModifier.NoModifier,
        0x30, 0, 0, "",
    )
    QApplication.sendEvent(canvas, press)
    assert main_window.fit_controller._held_key is None


def test_channel_to_display_is_identity_with_no_calibration(qapp):
    main_window = MainWindow()
    assert main_window.channel_to_display(100) == 100
    assert main_window.display_to_channel(100) == 100


def test_channel_to_display_applies_calibration_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    assert main_window.channel_to_display(100) == pytest.approx(60.0)
    assert main_window.display_to_channel(60.0) == pytest.approx(100.0, abs=1e-6)


def test_channel_to_display_is_identity_when_calibration_set_but_inactive(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = False
    assert main_window.channel_to_display(100) == 100


def test_marking_click_resolves_to_correct_channel_when_calibrated(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Click at keV 60 (= channel 100 under a=10, b=0.5) while holding 'r'.
    _held_key_click(main_window, "r", 60.0)
    assert main_window.fit_controller.state.pending_fit_click == pytest.approx(100.0, abs=1e-6)


def test_peak_toggle_click_resolves_to_correct_channel_when_calibrated(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()
    main_window.fit_controller.state.fit_region = (85.0, 115.0)  # channel space

    # Click at keV 60 (= channel 100 under a=10, b=0.5) while holding 'p'.
    _held_key_click(main_window, "p", 60.0)
    assert main_window.fit_controller.state.peak_positions == [pytest.approx(100.0, abs=1e-6)]

    # Clicking again at the same display position must resolve to the
    # same channel and fall within the (also correctly channel-converted)
    # proximity threshold of the existing peak, toggling it off -- this
    # exercises the proximity_channels conversion line, not just channel_x.
    _held_key_click(main_window, "p", 60.0)
    assert main_window.fit_controller.state.peak_positions == []


def test_full_fit_flow_commits_a_fit_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert main_window.fit_button.isEnabled() is False
    _held_key_click(main_window, "p", 100)
    assert main_window.fit_button.isEnabled() is True

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert len(result.peaks) == 1
    assert abs(result.peaks[0].position - 100.0) < 1.0


def test_marks_persist_after_a_successful_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])
    assert main_window.fit_button.isEnabled() is True


def test_results_table_headers_have_no_peak_column_and_have_chi2(qapp):
    main_window = MainWindow()
    table = main_window.fit_controller.results_table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["Fit", "Position", "FWHM", "Volume", "chi^2"]


def test_results_table_shows_one_row_per_peak_with_explicit_columns(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 1
    assert table.item(0, 1).text().startswith("100.0")  # Position
    assert "±" in table.item(0, 1).text()
    assert "±" in table.item(0, 2).text()  # FWHM
    assert "±" in table.item(0, 3).text()  # Volume
    assert table.item(0, 4).text() != ""  # chi^2


def test_results_table_has_one_row_per_peak_across_a_multi_peak_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 130.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                ),
                PeakResult(
                    position=120.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=800.0, area_err=40.0, amplitude=160.0, sigma=2.0,
                ),
            ],
        )
    )
    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 2
    assert table.item(0, 1).text().startswith("100.0")  # Position, peak 1
    assert table.item(1, 1).text().startswith("120.0")  # Position, peak 2
    assert table.item(0, 0).text() == table.item(1, 0).text()  # same Fit cell


def test_results_table_dims_hidden_fit_rows(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
            visible=False,
        )
    )
    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    default_color = QTableWidgetItem().foreground()
    assert table.item(0, 0).foreground() != default_color


def test_results_table_shows_channels_only_when_inactive(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.fit_controller.update_results_list()
    table = main_window.fit_controller.results_table
    position_text = table.item(0, 1).text()
    assert "keV" not in position_text
    assert position_text == "100.00 ± 0.10"


def test_results_table_shows_only_kev_when_active(qapp):
    """Columns switch units entirely rather than showing a combined
    "ch (keV)" string -- the narrow table columns in the real app
    truncated the combined format unreadably, so the unit is indicated
    once via the column header and the cell shows a single number."""
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    assert table.horizontalHeaderItem(1).text() == "Position (keV)"
    assert table.horizontalHeaderItem(2).text() == "FWHM (keV)"

    position_text = table.item(0, 1).text()
    # position 100.0 +/- 0.1 -> keV 60.0 +/- 0.05 (err scaled by |b|=0.5)
    assert position_text == "60.00 ± 0.05"

    fwhm_text = table.item(0, 2).text()
    # fwhm 5.0 +/- 0.2 -> keV 2.50 +/- 0.10
    assert fwhm_text == "2.50 ± 0.10"

    volume_text = table.item(0, 3).text()
    assert volume_text == "1000.0 ± 50.0"  # unchanged -- area has no keV equivalent


def test_results_table_headers_revert_to_channels_when_deactivated(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    from calibration import Calibration

    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.fit_controller.update_results_list()
    main_window._calibration_active = False
    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    assert table.horizontalHeaderItem(1).text() == "Position"
    assert table.horizontalHeaderItem(2).text() == "FWHM"


def test_results_table_fwhm_kev_uses_derivative_at_peak_position_not_fwhm_value(qapp):
    """Regression test: converting a width to keV must use the
    calibration's slope evaluated at the peak's own POSITION, not at
    the width's numeric value (which has no meaning as a channel
    index) -- for a quadratic calibration these differ substantially,
    unlike linear where the derivative is constant everywhere and the
    bug this guards against is invisible."""
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=0.5)
    main_window._calibration = cal
    main_window._calibration_active = True

    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    fwhm_text = table.item(0, 2).text()
    # Correct slope is the derivative AT THE PEAK'S POSITION (100), not
    # at the fwhm's own numeric value (5): derivative(100) = 1+2*0.5*100
    # = 101, vs. the buggy derivative(5) = 1+2*0.5*5 = 6 -- very different,
    # so this fails loudly if the bug reappears.
    expected_slope = abs(cal.derivative(100.0))
    expected_energy = expected_slope * 5.0
    expected_err = expected_slope * 0.2
    assert fwhm_text == f"{expected_energy:.2f} ± {expected_err:.2f}"


def test_remove_fit_from_context_menu_removes_the_correct_fit_by_row(qapp, monkeypatch):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(30.0, 50.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=40.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=500.0, area_err=25.0, amplitude=100.0, sigma=2.0,
                )
            ],
        )
    )
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.fit_controller.update_results_list()
    table = main_window.fit_controller.results_table
    assert table.rowCount() == 2
    second_row_item = table.item(1, 0)

    import fit_mode
    from PySide6.QtWidgets import QMenu

    class ImmediateMenu(QMenu):
        """Stands in for QMenu during this test. PySide6's compiled
        QMenu.exec() can't be intercepted by monkeypatching the class
        attribute the way addAction can -- Shiboken resolves exec()
        for a plain QMenu instance through a C-level slot that bypasses
        the patched Python attribute entirely, so the real (blocking,
        modal) popup would still run and this test would hang forever
        waiting for a selection that never comes. Overriding exec() on
        a genuine subclass works because that's ordinary Python
        instance-attribute resolution, not a monkeypatch of the base
        class -- it picks "Remove Fit" out of the real actions actually
        added via the real (unpatched) addAction, without ever opening
        a popup."""

        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text() == "Remove Fit":
                    return action
            return None

    monkeypatch.setattr(fit_mode, "QMenu", ImmediateMenu)

    position = table.visualItemRect(second_row_item).center()
    main_window.fit_controller._on_results_context_menu(position)

    assert len(spectrum.fits) == 1
    assert spectrum.fits[0].fit_region == (30.0, 50.0)


def test_double_click_a_peak_row_reloads_its_parent_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(30.0, 50.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=40.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=500.0, area_err=25.0, amplitude=100.0, sigma=2.0,
                )
            ],
        )
    )
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.fit_controller.update_results_list()
    table = main_window.fit_controller.results_table

    main_window.fit_controller._on_result_double_clicked(table.item(1, 0))

    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])


def test_row_to_fit_index_mapping_survives_an_earlier_multi_peak_fit(qapp, monkeypatch):
    # The first fit contributes TWO rows (0 and 1); the second fit's
    # single peak lands on row 2. A naive `item.row()`-as-fit-index
    # (i.e. skipping the self._results_row_fit_index[...] lookup) would
    # read active.fits[2], which doesn't exist for a 2-fit spectrum, or
    # would otherwise misattribute row 2 to the wrong fit -- every other
    # row-mapping test in this file uses only single-peak fits, so row
    # index and fit index always coincide there by construction and
    # can't catch this class of bug.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(60.0, 70.0),
            fit_region=(30.0, 50.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=40.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=500.0, area_err=25.0, amplitude=100.0, sigma=2.0,
                ),
                PeakResult(
                    position=45.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=400.0, area_err=20.0, amplitude=80.0, sigma=2.0,
                ),
            ],
        )
    )
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.fit_controller.update_results_list()
    table = main_window.fit_controller.results_table
    assert table.rowCount() == 3
    third_row_item = table.item(2, 0)

    # Double-clicking row 2 must reload the SECOND fit (fit_region
    # 85-115, single peak at 100.0) -- not raise an IndexError from
    # active.fits[2], and not silently reload the first fit instead.
    main_window.fit_controller._on_result_double_clicked(third_row_item)

    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])

    import fit_mode
    from PySide6.QtWidgets import QMenu

    class ImmediateMenu(QMenu):
        """See test_remove_fit_from_context_menu_removes_the_correct_fit_by_row
        for why this subclass -- rather than monkeypatching QMenu.exec
        directly -- is needed to avoid hanging on a real modal popup."""

        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text() == "Remove Fit":
                    return action
            return None

    monkeypatch.setattr(fit_mode, "QMenu", ImmediateMenu)

    # Removing via row 2's context menu must delete the SECOND fit only,
    # leaving the first (2-peak) fit untouched.
    position = table.visualItemRect(third_row_item).center()
    main_window.fit_controller._on_results_context_menu(position)

    assert len(spectrum.fits) == 1
    assert len(spectrum.fits[0].peaks) == 2
    assert spectrum.fits[0].fit_region == (30.0, 50.0)


def test_refitting_same_marks_with_changed_checkbox_appends_a_new_entry(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert spectrum.fits[0].link_widths is True
    assert spectrum.fits[1].link_widths is False


def test_parameters_panel_populates_after_a_fit(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    assert table.rowCount() == 3  # amp_0, pos_0, sigma (linked default)
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Shared FWHM"]


def test_parameters_panel_rebuilds_when_row_set_changes(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"

    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert table.rowCount() == 3  # amp_0, pos_0, sigma_0 (independent now)
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Peak 1 FWHM"]
    # Row set changed, so the earlier Fix checkbox must not have survived.
    assert table.cellWidget(2, 2).isChecked() is False


def test_parameters_panel_updates_values_in_place_when_row_set_is_unchanged(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"
    first_amplitude_text = table.item(0, 1).text()

    # Perturb the underlying data (marks/settings untouched, so the row
    # set stays identical) so the second fit's free amplitude value is
    # numerically different from the first. Without this, a re-fit of
    # unchanged data returns identical values, and this test would pass
    # even if the in-place value refresh were deleted entirely.
    spectrum.data[97:104] += 200

    main_window.fit_controller.run_fit()  # re-fit with the same checkboxes/marks

    assert table.rowCount() == 3
    assert table.cellWidget(2, 2).isChecked() is True  # Fix state survived

    new_result = spectrum.fits[-1]
    displayed_amplitude = table.item(0, 1).text()
    assert displayed_amplitude == f"{new_result.peaks[0].amplitude:.6g}"
    assert displayed_amplitude != first_amplitude_text


def test_marking_a_new_disjoint_region_resets_the_panel_and_the_next_fit_succeeds(qapp):
    """Regression test for a real bug: after fitting one region, marking
    a brand new region+peak elsewhere used to leave the Fit Parameters
    panel populated with the *first* fit's values. Since an unchecked
    row's value is read as an initial-guess override for the next fit
    (initial_guess_overrides_from_panel), the leftover position value
    (from the first peak) placed the optimizer's starting point outside
    the newly-marked region, and the second fit failed outright."""
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (
        500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))
    ).astype(np.int64)
    y[147:154] += (
        400 * np.exp(-((np.arange(147, 154) - 150.0) ** 2) / (2 * 3.0 ** 2))
    ).astype(np.int64)
    path = os.path.join(tempfile.mkdtemp(), "synthetic.txt")
    spectrum = LoadedSpectrum(path, y, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    assert len(spectrum.fits) == 1

    table = main_window.fit_controller.parameters_table
    assert table.rowCount() == 3  # still populated with fit 1's values

    _held_key_click(main_window, "r", 135)
    assert table.rowCount() == 3  # a lone pending click doesn't touch it
    _held_key_click(main_window, "r", 165)
    assert table.rowCount() == 0  # a *completed* new region resets it

    _held_key_click(main_window, "p", 150)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert all(f.visible for f in spectrum.fits)
    assert spectrum.fits[1].fit_region == pytest.approx((135.0, 165.0))
    assert spectrum.fits[1].peaks[0].position == pytest.approx(150.0, abs=1.0)


def test_clear_empties_the_parameters_panel(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    main_window.fit_controller.clear()

    assert main_window.fit_controller.parameters_table.rowCount() == 0


def test_width_link_and_left_tail_checkboxes_live_in_the_parameters_panel(qapp):
    main_window = MainWindow()

    assert isinstance(main_window.independent_widths_action, QCheckBox)
    assert isinstance(main_window.left_tail_action, QCheckBox)
    assert main_window.independent_widths_action.parentWidget() is main_window.fit_controller.parameters_dock.widget()
    assert main_window.left_tail_action.parentWidget() is main_window.fit_controller.parameters_dock.widget()


def test_fit_and_clear_are_shortcut_only_with_no_toolbar(qapp):
    main_window = MainWindow()

    assert main_window.fit_button.shortcut().toString() == "Ctrl+F"
    assert main_window.clear_fit_button.shortcut().toString() == "Ctrl+C"
    assert main_window.fit_button in main_window.actions()
    assert main_window.clear_fit_button in main_window.actions()
    assert not hasattr(main_window, "fit_toolbar")


def test_fixing_a_parameter_in_the_panel_holds_it_for_the_next_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"
    table.item(2, 1).setText("7.0")  # FWHM, not sigma -- the panel now displays/accepts FWHM

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    second = spectrum.fits[1]
    assert second.peaks[0].fwhm == pytest.approx(7.0)
    assert second.peaks[0].sigma_err == 0.0
    assert second.fixed_params.keys() == {"sigma"}
    assert second.fixed_params["sigma"] == pytest.approx(7.0 / FWHM_FACTOR)


def test_parameters_panel_displays_fwhm_not_sigma(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    fwhm = spectrum.fits[0].peaks[0].fwhm
    assert table.item(2, 0).text() == "Shared FWHM"
    assert table.item(2, 1).text() == f"{fwhm:.6g}"


def test_parameters_panel_displays_fwhm_per_peak_when_independent(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window.independent_widths_action.setChecked(True)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Peak 1 FWHM"]


def test_every_row_value_cell_is_editable_regardless_of_fix_state(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    for row in range(table.rowCount()):  # every row unchecked at this point
        assert table.item(row, 1).flags() & Qt.ItemFlag.ItemIsEditable

    table.cellWidget(0, 2).setChecked(True)
    assert table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable  # still editable once checked


def test_run_fit_passes_an_edited_unchecked_row_as_an_initial_guess_override(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()  # first fit populates the panel

    table = main_window.fit_controller.parameters_table
    table.item(0, 1).setText("777")  # edit "Peak 1 amplitude", left unchecked

    captured = {}
    original_fit_peaks = fit_mode.fit_peaks

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return original_fit_peaks(*args, **kwargs)

    monkeypatch.setattr(fit_mode, "fit_peaks", _spy)
    main_window.fit_controller.run_fit()

    assert captured["initial_guess_overrides"]["amp_0"] == 777.0


def test_run_fit_sets_a_timestamp_on_the_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert spectrum.fits[0].timestamp is not None
    assert spectrum.fits[0].timestamp[:4].isdigit()  # sane ISO-8601-ish prefix, not an exact match


def test_invalid_fixed_value_shows_a_status_message_instead_of_crashing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"
    table.item(2, 1).setText("not a number")

    main_window.fit_controller.run_fit()  # must not raise

    assert len(spectrum.fits) == 1  # second fit did not commit
    assert main_window.statusBar().currentMessage() != ""


def test_stale_fixed_parameter_is_dropped_without_aborting_the_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"

    # Changes the valid parameter set from "sigma" to "sigma_0" --
    # the panel hasn't rebuilt yet, so its Fix checkbox still refers
    # to the now-stale "sigma" name.
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    # The fit must succeed despite the stale fixed name (not abort with
    # a FitError), and the stale name must be silently dropped rather
    # than passed through.
    assert len(spectrum.fits) == 2
    assert spectrum.fits[1].fixed_params == {}


def test_double_click_reloads_a_committed_fit_for_editing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"
    main_window.fit_controller.run_fit()  # second entry, sigma fixed

    # Reset marks only (not clear()) -- clear() now deletes the active
    # spectrum's fits, and this test needs both committed fits to still
    # be listed so it can double-click the historical row below.
    main_window.fit_controller.reset_marks()
    assert main_window.fit_controller.state.bg_regions == []

    item = main_window.fit_controller.results_table.item(1, 0)
    main_window.fit_controller._on_result_double_clicked(item)

    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])
    assert main_window.independent_widths_action.isChecked() is False
    assert main_window.left_tail_action.isChecked() is False

    reloaded_table = main_window.fit_controller.parameters_table
    assert reloaded_table.rowCount() == 3
    assert reloaded_table.cellWidget(2, 2).isChecked() is True  # sigma was fixed
    assert reloaded_table.cellWidget(0, 2).isChecked() is False  # amplitude was free


def test_refitting_same_marks_hides_the_earlier_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert spectrum.fits[0].visible is False
    assert spectrum.fits[1].visible is True


def test_refitting_a_different_region_does_not_hide_the_first(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    _held_key_click(main_window, "b", 20)
    _held_key_click(main_window, "b", 35)
    _held_key_click(main_window, "b", 165)
    _held_key_click(main_window, "b", 180)
    _held_key_click(main_window, "r", 35)
    _held_key_click(main_window, "r", 165)
    # Peak marks persist across fits (see the 2026-07-14 "marks persist"
    # change), so the x=100 mark from the first fit is still present and
    # within pixel-proximity of this same x -- one "p" click here would
    # just remove that stale mark instead of adding a fresh one. Click
    # twice: the first click removes the stale mark, the second re-adds
    # it, leaving a single peak at x=100 as intended.
    _held_key_click(main_window, "p", 100)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert spectrum.fits[0].visible is True
    assert spectrum.fits[1].visible is True


def test_draw_committed_fits_skips_hidden_results(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
            visible=False,
        )
    )

    lines_before = len(main_window.axes.lines)
    patches_before = len(main_window.axes.patches)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    assert len(main_window.axes.lines) == lines_before
    assert len(main_window.axes.patches) == patches_before
    assert len(main_window.axes.texts) == texts_before


def test_draw_committed_fits_peak_label_shows_only_the_number(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )

    main_window.fit_controller.draw_committed_fits(spectrum)

    peak_label = main_window.axes.texts[-1]
    assert peak_label.get_text() == "100.0"
    assert "pos=" not in peak_label.get_text()


def test_draw_committed_fits_fit_color_differs_from_the_spectrum_color(qapp):
    """Regression guard for the reported "blue fit on a blue spectrum"
    clash: the drawn fit color must never equal the spectrum's own trace
    color, in either theme."""
    main_window = MainWindow()
    original_theme = main_window.settings.theme()
    try:
        spectrum = _make_active_spectrum(main_window)  # trace color "#1f77b4"
        spectrum.fits.append(
            FitResult(
                left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
                fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
                peaks=[
                    PeakResult(
                        position=100.0, position_err=0.1,
                        fwhm=5.0, fwhm_err=0.2,
                        area=1000.0, area_err=50.0,
                        amplitude=200.0, sigma=2.0,
                    )
                ],
            )
        )

        for dark in (False, True):
            main_window.dark_theme_action.setChecked(dark)
            main_window.axes.clear()
            main_window.fit_controller.draw_committed_fits(spectrum)
            peak_label = main_window.axes.texts[-1]
            assert peak_label.get_color() != spectrum.color
            assert peak_label.get_color() != "#1f77b4"
    finally:
        main_window.settings.set_theme(original_theme)


def test_draw_committed_fits_draws_one_component_line_per_peak(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 130.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                ),
                PeakResult(
                    position=115.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=800.0, area_err=40.0, amplitude=160.0, sigma=2.0,
                ),
            ],
        )
    )

    lines_before = len(main_window.axes.lines)
    main_window.fit_controller.draw_committed_fits(spectrum)

    # Background dashed line (1) + total curve (1) + one axvline position
    # marker per peak (2, pre-existing) + one new component line per peak
    # (2) = 6 new lines for this 2-peak fit, on top of whatever was
    # already there. Confirmed against the current (pre-Task-4) code
    # directly: for this exact 2-peak fixture, draw_committed_fits adds
    # exactly 4 lines before this task's change (1+1+2), so the 2 new
    # component lines bring it to 6.
    assert len(main_window.axes.lines) == lines_before + 6


def test_draw_committed_fits_draws_region_shading_and_annotation_for_integration(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )

    lines_before = len(main_window.axes.lines)
    patches_before = len(main_window.axes.patches)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    # 3 region spans (2 bg + 1 fit) as patches, 1 flat background line,
    # 1 annotation -- no peak curve/decomposition/position-marker lines.
    assert len(main_window.axes.patches) == patches_before + 3
    assert len(main_window.axes.lines) == lines_before + 1
    assert len(main_window.axes.texts) == texts_before + 1


def test_draw_committed_fits_draws_only_fit_region_and_annotation_without_background(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=None, right_bg_region=None,
            fit_region=(85.0, 115.0), background_density=0.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=0.0, background_area_err=0.0,
            background_centroid=0.0, background_centroid_err=0.0,
            background_fwhm=0.0, background_fwhm_err=0.0,
            background_skewness=0.0, background_skewness_err=0.0,
            net_area=1000.0, net_area_err=30.0,
            net_centroid=100.0, net_centroid_err=0.5,
            net_fwhm=8.0, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )

    patches_before = len(main_window.axes.patches)
    lines_before = len(main_window.axes.lines)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    # 1 region span (fit region only, no bg spans), 0 background lines,
    # 1 annotation.
    assert len(main_window.axes.patches) == patches_before + 1
    assert len(main_window.axes.lines) == lines_before + 0
    assert len(main_window.axes.texts) == texts_before + 1


def test_draw_committed_fits_skips_a_hidden_integration_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
            visible=False,
        )
    )

    lines_before = len(main_window.axes.lines)
    patches_before = len(main_window.axes.patches)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    assert len(main_window.axes.lines) == lines_before
    assert len(main_window.axes.patches) == patches_before
    assert len(main_window.axes.texts) == texts_before


def test_draw_committed_fits_draws_peak_position_at_calibrated_x(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.draw_committed_fits(spectrum)

    peak_label = main_window.axes.texts[-1]
    # Position 100.0 -> keV 60.0
    assert peak_label.get_position()[0] == pytest.approx(60.0)
    assert peak_label.get_text() == "60.0"


def test_draw_committed_fits_model_curve_unaffected_by_calibration(qapp):
    """The fitted curve's Y-values (counts) must be identical whether or
    not calibration is active -- only where it's drawn on X changes, the
    model itself is always evaluated in channel space."""
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )

    # Clear first: _make_active_spectrum's own spectrum trace (drawn via
    # steps-mid with no explicit linewidth) picks up matplotlib's default
    # lines.linewidth of 1.5 -- the same value used for the fit's total
    # curve below. Without clearing, the lw==1.5 filter would ambiguously
    # match both lines and could grab the unrelated spectrum trace instead
    # of the fit curve. Clearing before both captures keeps the filter
    # unambiguous (only the fit-drawn total curve has lw==1.5) in both
    # the uncalibrated and calibrated scenarios.
    main_window.axes.clear()
    main_window.fit_controller.draw_committed_fits(spectrum)
    uncalibrated_curve = [
        line.get_ydata().copy() for line in main_window.axes.lines if line.get_linewidth() == 1.5
    ][0]

    main_window.axes.clear()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.fit_controller.draw_committed_fits(spectrum)
    calibrated_curve = [
        line.get_ydata().copy() for line in main_window.axes.lines if line.get_linewidth() == 1.5
    ][0]

    np.testing.assert_allclose(uncalibrated_curve, calibrated_curve)


def test_draw_committed_fits_integration_result_draws_at_calibrated_x(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.draw_committed_fits(spectrum)

    # IntegrationResult draws exactly one Line2D (the background-density
    # dashed line, since axvspan patches aren't Line2D objects and this
    # branch never reaches the Gaussian/hypermet fit-curve code); its x
    # endpoints should be the calibrated fit-region bounds, not raw
    # channels.
    bg_line = main_window.axes.lines[-1]
    xdata = bg_line.get_xdata()
    assert xdata[0] == pytest.approx(52.5)  # channel 85 -> keV 10+0.5*85
    assert xdata[1] == pytest.approx(67.5)  # channel 115 -> keV 10+0.5*115

    # The centroid/FWHM/area annotation is anchored at the calibrated
    # net_centroid position.
    annotation = main_window.axes.texts[-1]
    assert annotation.get_position()[0] == pytest.approx(60.0)  # net_centroid 100 -> keV 60


@pytest.mark.xfail(
    strict=True,
    reason="NOT fixed by Task 3's _measure_width (verified): "
    "_make_active_spectrum's synthetic peak has no genuine left tail, so "
    "an accurate width guess correctly drives tail_fraction to 0, at "
    "which point tail_beta has zero effect on the model and its "
    "Jacobian column vanishes -- a real, expected parameter-"
    "identifiability degeneracy (confirmed: popt lands at tail_fraction="
    "0.0, tail_beta clamped to TAIL_BETA_MIN, pcov all-inf), not a "
    "width-estimate defect. The old region_width/(4*n_peaks) heuristic's "
    "badly-oversized starting sigma happened to push tail_fraction to "
    "its upper clamp instead, which incidentally avoided this "
    "degeneracy -- masking the issue rather than avoiding it correctly.",
)
def test_double_click_reloads_a_left_tail_fit_and_refitting_appends_a_new_entry(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 80)
    _held_key_click(main_window, "r", 120)
    _held_key_click(main_window, "p", 100)

    main_window.left_tail_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    first_result = spectrum.fits[0]
    assert first_result.tail_fraction is not None

    table = main_window.fit_controller.parameters_table
    tail_row = next(
        row for row in range(table.rowCount())
        if table.item(row, 0).text() == "Tail fraction (r)"
    )
    table.cellWidget(tail_row, 2).setChecked(True)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    second_result = spectrum.fits[1]
    assert "tail_fraction" in second_result.fixed_params

    main_window.fit_controller.clear()

    item = main_window.fit_controller.results_table.item(1, 0)
    main_window.fit_controller._on_result_double_clicked(item)

    assert main_window.left_tail_action.isChecked() is True
    reloaded_table = main_window.fit_controller.parameters_table
    reloaded_tail_row = next(
        row for row in range(reloaded_table.rowCount())
        if reloaded_table.item(row, 0).text() == "Tail fraction (r)"
    )
    assert reloaded_table.cellWidget(reloaded_tail_row, 2).isChecked() is True

    # Re-fit the reloaded entry -- must append a THIRD entry, leaving
    # the first two untouched.
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 3
    assert spectrum.fits[0] is first_result
    assert spectrum.fits[1] is second_result


def test_marking_order_is_free(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    # fit region and a peak marked before either background region
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    assert main_window.fit_button.isEnabled() is True

    main_window.fit_controller.run_fit()
    assert len(spectrum.fits) == 1


def test_bg_region_ring_buffer_eviction_via_clicks(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    # pytest.approx does not recurse into a list of tuples (only flat
    # sequences of numbers or a single tuple) -- compare each region
    # individually to absorb the floating-point noise from the
    # pixel-coordinate round-trip in _click/_held_key_click.
    regions = main_window.fit_controller.state.bg_regions
    assert len(regions) == 2
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))

    _held_key_click(main_window, "b", 150)
    _held_key_click(main_window, "b", 160)
    regions = main_window.fit_controller.state.bg_regions
    assert len(regions) == 2
    assert regions[0] == pytest.approx((115.0, 130.0))
    assert regions[1] == pytest.approx((150.0, 160.0))


def test_peak_click_toggle_adds_and_removes(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    assert main_window.fit_controller.state.peak_positions == [100.0]

    _held_key_click(main_window, "p", 100)  # same spot -- removes it
    assert main_window.fit_controller.state.peak_positions == []


def test_peak_click_outside_fit_region_is_rejected_with_a_hint(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 150)

    assert main_window.fit_controller.state.peak_positions == []
    assert main_window.statusBar().currentMessage() != ""


def test_plot_data_preserve_view_keeps_current_zoom(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    main_window.axes.set_xlim(80, 120)
    main_window._plot_data(preserve_view=True)

    assert main_window.axes.get_xlim() == (80.0, 120.0)


def test_run_fit_preserves_the_current_zoom(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.axes.set_xlim(80, 120)

    _held_key_click(main_window, "b", 81)
    _held_key_click(main_window, "b", 86)
    _held_key_click(main_window, "b", 114)
    _held_key_click(main_window, "b", 119)
    _held_key_click(main_window, "r", 90)
    _held_key_click(main_window, "r", 110)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    assert main_window.axes.get_xlim() == (80.0, 120.0)


def test_clear_discards_in_progress_marks_without_committing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.bg_regions == []
    assert main_window.fit_button.isEnabled() is False
    assert len(spectrum.fits) == 0


def test_clear_deletes_every_fit_for_the_active_spectrum(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    assert len(spectrum.fits) == 1

    main_window.fit_controller.clear()

    assert spectrum.fits == []


def test_reset_marks_clears_progress_without_touching_fits_or_replotting(qapp, monkeypatch):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    assert len(spectrum.fits) == 1

    # _plot_data() unconditionally calls nav_toolbar.push_current() near
    # its end (see main_window.py), and nothing else on this path does --
    # spying on it is a cheap, precise proxy for "a replot was triggered",
    # in the same monkeypatch-spy style already used elsewhere in this file
    # (e.g. test_run_fit_passes_an_edited_unchecked_row_as_an_initial_guess_
    # override).
    replot_calls = []
    monkeypatch.setattr(
        main_window.nav_toolbar, "push_current", lambda: replot_calls.append(True)
    )

    main_window.fit_controller.reset_marks()

    assert len(spectrum.fits) == 1  # untouched, unlike clear()
    assert main_window.fit_controller.state.bg_regions == []
    assert main_window.fit_controller.state.fit_region is None
    assert main_window.fit_controller.state.peak_positions == []
    assert replot_calls == []  # no replot triggered, unlike clear()


def test_clear_forces_a_replot_so_the_canvas_actually_goes_blank(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    lines_with_fit = len(main_window.axes.lines)
    patches_with_fit = len(main_window.axes.patches)

    main_window.fit_controller.clear()

    # Only the spectrum's own step line should remain -- no fit overlay,
    # no in-progress marking artists.
    assert len(main_window.axes.lines) == 1
    assert len(main_window.axes.patches) == 0
    assert len(main_window.axes.lines) < lines_with_fit
    assert len(main_window.axes.patches) < patches_with_fit


def test_clear_with_no_active_spectrum_does_not_crash(qapp):
    main_window = MainWindow()
    main_window.fit_controller.clear()


def test_fit_button_disabled_when_active_spectrum_hidden(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    assert main_window.fit_button.isEnabled() is True

    spectrum.visible = False
    main_window._update_fit_mode_availability()

    assert main_window.fit_button.isEnabled() is False


def test_fit_failure_leaves_marks_intact_and_shows_message(qapp):
    # Fit region deliberately far too narrow (1 data point) for the 2
    # peaks marked in it -- fit_peaks() raises FitError for "too few
    # data points", which run_fit() must catch, show as a status-bar
    # message, and NOT commit a result or reset the in-progress marks.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    # Zoom in before the r/p clicks so the two peak positions (0.8 data
    # units apart) are comfortably more than PEAK_CLICK_PIXEL_PROXIMITY
    # (8px) apart on screen -- otherwise, at the default full-spectrum
    # zoom, the second p-click would land within proximity of the first
    # and be treated as removing it instead of adding a second peak.
    # Done after the b-clicks (at 70/85/115/130, outside this narrower
    # view) so those clicks still land inside the axes' pixel bounding
    # box under the wider default view.
    main_window.axes.set_xlim(90, 110)
    _held_key_click(main_window, "r", 99.5)
    _held_key_click(main_window, "r", 100.5)
    _held_key_click(main_window, "p", 99.6)
    _held_key_click(main_window, "p", 100.4)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 0
    assert main_window.fit_controller.state.peak_positions == pytest.approx([99.6, 100.4])
    assert main_window.statusBar().currentMessage() != ""


def test_status_message_not_immediately_clobbered_by_mouse_move(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "p", 100)  # no fit region yet -- rejected with a hint

    message_after_hint = main_window.statusBar().currentMessage()
    assert message_after_hint != ""

    _dispatch(main_window, "motion_notify_event", 50)

    assert main_window.statusBar().currentMessage() == message_after_hint


def test_status_suppression_clears_on_key_release(qapp):
    main_window = MainWindow()
    canvas = main_window.canvas
    _make_active_spectrum(main_window)

    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, press)

    hint_message = main_window.statusBar().currentMessage()
    assert hint_message != ""

    # While "b" is still held, the hover readout must stay suppressed --
    # the status bar should still show the hint, unchanged.
    _dispatch(main_window, "motion_notify_event", 50)
    assert main_window.statusBar().currentMessage() == hint_message

    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, release)

    # Once "b" is released, the suppression window must end immediately
    # (not linger for the rest of the 60s hint duration) so the next
    # mouse move produces a live channel/counts readout again.
    _dispatch(main_window, "motion_notify_event", 50)
    assert main_window.statusBar().currentMessage() != hint_message


def test_plot_data_draws_committed_fit_overlay(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0),
            right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0),
            background_slope=0.0,
            background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.spectra.append(spectrum)

    main_window._plot_data()

    assert len(main_window.axes.patches) == 3
    # spectrum step line + background dashed line + total curve +
    # one component line per peak (1) + one axvline per peak (1) = 5.
    assert len(main_window.axes.lines) == 5


def _fit_result_with_one_peak():
    return FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[
            PeakResult(
                position=100.0, position_err=0.1,
                fwhm=5.0, fwhm_err=0.2,
                area=1000.0, area_err=50.0,
                amplitude=200.0, sigma=2.0,
            )
        ],
    )


def test_results_panel_lists_committed_fit(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(_fit_result_with_one_peak())
    main_window.spectra.append(spectrum)

    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 1
    assert "100.00" in table.item(0, 1).text()  # Position
    assert "5.00" in table.item(0, 2).text()  # FWHM


def test_results_panel_updates_when_active_spectrum_changes(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum_a = LoadedSpectrum("a.txt", y, "#1f77b4")
    spectrum_a.active = True
    spectrum_a.fits.append(_fit_result_with_one_peak())
    spectrum_b = LoadedSpectrum("b.txt", y, "#ff7f0e")
    main_window.spectra.extend([spectrum_a, spectrum_b])
    main_window.fit_controller.update_results_list()
    assert main_window.fit_controller.results_table.rowCount() == 1

    main_window._on_active_toggled("b.txt", True)

    assert main_window.fit_controller.results_table.rowCount() == 0


def test_clear_all_fits_empties_panel_and_spectrum(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(_fit_result_with_one_peak())
    main_window.spectra.append(spectrum)
    main_window.fit_controller.update_results_list()

    spectrum.fits.clear()
    main_window._plot_data()
    main_window.fit_controller.update_results_list()

    assert main_window.fit_controller.results_table.rowCount() == 0


def test_clear_progress_survives_an_intervening_full_replot(qapp):
    # Reproduces a real crash: mark one background region (creating an
    # in-progress artist), then something else triggers a full
    # axes.clear() (in the real app: the results panel's "Remove
    # Fit"/"Clear All Fits" context menu calling _plot_data() while a
    # fit is still mid-marking) before the in-progress fit is finished.
    # Finishing it afterward (Clear, here) must not raise.
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)

    main_window._plot_data()

    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.bg_regions == []


def test_independent_widths_checkbox_is_passed_to_fit_peaks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.independent_widths_action.setChecked(True)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    assert spectrum.fits[0].link_widths is False


@pytest.mark.xfail(
    strict=True,
    reason="NOT fixed by Task 3's _measure_width (verified): "
    "_make_active_spectrum's synthetic peak has no genuine left tail, so "
    "an accurate width guess correctly drives tail_fraction to 0, at "
    "which point tail_beta has zero effect on the model and its "
    "Jacobian column vanishes -- a real, expected parameter-"
    "identifiability degeneracy (confirmed: popt lands at tail_fraction="
    "0.0, tail_beta clamped to TAIL_BETA_MIN, pcov all-inf), not a "
    "width-estimate defect. The old region_width/(4*n_peaks) heuristic's "
    "badly-oversized starting sigma happened to push tail_fraction to "
    "its upper clamp instead, which incidentally avoided this "
    "degeneracy -- masking the issue rather than avoiding it correctly.",
)
def test_left_tail_checkbox_is_passed_to_fit_peaks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.left_tail_action.setChecked(True)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    assert spectrum.fits[0].tail_fraction is not None


def test_plot_data_draws_committed_fit_overlay_with_left_tail(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    fit_result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[
            PeakResult(
                position=100.0, position_err=0.1,
                fwhm=5.0, fwhm_err=0.2,
                area=1000.0, area_err=50.0,
                amplitude=200.0, sigma=2.0,
            )
        ],
        link_widths=True,
        tail_fraction=0.1, tail_fraction_err=0.02,
        tail_beta=3.0, tail_beta_err=0.5,
    )
    spectrum.fits.append(fit_result)
    main_window.spectra.append(spectrum)

    main_window._plot_data()  # must not raise

    assert len(main_window.axes.patches) == 3
    # spectrum step line + background dashed line + total curve +
    # one component line per peak (1) + one axvline per peak (1) = 5.
    assert len(main_window.axes.lines) == 5

    # The drawn curve must reflect the tail-aware formula, not a plain
    # Gaussian -- otherwise this test would pass even if the
    # tail_fraction branch in draw_committed_fits were broken/skipped.
    # Line order from draw_committed_fits: [0] spectrum step line,
    # [1] background dashed line, [2] total fit curve, [3] per-peak
    # component line, [4] peak-position axvline (one per peak) -- the
    # total curve is third-from-last here.
    drawn_curve = main_window.axes.lines[-3].get_ydata()
    x_dense = np.linspace(90.0, 110.0, 200)
    expected_curve = 20.0 + fit_result.peaks[0].amplitude * hypermet_left_tail(
        x_dense, fit_result.peaks[0].position, fit_result.peaks[0].sigma,
        fit_result.tail_fraction, fit_result.tail_beta,
    )
    plain_gaussian_curve = 20.0 + fit_result.peaks[0].amplitude * np.exp(
        -((x_dense - fit_result.peaks[0].position) ** 2) / (2 * fit_result.peaks[0].sigma ** 2)
    )
    np.testing.assert_allclose(drawn_curve, expected_curve, rtol=1e-9)
    assert not np.allclose(drawn_curve, plain_gaussian_curve, rtol=1e-3)


def test_results_panel_shows_tail_and_width_link_info(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0),
            right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0),
            background_slope=0.0,
            background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
            link_widths=False,
            tail_fraction=0.1, tail_fraction_err=0.02,
            tail_beta=3.0, tail_beta_err=0.5,
        )
    )
    main_window.spectra.append(spectrum)

    main_window.fit_controller.update_results_list()

    tooltip = main_window.fit_controller.results_table.item(0, 0).toolTip()
    assert "independent widths" in tooltip
    assert "r=0.10" in tooltip
    assert "volume excludes tail" in tooltip


def test_results_panel_shows_full_and_net_area_for_a_gaussian_fit(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0),
            right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0),
            background_slope=0.0,
            background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                    full_area=1420.0, full_area_err=50.0,
                )
            ],
            gross_area=1420.0, gross_area_err=37.7,
            net_area=1000.0, net_area_err=50.0,
            reduced_chi2=1.42,
        )
    )
    main_window.spectra.append(spectrum)

    main_window.fit_controller.update_results_list()

    tooltip = main_window.fit_controller.results_table.item(0, 0).toolTip()
    assert "peak full (no bg subtracted): 1420.0 ± 50.0" in tooltip
    assert "peak net (bg subtracted): 1000.0 ± 50.0" in tooltip
    assert "region full (no bg subtracted): 1420.0 ± 37.7" in tooltip
    assert "region net (bg subtracted): 1000.0 ± 50.0" in tooltip
    assert "reduced chi^2: 1.42" in tooltip


def test_run_fit_appends_to_the_auto_log_next_to_the_spectrum_file(qapp, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    log_path = tmp_path / "eu_fits.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_run_fit_appends_a_second_line_for_a_second_fit(qapp, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    log_path = tmp_path / "eu_fits.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_run_fit_shows_a_status_message_when_the_auto_log_write_fails(qapp, tmp_path, monkeypatch):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    spectrum = _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fit_mode.fit_export, "append_auto_log", _raise)
    main_window.fit_controller.run_fit()  # must not raise

    assert len(spectrum.fits) == 1  # the fit still committed
    assert main_window.statusBar().currentMessage() != ""


def test_export_actions_appear_when_right_clicking_an_existing_fit(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from PySide6.QtWidgets import QMenu

    captured = {}

    class CapturingMenu(QMenu):
        def exec(self, *args, **kwargs):
            captured["actions"] = [a.text() for a in self.actions()]
            return None

    monkeypatch.setattr(fit_mode, "QMenu", CapturingMenu)

    table = main_window.fit_controller.results_table
    position = table.visualItemRect(table.item(0, 0)).center()
    main_window.fit_controller._on_results_context_menu(position)

    assert captured["actions"] == [
        "Remove Fit", "Export This Fit...", "Export All Fits...", "Clear All Fits",
    ]


def test_export_all_fits_action_absent_when_there_are_no_fits(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    from PySide6.QtWidgets import QMenu

    captured = {}

    class CapturingMenu(QMenu):
        def exec(self, *args, **kwargs):
            captured["actions"] = [a.text() for a in self.actions()]
            return None

    monkeypatch.setattr(fit_mode, "QMenu", CapturingMenu)

    table = main_window.fit_controller.results_table
    main_window.fit_controller._on_results_context_menu(table.rect().center())

    assert captured["actions"] == ["Clear All Fits"]


def test_export_this_fit_writes_a_report_to_the_chosen_path(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from PySide6.QtWidgets import QMenu

    class ImmediateExportMenu(QMenu):
        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text() == "Export This Fit...":
                    return action
            return None

    monkeypatch.setattr(fit_mode, "QMenu", ImmediateExportMenu)
    out_path = tmp_path / "chosen_report.txt"
    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))

    table = main_window.fit_controller.results_table
    position = table.visualItemRect(table.item(0, 0)).center()
    main_window.fit_controller._on_results_context_menu(position)

    text = out_path.read_text(encoding="utf-8")
    assert "Fit 1" in text
    assert spectrum_path in text


def test_export_all_fits_writes_every_fit_in_order(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    from PySide6.QtWidgets import QMenu

    class ImmediateExportAllMenu(QMenu):
        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text() == "Export All Fits...":
                    return action
            return None

    monkeypatch.setattr(fit_mode, "QMenu", ImmediateExportAllMenu)
    out_path = tmp_path / "all_report.txt"
    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))

    table = main_window.fit_controller.results_table
    position = table.visualItemRect(table.item(0, 0)).center()
    main_window.fit_controller._on_results_context_menu(position)

    text = out_path.read_text(encoding="utf-8")
    assert text.index("Fit 1") < text.index("Fit 2")


def test_export_cancelled_dialog_does_not_write_a_report_file(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()  # writes eu_fits.jsonl (the auto-log)

    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    controller = main_window.fit_controller
    controller._export_fits(main_window.spectra[0], [0])  # must not raise

    assert [p.name for p in tmp_path.iterdir()] == ["eu_fits.jsonl"]


def test_export_shows_a_status_message_when_the_write_fails(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    bad_path = str(tmp_path / "missing_dir" / "report.txt")
    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: (bad_path, ""))

    controller = main_window.fit_controller
    controller._export_fits(main_window.spectra[0], [0])  # must not raise

    assert main_window.statusBar().currentMessage() != ""


def test_ready_to_integrate_needs_bg_regions_and_fit_region_but_not_peaks(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    state = main_window.fit_controller.state

    assert state.ready_to_integrate() is False
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    assert state.ready_to_integrate() is False
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert state.ready_to_integrate() is True  # no peak marks needed


def test_integrate_button_has_ctrl_i_shortcut_and_is_gated(qapp):
    main_window = MainWindow()
    assert main_window.integrate_button.shortcut().toString() == "Ctrl+I"
    assert main_window.integrate_button.isEnabled() is False

    _make_active_spectrum(main_window)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert main_window.integrate_button.isEnabled() is True


def test_run_integration_appends_an_integration_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)

    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert isinstance(result, IntegrationResult)
    assert result.timestamp is not None


def test_ready_to_integrate_allows_zero_background_regions(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    state = main_window.fit_controller.state

    assert state.ready_to_integrate() is False
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert state.ready_to_integrate() is True  # no background marks needed


def test_integrate_button_enables_with_zero_background_regions(qapp):
    main_window = MainWindow()
    assert main_window.integrate_button.isEnabled() is False

    _make_active_spectrum(main_window)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert main_window.integrate_button.isEnabled() is True  # no background marks needed


def test_run_integration_with_no_background_produces_a_backgroundless_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert isinstance(result, IntegrationResult)
    assert result.has_background is False
    assert result.left_bg_region is None
    assert result.right_bg_region is None
    assert result.net_area == result.gross_area


def test_results_table_shows_a_region_row_for_an_integration_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 1
    result = spectrum.fits[0]
    assert table.item(0, 1).text() == f"{result.net_centroid:.2f} ± {result.net_centroid_err:.2f}"
    assert table.item(0, 2).text() == f"{result.net_fwhm:.2f} ± {result.net_fwhm_err:.2f}"
    assert table.item(0, 3).text() == f"{result.net_area:.1f} ± {result.net_area_err:.1f}"
    assert table.item(0, 4).text() == "—"  # no chi^2 concept for Integration
    tooltip = table.item(0, 0).toolTip()
    assert "Gross:" in tooltip
    assert "Background:" in tooltip
    assert "Net:" in tooltip


def test_tooltip_shows_dual_units_for_centroid_and_fwhm_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.fit_controller.update_results_list()

    tooltip = main_window.fit_controller.results_table.item(0, 0).toolTip()
    assert "keV" in tooltip


def test_integration_tooltip_is_a_single_line_without_background(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=None, right_bg_region=None,
            fit_region=(85.0, 115.0), background_density=0.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=0.0, background_area_err=0.0,
            background_centroid=0.0, background_centroid_err=0.0,
            background_fwhm=0.0, background_fwhm_err=0.0,
            background_skewness=0.0, background_skewness_err=0.0,
            net_area=1000.0, net_area_err=30.0,
            net_centroid=100.0, net_centroid_err=0.5,
            net_fwhm=8.0, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )

    main_window.fit_controller.update_results_list()

    tooltip = main_window.fit_controller.results_table.item(0, 0).toolTip()
    assert "\n" not in tooltip
    assert "Gross" not in tooltip
    assert "Background" not in tooltip
    assert "Net" not in tooltip
    assert "Area=1000.0" in tooltip


def test_results_table_dims_hidden_integration_rows(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.3,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=600.0, background_area_err=20.0,
            background_centroid=100.0, background_centroid_err=0.5,
            background_fwhm=8.0, background_fwhm_err=0.3,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=400.0, net_area_err=36.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=8.0, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.2,
            visible=False,
        )
    )
    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    default_color = QTableWidgetItem().foreground()
    assert table.item(0, 0).foreground() != default_color


def test_results_table_integration_row_shows_only_kev_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    # net_centroid 100.0 +/- 0.6 -> keV 60.00 +/- 0.30
    assert table.item(0, 1).text() == "60.00 ± 0.30"


def test_run_integration_ignores_any_marked_peaks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)  # a peak is marked too

    main_window.fit_controller.run_integration()  # must not crash or use it

    assert len(spectrum.fits) == 1
    assert isinstance(spectrum.fits[0], IntegrationResult)


def test_marks_persist_after_a_successful_integration(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)

    main_window.fit_controller.run_integration()

    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))


def test_run_integration_failure_leaves_marks_intact_and_shows_message(qapp):
    # Fit region deliberately placed strictly between two integer
    # channels (no data point falls inside) -- integrate_region() raises
    # FitError for "contains no data", which run_integration() must
    # catch, show as a status-bar message, and NOT commit a result or
    # reset the in-progress marks.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    # Zoom in before the r-clicks so the two boundaries (0.2/0.8 data
    # units off a whole channel) don't get lost in the pixel-coordinate
    # round-trip noise of the wider default view -- same technique as
    # test_fit_failure_leaves_marks_intact_and_shows_message. Done after
    # the b-clicks (outside this narrower view) so those still land
    # inside the axes' pixel bounding box under the wider default view.
    main_window.axes.set_xlim(95, 105)
    _held_key_click(main_window, "r", 100.2)
    _held_key_click(main_window, "r", 100.8)

    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 0
    assert main_window.fit_controller.state.fit_region == pytest.approx((100.2, 100.8))
    assert main_window.statusBar().currentMessage() != ""


def test_double_click_reloads_an_integration_result_marks_only(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    # Reset marks only (not clear()) -- clear() now deletes the active
    # spectrum's fits, and this test needs the committed integration
    # result to still be listed so it can double-click its row below.
    main_window.fit_controller.reset_marks()
    assert main_window.fit_controller.state.bg_regions == []

    item = main_window.fit_controller.results_table.item(0, 0)
    main_window.fit_controller._on_result_double_clicked(item)

    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == []


def test_run_integration_appends_to_the_auto_log(qapp, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    log_path = tmp_path / "eu_fits.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "integration"


def test_calibration_menu_action_exists(qapp):
    main_window = MainWindow()
    assert hasattr(main_window, "calibration_action")
    assert main_window.calibration_action.text() == "Calibration..."


def test_applying_calibration_from_dialog_updates_state_and_replots(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    applied = Calibration(kind="linear", a=10.0, b=0.5)

    def fake_exec(self):
        self.result_calibration = applied
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    assert main_window._calibration == applied
    assert main_window._calibration_active is True
    assert main_window.axes.get_xlabel() == "Energy (keV)"


def test_cancelling_calibration_dialog_leaves_state_untouched(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    def fake_exec(self):
        # Simulate a dialog where the user entered valid values (so
        # result_calibration/result_active would be non-default if
        # applied) but then hit Cancel -- this must distinguish "reject
        # correctly ignored" from "the Accepted-guard was never checked
        # at all", which a dialog left at its untouched defaults cannot.
        self.result_calibration = Calibration(kind="linear", a=99.0, b=1.0)
        self.result_active = True
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    assert main_window._calibration is None
    assert main_window._calibration_active is False
    assert main_window.axes.get_xlabel() == "Channel"


def test_plot_data_draws_calibrated_x_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window._plot_data()

    line = main_window.axes.lines[0]
    xdata = line.get_xdata()
    assert xdata[0] == pytest.approx(10.0)  # channel 0 -> a
    assert xdata[-1] == pytest.approx(10.0 + 0.5 * (len(spectrum.data) - 1))
    assert main_window.axes.get_xlabel() == "Energy (keV)"


def test_plot_data_draws_raw_channels_when_inactive(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._plot_data()
    line = main_window.axes.lines[0]
    xdata = line.get_xdata()
    assert xdata[0] == 0
    assert main_window.axes.get_xlabel() == "Channel"


def test_plot_data_xlim_spans_calibrated_range_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window._plot_data()

    xlim = main_window.axes.get_xlim()
    max_channel = len(spectrum.data) - 1
    assert xlim[0] == pytest.approx(10.0, abs=1.0)
    assert xlim[1] == pytest.approx(10.0 + 0.5 * max_channel, abs=1.0)


def test_autoscale_y_uses_correct_channel_range_when_calibrated(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)  # 200 channels -> calibrated range [10, 109.5] keV
    spectrum.data[50] = 99999  # channel 50 -> keV 35, OUTSIDE the range queried below
    # channel 150 -> keV 85, INSIDE the range queried below. Set well above
    # 520 (the synthetic spectrum's own built-in bump peak at channel 100,
    # see _make_active_spectrum) so this value can't accidentally satisfy
    # the assertion below via the *unfixed* code's channel-vs-keV mixup,
    # which misinterprets the query as raw channels [70, 101) -- a range
    # that happens to include that channel-100 bump but not channel 150.
    spectrum.data[150] = 5000
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Query the y-autoscale over keV [70, 100] -> channel [120, 180]:
    # excludes channel 50's spike, includes channel 150's value.
    main_window._autoscale_y((70.0, 100.0))
    ylim = main_window.axes.get_ylim()
    assert ylim[1] < 99999  # the out-of-range spike must not be included
    assert ylim[1] >= 5000  # the in-range value must be


def test_show_full_spectrum_spans_calibrated_range(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    main_window._show_full_spectrum()
    xlim = main_window.axes.get_xlim()
    max_channel = len(spectrum.data) - 1
    assert xlim[0] == pytest.approx(10.0, abs=1.0)
    assert xlim[1] == pytest.approx(10.0 + 0.5 * max_channel, abs=1.0)


def test_zoom_x_stays_within_calibrated_bounds(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Zoom out repeatedly -- the view must clamp to the calibrated full
    # range, not the raw channel range (10..109.5 for this spectrum),
    # which for a 200-channel spectrum would be far narrower than
    # letting it clamp to [0, 199].
    for _ in range(10):
        main_window._zoom_x(2.0)
    xlim = main_window.axes.get_xlim()
    max_channel = len(spectrum.data) - 1
    assert xlim[0] == pytest.approx(10.0, abs=1.0)
    assert xlim[1] == pytest.approx(10.0 + 0.5 * max_channel, abs=1.0)


def test_zoom_x_handles_negative_b_calibration(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=1000.0, b=-0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    max_channel = len(spectrum.data) - 1
    full_lo = main_window.channel_to_display(0)
    full_hi = main_window.channel_to_display(max_channel)

    # Zoom out repeatedly from the full view -- must clamp to the full
    # calibrated range (regardless of axis orientation), not collapse
    # to a near-zero-width view (the b<0 zoom-arithmetic bug this
    # guards against).
    for _ in range(10):
        main_window._zoom_x(2.0)
    xlim = main_window.axes.get_xlim()
    assert min(xlim) == pytest.approx(min(full_lo, full_hi), abs=1.0)
    assert max(xlim) == pytest.approx(max(full_lo, full_hi), abs=1.0)

    # Zoom in from there -- width should shrink smoothly to roughly
    # half, not collapse to width ~1 (the pre-fix symptom).
    main_window._zoom_x(0.5)
    zoomed_xlim = main_window.axes.get_xlim()
    zoomed_width = abs(zoomed_xlim[1] - zoomed_xlim[0])
    full_width = abs(full_hi - full_lo)
    assert zoomed_width == pytest.approx(full_width * 0.5, rel=0.1)


def test_on_mouse_move_reports_correct_channel_when_calibrated(qapp):
    from calibration import Calibration
    from matplotlib.backend_bases import MouseEvent

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data[100] = 12345
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Channel 100 -> keV 10 + 0.5*100 = 60.
    px, py = main_window.axes.transData.transform((60.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window._on_mouse_move(event)

    message = main_window.statusBar().currentMessage()
    assert "12345" in message


def test_redraw_progress_draws_marks_at_calibrated_position(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    fc = main_window.fit_controller
    fc.state.peak_positions = [100.0]  # channel 100
    fc._redraw_progress()

    # The peak-position progress line should be drawn at keV 60, not
    # channel 100.
    peak_lines = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ]
    assert len(peak_lines) == 1
    assert peak_lines[0].get_xdata()[0] == pytest.approx(60.0)


def test_marks_survive_toggling_calibration_mid_session(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    fc = main_window.fit_controller

    # Mark a peak position while uncalibrated.
    fc.state.fit_region = (85.0, 115.0)
    fc.state.peak_positions = [100.0]
    fc._redraw_progress()
    peak_line = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ][0]
    assert peak_line.get_xdata()[0] == pytest.approx(100.0)

    # Turn calibration on -- the mark's stored value is untouched, only
    # where it's drawn changes.
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    fc._redraw_progress()
    assert fc.state.peak_positions == [100.0]  # stored value unchanged
    peak_line = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ][0]
    assert peak_line.get_xdata()[0] == pytest.approx(60.0)  # drawn at keV now

    # Turn it back off -- redraws at the original channel position again.
    main_window._calibration_active = False
    fc._redraw_progress()
    assert fc.state.peak_positions == [100.0]
    peak_line = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ][0]
    assert peak_line.get_xdata()[0] == pytest.approx(100.0)


def test_run_fit_auto_log_includes_kev_when_calibrated(qapp, tmp_path):
    import json

    from calibration import Calibration

    main_window = MainWindow()
    path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=path)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    # Clicks resolve through display_to_channel when calibration is
    # active (see test_marking_click_resolves_to_correct_channel_when_calibrated
    # above) -- convert the intended channel positions to display (keV)
    # coordinates first so the marks land on the same channels 70/85/
    # 115/130/100 as the uncalibrated sibling test below, rather than
    # on their raw keV-mislabeled values (which would push the right
    # background region past the end of the 200-channel spectrum).
    to_display = main_window.channel_to_display
    _held_key_click(main_window, "b", to_display(70))
    _held_key_click(main_window, "b", to_display(85))
    _held_key_click(main_window, "b", to_display(115))
    _held_key_click(main_window, "b", to_display(130))
    _held_key_click(main_window, "r", to_display(85))
    _held_key_click(main_window, "r", to_display(115))
    _held_key_click(main_window, "p", to_display(100))
    main_window.fit_controller.run_fit()

    from fit_export import auto_log_path
    with open(auto_log_path(path), encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["peaks"][0]["position_keV"] is not None


def test_run_fit_auto_log_omits_kev_when_not_calibrated(qapp, tmp_path):
    import json

    main_window = MainWindow()
    path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from fit_export import auto_log_path
    with open(auto_log_path(path), encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["peaks"][0].get("position_keV") is None


def test_calibration_toolbar_actions_exist(qapp):
    main_window = MainWindow()
    assert hasattr(main_window, "calibration_load_action")
    assert main_window.calibration_load_action.text() == "Load Calibration..."
    assert hasattr(main_window, "calibration_toggle_action")
    assert main_window.calibration_toggle_action.text() == "Calibration Active"
    assert main_window.calibration_toggle_action.isCheckable() is True


def test_calibration_toggle_action_starts_disabled_with_no_calibration_set(qapp):
    main_window = MainWindow()
    assert main_window.calibration_toggle_action.isEnabled() is False
    assert main_window.calibration_toggle_action.isChecked() is False


def test_calibration_load_action_opens_the_dialog(qapp, monkeypatch):
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    opened = []

    def fake_exec(self):
        opened.append(True)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window.calibration_load_action.trigger()
    assert opened == [True]


def test_calibration_toggle_action_becomes_enabled_after_calibration_set(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()

    def fake_exec(self):
        self.result_calibration = Calibration(kind="linear", a=10.0, b=0.5)
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    assert main_window.calibration_toggle_action.isEnabled() is True
    assert main_window.calibration_toggle_action.isChecked() is True


def test_applying_calibration_preserves_the_equivalent_channel_view(qapp, monkeypatch):
    """Regression test: applying a calibration must keep the same
    detector-channel region in view (now shown in keV), not reset to
    the full spectrum -- a real bug reported after the calibration
    feature's first release."""
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window.axes.set_xlim(80.0, 120.0)  # simulates a user having zoomed in

    def fake_exec(self):
        self.result_calibration = Calibration(kind="linear", a=10.0, b=0.5)
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    xlim = main_window.axes.get_xlim()
    # Channel 80 -> keV 50.0, channel 120 -> keV 70.0 -- the SAME
    # channel region, not the full spectrum's [10.0, 109.5] range.
    assert xlim[0] == pytest.approx(50.0, abs=1.0)
    assert xlim[1] == pytest.approx(70.0, abs=1.0)


def test_deactivating_calibration_via_dialog_preserves_the_equivalent_view(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()
    main_window.axes.set_xlim(50.0, 70.0)  # zoomed to keV 50-70 (channels 80-120)

    def fake_exec(self):
        self.result_calibration = main_window._calibration
        self.result_active = False
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    xlim = main_window.axes.get_xlim()
    assert xlim[0] == pytest.approx(80.0, abs=1.0)
    assert xlim[1] == pytest.approx(120.0, abs=1.0)


def test_calibration_toggle_action_flips_active_state_and_preserves_view(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = False
    main_window.calibration_toggle_action.setEnabled(True)
    main_window._plot_data()
    main_window.axes.set_xlim(80.0, 120.0)

    main_window.calibration_toggle_action.setChecked(True)

    assert main_window._calibration_active is True
    xlim = main_window.axes.get_xlim()
    assert xlim[0] == pytest.approx(50.0, abs=1.0)
    assert xlim[1] == pytest.approx(70.0, abs=1.0)

    main_window.calibration_toggle_action.setChecked(False)
    assert main_window._calibration_active is False
    xlim = main_window.axes.get_xlim()
    assert xlim[0] == pytest.approx(80.0, abs=1.0)
    assert xlim[1] == pytest.approx(120.0, abs=1.0)


def test_toggle_action_and_dialog_stay_in_sync(qapp, monkeypatch):
    """Applying a calibration via the dialog must update the toolbar
    toggle button's checked state to match, so the two controls never
    disagree about whether calibration is active."""
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    def fake_exec_active(self):
        self.result_calibration = Calibration(kind="linear", a=10.0, b=0.5)
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec_active)
    main_window._open_calibration_dialog()
    assert main_window.calibration_toggle_action.isChecked() is True

    def fake_exec_inactive(self):
        self.result_calibration = main_window._calibration
        self.result_active = False
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec_inactive)
    main_window._open_calibration_dialog()
    assert main_window.calibration_toggle_action.isChecked() is False


def test_calibration_toggle_action_icon_updates_with_checked_state(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window.calibration_toggle_action.setEnabled(True)

    off_icon_bytes = main_window.calibration_toggle_action.icon().pixmap(24, 24).toImage()
    main_window.calibration_toggle_action.setChecked(True)
    on_icon_bytes = main_window.calibration_toggle_action.icon().pixmap(24, 24).toImage()
    assert off_icon_bytes != on_icon_bytes


def test_parameters_panel_shows_channels_only_when_inactive(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    assert table.columnCount() == 3
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Shared FWHM"]
    assert "keV" not in table.item(1, 1).text()


def test_parameters_panel_shows_kev_only_when_active(qapp):
    """Position/FWHM rows switch units entirely (label gains a "(keV)"
    suffix, Value shows the keV number directly) rather than showing a
    combined "ch (keV)" string -- unlike the read-only Fit Results
    table, this Value column is editable and feeds directly into the
    next fit_peaks() call, so it can only ever hold one plain,
    parseable number at a time."""
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.calibration_toggle_action.setEnabled(True)
    main_window.calibration_toggle_action.setChecked(True)
    main_window._plot_data()

    to_display = main_window.channel_to_display
    _held_key_click(main_window, "b", to_display(70))
    _held_key_click(main_window, "b", to_display(85))
    _held_key_click(main_window, "b", to_display(115))
    _held_key_click(main_window, "b", to_display(130))
    _held_key_click(main_window, "r", to_display(85))
    _held_key_click(main_window, "r", to_display(115))
    _held_key_click(main_window, "p", to_display(100))
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    fit = main_window.spectra[0].fits[0]
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position (keV)", "Shared FWHM (keV)"]
    # amplitude has no energy-axis equivalent, matching Volume/chi^2 in
    # the Fit Results table -- shown unconverted.
    assert table.item(0, 1).text() == f"{fit.peaks[0].amplitude:.6g}"
    # position converts through the full calibration.
    assert table.item(1, 1).text() == f"{main_window._calibration.apply(fit.peaks[0].position):.6g}"
    # FWHM scales by the derivative at the peak's own position.
    slope = abs(main_window._calibration.derivative(fit.peaks[0].position))
    assert table.item(2, 1).text() == f"{slope * fit.peaks[0].fwhm:.6g}"


def test_parameters_panel_independent_widths_use_each_peaks_own_position(qapp):
    """Regression guard: the same derivative-evaluation-point mistake
    already found and fixed multiple times elsewhere in this feature
    (evaluating a width's keV slope at the width's own value, or at
    the wrong peak's position, instead of at THAT peak's own position)
    must not recur here. Uses a quadratic calibration and two peaks at
    very different positions, so a wrong reference position produces a
    substantially different (and easily detectable) wrong answer."""
    from calibration import Calibration

    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))).astype(np.int64)
    y[145:152] += (300 * np.exp(-((np.arange(145, 152) - 148.0) ** 2) / (2 * 2.5 ** 2))).astype(np.int64)
    spectrum = LoadedSpectrum(os.path.join(tempfile.mkdtemp(), "two_peaks.txt"), y, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()

    main_window._calibration = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.001)
    main_window._calibration_active = True
    main_window.calibration_toggle_action.setEnabled(True)
    main_window.calibration_toggle_action.setChecked(True)
    main_window._plot_data()
    main_window.independent_widths_action.setChecked(True)

    to_display = main_window.channel_to_display
    _held_key_click(main_window, "b", to_display(70))
    _held_key_click(main_window, "b", to_display(85))
    _held_key_click(main_window, "b", to_display(160))
    _held_key_click(main_window, "b", to_display(175))
    _held_key_click(main_window, "r", to_display(85))
    _held_key_click(main_window, "r", to_display(160))
    _held_key_click(main_window, "p", to_display(100))
    _held_key_click(main_window, "p", to_display(148))
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    fit = spectrum.fits[0]
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == [
        "Peak 1 amplitude", "Peak 1 position (keV)", "Peak 1 FWHM (keV)",
        "Peak 2 amplitude", "Peak 2 position (keV)", "Peak 2 FWHM (keV)",
    ]
    cal = main_window._calibration
    peak1_slope = abs(cal.derivative(fit.peaks[0].position))
    peak2_slope = abs(cal.derivative(fit.peaks[1].position))
    assert peak1_slope != pytest.approx(peak2_slope, rel=0.05)  # positions differ enough to matter
    assert table.item(2, 1).text() == f"{peak1_slope * fit.peaks[0].fwhm:.6g}"
    assert table.item(5, 1).text() == f"{peak2_slope * fit.peaks[1].fwhm:.6g}"


def test_parameters_panel_value_column_switches_units_on_calibration_toggle(qapp):
    """The Value column must not lag behind an already-populated panel
    when calibration is toggled without re-fitting -- mirrors the same
    guarantee the plot and Fit Results table already have."""
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    assert table.item(1, 0).text() == "Peak 1 position"
    assert table.item(1, 1).text() == "100"

    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window.calibration_toggle_action.setEnabled(True)
    main_window.calibration_toggle_action.setChecked(True)  # no re-fit in between

    assert table.item(1, 0).text() == "Peak 1 position (keV)"
    assert table.item(1, 1).text() == "60"


def test_parameters_panel_calibration_toggle_resets_fix_checkboxes(qapp):
    """Toggling calibration is treated like a parameter-set change --
    a full rebuild -- rather than trying to re-interpret an
    already-displayed, possibly hand-edited value in the new unit,
    which risks silently feeding a badly wrong number into the next
    fit. Deliberate trade-off: fixing a row, then toggling calibration
    before the next fit, clears that fix rather than risking a silent
    misconversion of whatever the user typed."""
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window.calibration_toggle_action.setEnabled(True)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"
    assert table.cellWidget(2, 2).isChecked() is True

    main_window.calibration_toggle_action.setChecked(True)

    assert table.cellWidget(2, 2).isChecked() is False  # cleared, not silently reinterpreted
    assert table.item(2, 0).text() == "Shared FWHM (keV)"


def test_parameters_panel_fixed_row_value_survives_a_refit_with_different_values(qapp):
    """A Fixed row's Value is deliberately frozen at whatever the user
    last saw/edited, even across a re-fit that computes a different
    value for that same parameter (same parameter names AND unchanged
    calibration state, so update_parameters_panel takes its
    in-place-update branch, not a full rebuild) -- it must stay frozen
    rather than silently being overwritten by the new, un-displayed
    fit result."""
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.calibration_toggle_action.setEnabled(True)
    main_window.calibration_toggle_action.setChecked(True)

    fc = main_window.fit_controller
    names = ["amp_0", "pos_0", "sigma"]
    fc.update_parameters_panel(names, {"amp_0": 200.0, "pos_0": 100.0, "sigma": 2.0})

    table = fc.parameters_table
    table.cellWidget(1, 2).setChecked(True)  # fix "Peak 1 position (keV)"
    frozen_value = table.item(1, 1).text()
    assert frozen_value == "60"  # 10 + 0.5*100

    # Same parameter names, same calibration state, but a different fit
    # result for the fixed position -- as if the user re-fit after
    # editing an unrelated mark.
    fc.update_parameters_panel(names, {"amp_0": 210.0, "pos_0": 130.0, "sigma": 2.1})

    assert table.item(1, 1).text() == frozen_value
    assert table.item(1, 1).text() != "75"  # NOT 10+0.5*130 -- would mean it drifted


def test_fixed_kev_position_is_honored_by_a_real_refit(qapp):
    """End-to-end regression guard for the highest-risk path in this
    feature: a keV value typed into the editable Parameters panel,
    fixed, and read back must convert to the exact channel value that
    fit_peaks() actually honors in a real re-fit -- not just display
    correctly, and not just parse correctly in isolation. A bug here
    would silently corrupt fit results, not just misrender text."""
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.calibration_toggle_action.setEnabled(True)
    main_window.calibration_toggle_action.setChecked(True)
    main_window._plot_data()

    to_display = main_window.channel_to_display
    _held_key_click(main_window, "b", to_display(70))
    _held_key_click(main_window, "b", to_display(85))
    _held_key_click(main_window, "b", to_display(115))
    _held_key_click(main_window, "b", to_display(130))
    _held_key_click(main_window, "r", to_display(85))
    _held_key_click(main_window, "r", to_display(115))
    _held_key_click(main_window, "p", to_display(100))
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    assert table.item(1, 0).text() == "Peak 1 position (keV)"

    # 70.0 keV = channel 120 under a=10, b=0.5 -- distinct from both the
    # fit's own position (~60 keV / channel 100) and from 70 misread as
    # a channel number, so either kind of conversion bug is detectable.
    table.item(1, 1).setText("70.0")
    table.cellWidget(1, 2).setChecked(True)  # fix "Peak 1 position (keV)"

    main_window.fit_controller.run_fit()

    refit = main_window.spectra[0].fits[-1]
    assert refit.peaks[0].position == pytest.approx(120.0, abs=0.5)
    assert refit.peaks[0].position_err == 0.0  # fixed parameters carry zero uncertainty


def test_fixed_kev_fwhm_with_independent_widths_is_honored_by_a_real_refit(qapp):
    """Same end-to-end guard as test_fixed_kev_position_is_honored_by_a_real_refit,
    for the FWHM/independent-widths path -- the one with a derivative
    reference-position lookup (peak's own position, not a shared or
    wrong one), the specific mistake this feature's history has caught
    multiple times in the display-only direction. This test exercises
    the READ-BACK direction feeding a real fit_peaks() call, which
    prior review found had no coverage at all."""
    from calibration import Calibration

    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))).astype(np.int64)
    y[145:152] += (300 * np.exp(-((np.arange(145, 152) - 148.0) ** 2) / (2 * 2.5 ** 2))).astype(np.int64)
    spectrum = LoadedSpectrum(os.path.join(tempfile.mkdtemp(), "two_peaks.txt"), y, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()

    main_window._calibration = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.001)
    main_window._calibration_active = True
    main_window.calibration_toggle_action.setEnabled(True)
    main_window.calibration_toggle_action.setChecked(True)
    main_window._plot_data()
    main_window.independent_widths_action.setChecked(True)

    to_display = main_window.channel_to_display
    _held_key_click(main_window, "b", to_display(70))
    _held_key_click(main_window, "b", to_display(85))
    _held_key_click(main_window, "b", to_display(160))
    _held_key_click(main_window, "b", to_display(175))
    _held_key_click(main_window, "r", to_display(85))
    _held_key_click(main_window, "r", to_display(160))
    _held_key_click(main_window, "p", to_display(100))
    _held_key_click(main_window, "p", to_display(148))
    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    fit = spectrum.fits[-1]
    cal = main_window._calibration

    # Fix peak 2's FWHM to its own currently-displayed (correctly
    # converted) keV value, unedited -- if the read-back path used peak
    # 1's position (or no position at all) as the derivative reference
    # instead of peak 2's own, this round-trip would land on a visibly
    # different channel FWHM than the fit originally produced.
    peak2_fwhm_kev = table.item(5, 1).text()
    table.cellWidget(5, 2).setChecked(True)  # fix "Peak 2 FWHM (keV)"

    main_window.fit_controller.run_fit()

    refit = spectrum.fits[-1]
    assert refit.peaks[1].fwhm == pytest.approx(fit.peaks[1].fwhm, abs=0.01)
    assert refit.peaks[1].fwhm_err == 0.0  # fixed parameters carry zero uncertainty
    # Sanity: the two peaks' derivatives really do differ enough that a
    # wrong reference position would have produced a detectably wrong
    # channel FWHM, not one that happens to coincide.
    peak1_slope = abs(cal.derivative(fit.peaks[0].position))
    peak2_slope = abs(cal.derivative(fit.peaks[1].position))
    assert peak1_slope != pytest.approx(peak2_slope, rel=0.05)


def test_clear_all_fits_shortcut_deletes_fits(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(_fit_result_with_one_peak())

    main_window.fit_controller.clear_all_fits_action.trigger()

    assert spectrum.fits == []


def test_clear_all_fits_action_has_shortcut(qapp):
    from PySide6.QtGui import QKeySequence
    main_window = MainWindow()
    assert main_window.fit_controller.clear_all_fits_action.shortcut() == QKeySequence("Ctrl+Shift+C")


def test_export_all_fits_action_has_shortcut(qapp):
    from PySide6.QtGui import QKeySequence
    main_window = MainWindow()
    assert main_window.fit_controller.export_all_fits_action.shortcut() == QKeySequence("Ctrl+E")


def test_export_all_fits_shortcut_does_nothing_with_no_fits(qapp, monkeypatch):
    # A plain "must not raise" assertion is too weak here: QFileDialog's
    # modal .exec() never returns under the offscreen Qt platform this
    # suite runs under, so if _export_all_fits()'s no-fits guard ever
    # regresses, an unguarded call would hang the whole test run instead
    # of failing cleanly. Spying on getSaveFileName -- the same
    # monkeypatch target test_export_cancelled_dialog_does_not_write_a_report_file
    # uses above -- proves the dialog path was never reached at all.
    save_dialog_calls = []
    monkeypatch.setattr(
        fit_mode.QFileDialog, "getSaveFileName",
        lambda *a, **k: save_dialog_calls.append(True) or ("", ""),
    )
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    main_window.fit_controller.export_all_fits_action.trigger()

    assert save_dialog_calls == []
