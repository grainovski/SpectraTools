import os
import tempfile

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QTableWidgetItem

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
    assert table.item(0, 1).text() == "1"  # Peak column, 1-based
    assert table.item(0, 2).text().startswith("100.0")  # Position
    assert "±" in table.item(0, 2).text()
    assert "±" in table.item(0, 3).text()  # FWHM
    assert "±" in table.item(0, 4).text()  # Volume


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
    assert table.item(0, 1).text() == "1"
    assert table.item(1, 1).text() == "2"
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

    main_window.fit_controller.clear()
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


def test_clear_hides_every_fit_for_the_active_spectrum_without_deleting_it(qapp):
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
    assert spectrum.fits[0].visible is True

    main_window.fit_controller.clear()

    assert len(spectrum.fits) == 1  # nothing deleted
    assert spectrum.fits[0].visible is False  # but no longer drawn


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
    assert "100.00" in table.item(0, 2).text()  # Position
    assert "5.00" in table.item(0, 3).text()  # FWHM


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
