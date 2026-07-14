import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from peak_fit import FitResult, PeakResult, hypermet_left_tail
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


def _make_active_spectrum(main_window):
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (
        500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))
    ).astype(np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
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
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Shared sigma"]


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
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"

    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert table.rowCount() == 3  # amp_0, pos_0, sigma_0 (independent now)
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Peak 1 sigma"]
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
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"
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
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"
    table.item(2, 1).setText("5.0")

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    second = spectrum.fits[1]
    assert second.peaks[0].sigma == 5.0
    assert second.peaks[0].sigma_err == 0.0
    assert second.fixed_params == {"sigma": 5.0}


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
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"

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
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"
    main_window.fit_controller.run_fit()  # second entry, sigma fixed

    main_window.fit_controller.clear()
    assert main_window.fit_controller.state.bg_regions == []

    item = main_window.fit_controller.results_list.item(1)
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
    assert len(main_window.axes.lines) == 4


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

    assert main_window.fit_controller.results_list.count() == 1
    text = main_window.fit_controller.results_list.item(0).text()
    assert "100.00" in text
    assert "5.00" in text


def test_results_panel_updates_when_active_spectrum_changes(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum_a = LoadedSpectrum("a.txt", y, "#1f77b4")
    spectrum_a.active = True
    spectrum_a.fits.append(_fit_result_with_one_peak())
    spectrum_b = LoadedSpectrum("b.txt", y, "#ff7f0e")
    main_window.spectra.extend([spectrum_a, spectrum_b])
    main_window.fit_controller.update_results_list()
    assert main_window.fit_controller.results_list.count() == 1

    main_window._on_active_toggled("b.txt", True)

    assert main_window.fit_controller.results_list.count() == 0


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

    assert main_window.fit_controller.results_list.count() == 0


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
    assert len(main_window.axes.lines) == 4

    # The drawn curve must reflect the tail-aware formula, not a plain
    # Gaussian -- otherwise this test would pass even if the
    # tail_fraction branch in draw_committed_fits were broken/skipped.
    # Line order from draw_committed_fits: [0] spectrum step line,
    # [1] background dashed line, [2] total fit curve, [3] peak-position
    # axvline (one per peak) -- the curve is second-to-last, not last.
    drawn_curve = main_window.axes.lines[-2].get_ydata()
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

    text = main_window.fit_controller.results_list.item(0).text()
    assert "independent widths" in text
    assert "r=0.10" in text
    assert "volume excludes tail" in text
