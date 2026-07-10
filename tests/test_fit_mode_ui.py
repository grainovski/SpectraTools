import numpy as np
from matplotlib.backend_bases import MouseEvent

from main_window import MainWindow
from spectrum import LoadedSpectrum


def _dispatch(main_window, name, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent(name, main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process(name, event)


def _drag(main_window, x_start, x_end, y=10.0):
    _dispatch(main_window, "button_press_event", x_start, y)
    _dispatch(main_window, "button_release_event", x_end, y)


def _click(main_window, x, y=10.0):
    _drag(main_window, x, x, y)


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


def test_toggling_fit_mode_disables_nav_toolbar(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    assert main_window.nav_toolbar.isEnabled() is True
    main_window.fit_mode_action.setChecked(True)
    assert main_window.fit_controller.enabled is True
    assert main_window.nav_toolbar.isEnabled() is False

    main_window.fit_mode_action.setChecked(False)
    assert main_window.nav_toolbar.isEnabled() is True


def test_full_fit_flow_commits_a_fit_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)

    _drag(main_window, 70, 85)    # left background region
    _drag(main_window, 115, 130)  # right background region
    _drag(main_window, 85, 115)   # fit region
    assert main_window.fit_button.isEnabled() is False
    _click(main_window, 100)      # peak position
    assert main_window.fit_button.isEnabled() is True

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert len(result.peaks) == 1
    assert abs(result.peaks[0].position - 100.0) < 1.0


def test_clear_discards_in_progress_marks_without_committing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)
    _drag(main_window, 70, 85)
    _drag(main_window, 115, 130)
    _drag(main_window, 85, 115)
    _click(main_window, 100)

    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.step == "left_bg"
    assert main_window.fit_button.isEnabled() is False
    assert len(spectrum.fits) == 0


def test_fit_mode_disabled_when_active_spectrum_hidden(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    spectrum.visible = False
    main_window._update_fit_mode_availability()

    assert main_window.fit_mode_action.isEnabled() is False


def test_fit_failure_leaves_marks_intact_and_shows_message(qapp):
    # Fit region deliberately far too narrow (1 data point) for the 2
    # peaks marked in it -- fit_peaks() raises FitError for "too few
    # data points", which run_fit() must catch, show as a status-bar
    # message, and NOT commit a result or reset the in-progress marks.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)
    _drag(main_window, 70, 85)
    _drag(main_window, 115, 130)
    _drag(main_window, 99.5, 100.5)
    _click(main_window, 100)
    _click(main_window, 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 0
    assert main_window.fit_controller.state.step == "marking_peaks"
    assert main_window.fit_controller.state.peak_positions == [100.0, 100.0]
    assert main_window.statusBar().currentMessage() != ""


def test_status_message_not_immediately_clobbered_by_mouse_move(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)
    # Zero-width click while awaiting the first region drag (state is
    # still STEP_LEFT_BG right after enabling fit mode) -- triggers the
    # "Drag to select a region" hint via the zero-width-drag check in
    # on_release.
    _click(main_window, 100)

    message_after_hint = main_window.statusBar().currentMessage()
    assert message_after_hint != ""

    # Simulate a mouse-move immediately afterward, as would happen in real
    # use -- the hint must survive this, not get instantly overwritten by
    # the hover readout that _on_mouse_move normally shows.
    _dispatch(main_window, "motion_notify_event", 50)

    assert main_window.statusBar().currentMessage() == message_after_hint


from peak_fit import FitResult, PeakResult


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

    # 3 shaded spans (left bg, right bg, fit region) = 3 patches;
    # spectrum data line + background dashed line + fitted curve +
    # 1 peak marker = 4 lines.
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
    # Reproduces a real crash: mark one region (creating an in-progress
    # artist), then something else triggers a full axes.clear() (here,
    # simulated directly -- in the real app this happens via the results
    # panel's "Remove Fit"/"Clear All Fits" context menu calling
    # _plot_data() while a fit is still mid-marking) before the
    # in-progress fit is finished. Finishing it afterward (Clear, in
    # this test) must not raise NotImplementedError.
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)
    _drag(main_window, 70, 85)  # marks the left background region

    # Something unrelated triggers a full replot, invalidating the
    # in-progress artist's _remove_method.
    main_window._plot_data()

    # Must not raise.
    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.step == "left_bg"
