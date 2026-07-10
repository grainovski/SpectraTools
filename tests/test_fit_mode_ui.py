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
