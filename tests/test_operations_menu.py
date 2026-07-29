import os
import tempfile

import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog, QMenu

from main_window import MainWindow
from spectrum import LoadedSpectrum


def _click(main_window, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process("button_press_event", event)


def _held_key_click(main_window, key_char, xdata, ydata=10.0):
    main_window.fit_controller._held_key = key_char
    _click(main_window, xdata, ydata)
    main_window.fit_controller._held_key = None


def _make_active_spectrum(main_window, path=None):
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


def _commit_a_fit(main_window):
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()


class _FakeFit:
    """Stand-in for a FitResult/IntegrationResult, used where a test only
    cares whether normalize's fit-clearing touches a given spectrum's
    .fits list -- not what a real fit looks like. Every real fit result
    always has `.visible` (peak_fit.FitResult/IntegrationResult both
    default it to True), and draw_committed_fits (fit_mode.py) reads
    that attribute first, before anything else, for every spectrum
    _plot_data redraws -- including spectra a given operation didn't
    touch. `visible = False` here makes that redraw skip this fake
    entry immediately, instead of crashing on the fit-region/background
    fields a bare placeholder (e.g. a plain string) doesn't have."""
    visible = False


def _menu_named(main_window, title):
    # Deliberately not `next(a.menu() for a in main_window.menuBar().actions()
    # if a.text() == title)`: under PySide6 6.8.3, a QMenu fetched via
    # QAction.menu() has its underlying C++ object torn down for real (not
    # just a stale wrapper -- a fresh lookup afterward finds it gone too)
    # once every Python reference to the specific QAction wrapper it came
    # from is dropped, regardless of the QMenu's real Qt parent (the menu
    # bar) still being alive -- and that QAction wrapper doesn't reliably
    # outlive the calling frame (confirmed to still break even when
    # returned from inside a plain for-loop, once that loop itself is
    # nested one call deeper, as it is here from every test function).
    # Walking the widget tree directly with findChildren sidesteps
    # QAction.menu() entirely, so it isn't subject to that lifetime quirk.
    for menu in main_window.findChildren(QMenu):
        if menu.title() == title:
            return menu
    return None


def test_operations_menu_exists_with_calibration_items(qapp):
    main_window = MainWindow()
    titles = [a.text() for a in main_window.menuBar().actions()]
    assert "&Operations" in titles

    operations_menu = _menu_named(main_window, "&Operations")
    item_texts = [a.text() for a in operations_menu.actions() if not a.isSeparator()]
    # Prefix check, not exact-equality: this menu is designed to grow (Multiply/
    # Rebin/Normalize are appended after the separator by later tasks), so this
    # only pins down that the calibration items still lead, in order.
    assert item_texts[:2] == ["Calibration...", "Toggle Calibration Active"]


def test_calibration_no_longer_under_view(qapp):
    main_window = MainWindow()
    view_menu = _menu_named(main_window, "&View")
    view_texts = [a.text() for a in view_menu.actions() if not a.isSeparator()]
    assert "Calibration..." not in view_texts


def test_view_menu_still_has_its_other_items(qapp):
    main_window = MainWindow()
    view_menu = _menu_named(main_window, "&View")
    view_texts = [a.text() for a in view_menu.actions() if not a.isSeparator()]
    assert view_texts == ["Log scale Y", "Spectra", "Dark theme"]


def test_new_shortcuts_are_set(qapp):
    main_window = MainWindow()
    assert main_window.exit_action.shortcut() == QKeySequence("Ctrl+Q")
    assert main_window.calibration_action.shortcut() == QKeySequence("Ctrl+L")
    assert main_window.calibration_active_menu_action.shortcut() == QKeySequence("Ctrl+T")
    assert main_window.dark_theme_action.shortcut() == QKeySequence("Ctrl+D")
    assert main_window.log_scale_action.shortcut() == QKeySequence("Ctrl+G")
    assert main_window.toggle_spectrum_panel_action.shortcut() == QKeySequence("Ctrl+1")
    assert main_window.zoom_in_action.shortcut() == QKeySequence("Ctrl+=")
    assert main_window.zoom_out_action.shortcut() == QKeySequence("Ctrl+-")
    assert main_window.full_spectrum_action.shortcut() == QKeySequence("Ctrl+0")
    assert main_window.fit_controller.toggle_results_panel_action.shortcut() == QKeySequence("Ctrl+2")
    assert main_window.fit_controller.toggle_parameters_panel_action.shortcut() == QKeySequence("Ctrl+3")


def test_toggle_calibration_active_menu_item_syncs_with_toolbar_button(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    main_window._apply_calibration_change(Calibration(kind="linear", a=1.0, b=1.0), True)

    assert main_window.calibration_active_menu_action.isChecked() is True
    assert main_window.calibration_toggle_action.isChecked() is True

    main_window.calibration_active_menu_action.setChecked(False)

    assert main_window.calibration_toggle_action.isChecked() is False
    assert main_window._calibration_active is False


def test_toggle_calibration_active_menu_item_disabled_with_no_calibration(qapp):
    main_window = MainWindow()
    assert main_window.calibration_active_menu_action.isEnabled() is False


def test_apply_multiply_scales_data(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._apply_multiply(spectrum, 2.0)

    assert spectrum.data[0] == 40  # baseline 20 * 2


def test_apply_multiply_deletes_fits_and_resets_marks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    _commit_a_fit(main_window)
    assert len(spectrum.fits) == 1
    _held_key_click(main_window, "b", 70)

    main_window._apply_multiply(spectrum, 2.0)

    assert spectrum.fits == []
    assert main_window.fit_controller.state.pending_bg_click is None


def test_apply_multiply_preserves_the_current_view(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    spectrum = main_window.spectra[0]
    main_window.axes.set_xlim(10, 50)

    main_window._apply_multiply(spectrum, 2.0)

    assert main_window.axes.get_xlim() == (10.0, 50.0)


def test_apply_multiply_leaves_other_spectra_untouched(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    original_b = spectrum_b.data.copy()

    main_window._apply_multiply(spectrum_a, 2.0)

    assert list(spectrum_b.data) == list(original_b)


def test_open_multiply_dialog_applies_the_entered_factor(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    def fake_exec(self):
        self.result_factor = 3.0
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_multiply_dialog()

    assert spectrum.data[0] == 60


def test_open_multiply_dialog_does_nothing_when_cancelled(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    original = spectrum.data.copy()

    def fake_exec(self):
        self.result_factor = None
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_multiply_dialog()

    assert list(spectrum.data) == list(original)


def test_open_multiply_dialog_does_nothing_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    main_window._open_multiply_dialog()  # must not raise


def test_multiply_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.multiply_action.isEnabled() is False


def test_multiply_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.multiply_action.isEnabled() is True


def test_multiply_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.multiply_action.shortcut() == QKeySequence("Ctrl+M")


def test_multiply_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.multiply_action in main_window.operations_menu.actions()


def test_apply_rebin_reduces_channel_count_and_sums(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

    main_window._apply_rebin(spectrum, 2)

    assert list(spectrum.data) == [3, 7, 11]


def test_apply_rebin_adjusts_linear_calibration(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)
    main_window._calibration = Calibration(kind="linear", a=1.0, b=2.0)

    main_window._apply_rebin(spectrum, 2)

    assert main_window._calibration.a == 1.0
    assert main_window._calibration.b == 4.0


def test_apply_rebin_adjusts_quadratic_calibration(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)
    main_window._calibration = Calibration(kind="quadratic", a=1.0, b=2.0, c=0.5)

    main_window._apply_rebin(spectrum, 3)

    assert main_window._calibration.a == 1.0
    assert main_window._calibration.b == 6.0
    assert main_window._calibration.c == 4.5


def test_apply_rebin_with_no_calibration_leaves_it_none(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._apply_rebin(spectrum, 2)

    assert main_window._calibration is None


def test_apply_rebin_deletes_fits_and_resets_marks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    _commit_a_fit(main_window)
    assert len(spectrum.fits) == 1
    _held_key_click(main_window, "b", 70)

    main_window._apply_rebin(spectrum, 2)

    assert spectrum.fits == []
    assert main_window.fit_controller.state.pending_bg_click is None


def test_apply_rebin_does_not_preserve_the_current_view(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window.axes.set_xlim(10, 50)

    main_window._apply_rebin(spectrum, 2)

    assert main_window.axes.get_xlim() != (10.0, 50.0)


def test_apply_rebin_warns_when_other_spectra_are_loaded(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=1.0, b=2.0)

    main_window._apply_rebin(spectrum_a, 2)

    assert main_window._calibration.b == 4.0
    message = main_window.statusBar().currentMessage()
    assert "other loaded spectra" in message.lower()


def test_apply_rebin_no_warning_with_only_one_spectrum_loaded(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=1.0, b=2.0)

    main_window._apply_rebin(spectrum, 2)

    message = main_window.statusBar().currentMessage()
    assert "other loaded spectra" not in message.lower()


def test_apply_rebin_warning_survives_a_mouse_move(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=1.0, b=2.0)

    main_window._apply_rebin(spectrum_a, 2)

    ax = main_window.axes
    px, py = ax.transData.transform((10.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window.canvas.callbacks.process("motion_notify_event", event)

    message = main_window.statusBar().currentMessage()
    assert "other loaded spectra" in message.lower()


def test_open_rebin_dialog_applies_the_entered_factor(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)

    def fake_exec(self):
        self.result_factor = 2
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_rebin_dialog()

    assert list(spectrum.data) == [3, 7]


def test_open_rebin_dialog_does_nothing_when_cancelled(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    original = spectrum.data.copy()

    def fake_exec(self):
        self.result_factor = None
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_rebin_dialog()

    assert list(spectrum.data) == list(original)


def test_open_rebin_dialog_does_nothing_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    main_window._open_rebin_dialog()  # must not raise


def test_rebin_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.rebin_action.isEnabled() is False


def test_rebin_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.rebin_action.isEnabled() is True


def test_rebin_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.rebin_action.shortcut() == QKeySequence("Ctrl+R")


def test_rebin_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.rebin_action in main_window.operations_menu.actions()


def test_normalize_disabled_with_fewer_than_two_visible_spectra(qapp):
    main_window = MainWindow()
    assert main_window.normalize_action.isEnabled() is False
    _make_active_spectrum(main_window)
    assert main_window.normalize_action.isEnabled() is False


def test_normalize_enabled_with_two_visible_spectra(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    assert main_window.normalize_action.isEnabled() is True


def test_normalize_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.normalize_action.shortcut() == QKeySequence("Ctrl+N")


def test_normalize_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.normalize_action in main_window.operations_menu.actions()


def test_normalize_with_no_marks_shows_a_status_message(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)

    main_window._normalize_spectra()

    assert "mark" in main_window.statusBar().currentMessage().lower()


def test_normalize_no_marks_message_survives_a_mouse_move(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)

    main_window._normalize_spectra()

    ax = main_window.axes
    px, py = ax.transData.transform((10.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window.canvas.callbacks.process("motion_notify_event", event)

    assert "mark" in main_window.statusBar().currentMessage().lower()


def test_normalize_single_marker_scales_by_bin_count(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20, 30], dtype=np.int64)
    spectrum_b.data = np.array([5, 40, 15], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_a.data) == [20, 40, 60]
    assert list(spectrum_b.data) == [5, 40, 15]


def test_normalize_region_marker_scales_by_area(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20, 30, 5], dtype=np.int64)
    spectrum_b.data = np.array([5, 10, 15, 5], dtype=np.int64)
    main_window.fit_controller.state.fit_region = (1, 2)

    main_window._normalize_spectra()

    assert list(spectrum_a.data) == [10, 20, 30, 5]
    assert list(spectrum_b.data) == [10, 20, 30, 10]


def test_normalize_skips_a_zero_reference_spectrum_with_a_message(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([0, 0], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_b.data) == [0, 0]
    message = main_window.statusBar().currentMessage()
    assert os.path.basename(spectrum_b.path) in message


def test_normalize_skipped_message_survives_a_mouse_move(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([0, 0], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    ax = main_window.axes
    px, py = ax.transData.transform((10.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window.canvas.callbacks.process("motion_notify_event", event)

    message = main_window.statusBar().currentMessage()
    assert os.path.basename(spectrum_b.path) in message


def test_normalize_all_zero_reference_values_shows_a_message_instead_of_silently_doing_nothing(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([0, 0], dtype=np.int64)
    spectrum_b.data = np.array([0, 0], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_a.data) == [0, 0]
    assert list(spectrum_b.data) == [0, 0]
    message = main_window.statusBar().currentMessage()
    assert "nothing to normalize" in message.lower()
    assert main_window.fit_controller.state.pending_fit_click is None


def test_normalize_all_zero_reference_values_replots_after_resetting_marks(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    main_window.spectra[0].data = np.zeros_like(main_window.spectra[0].data)
    spectrum_b.data = np.zeros_like(spectrum_b.data)
    # A real click (not just setting state.pending_fit_click directly) so an
    # actual mark artist gets drawn on the canvas -- reset_marks() detaches
    # it from the axes synchronously either way (artist.remove() doesn't
    # need a redraw to update axes.lines), so checking _progress_artists or
    # axes.lines alone can't tell a replot apart from no replot at all; both
    # already read empty/removed the instant reset_marks() runs. What can't
    # happen without an actual replot is nav_toolbar.push_current(), called
    # unconditionally near the end of _plot_data() (see main_window.py) and
    # nothing else on this path -- spying on it is the same proxy
    # test_reset_marks_clears_progress_without_touching_fits_or_replotting
    # (test_fit_mode_ui.py) uses for this exact "was a replot triggered"
    # question.
    _held_key_click(main_window, "r", 10)
    assert main_window.fit_controller._progress_artists != []

    replot_calls = []
    monkeypatch.setattr(
        main_window.nav_toolbar, "push_current", lambda: replot_calls.append(True)
    )

    main_window._normalize_spectra()

    assert replot_calls != []
    assert main_window.fit_controller._progress_artists == []


def test_normalize_clears_fits_only_on_rescaled_spectra(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([5, 40], dtype=np.int64)
    fake_fit_a = _FakeFit()
    fake_fit_b = _FakeFit()
    spectrum_a.fits.append(fake_fit_a)
    spectrum_b.fits.append(fake_fit_b)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert spectrum_a.fits == []
    assert spectrum_b.fits == [fake_fit_b]


def test_normalize_resets_in_progress_marks(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert main_window.fit_controller.state.pending_fit_click is None


def test_normalize_ignores_hidden_spectra(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_c = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([5, 40], dtype=np.int64)
    spectrum_c.data = np.array([1, 1000], dtype=np.int64)
    spectrum_c.visible = False
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_c.data) == [1, 1000]
