import os
import tempfile

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog, QMenu

import fit_mode
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


def test_apply_multiply_shows_a_warning_instead_of_crashing_on_overflow(qapp, monkeypatch):
    # A factor like 1e20 is finite and > 0, so it passes the dialog's
    # own validation (see Task 4's Multiply-by-Factor tests above) --
    # but still overflows int64 one layer down, in spectrum_operations.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    original = spectrum.data.copy()
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda *a, **k: warnings.append(a) or None,
    )

    main_window._apply_multiply(spectrum, 1e20)  # must not raise

    assert len(warnings) == 1
    assert list(spectrum.data) == list(original)  # left untouched, not corrupted


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


def test_open_multiply_dialog_validator_rejects_infinite_factor(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    captured = {}

    def fake_exec(self):
        captured["validate"] = self._validate
        self.result_factor = None
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_multiply_dialog()

    validate = captured["validate"]
    assert validate(float("inf")) is not None
    assert validate(float("-inf")) is not None
    assert validate(1e20) is None  # still a legal finite factor, even if large -- only inf/nan are rejected


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

    # New channel k is the sum of old channels 2k and 2k+1, so it sits at
    # old channel 2k + 0.5 -- the CENTRE of the group. The constant term
    # therefore moves to the old energy at channel 0.5, not at 0. It used
    # to stay at a=1.0, which left every rebinned spectrum's energy axis
    # low by half a channel (3.5 channels at factor 8).
    assert main_window._calibration.a == pytest.approx(1.0 + 2.0 * 0.5)
    assert main_window._calibration.b == 4.0


def test_apply_rebin_adjusts_quadratic_calibration(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)
    main_window._calibration = Calibration(kind="quadratic", a=1.0, b=2.0, c=0.5)

    main_window._apply_rebin(spectrum, 3)

    # Factor 3 groups old channels [3k, 3k+2], centred at 3k + 1.
    offset = 1.0
    assert main_window._calibration.a == pytest.approx(1.0 + 2.0 * offset + 0.5 * offset ** 2)
    assert main_window._calibration.b == pytest.approx(3 * (2.0 + 2 * 0.5 * offset))
    assert main_window._calibration.c == pytest.approx(4.5)


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


def test_apply_add_produces_summed_data(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20, 30], dtype=np.int64)
    spectrum_b.data = np.array([1, 2, 3], dtype=np.int64)

    main_window._apply_add(spectrum_a, spectrum_b, 2.0)

    assert list(main_window.spectra[-1].data) == [12, 24, 36]


def test_apply_add_makes_the_result_active_and_deactivates_the_rest(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = True
    spectrum_b.active = False

    main_window._apply_add(spectrum_a, spectrum_b, 1.0)

    result = main_window.spectra[-1]
    assert result.active is True
    assert spectrum_a.active is False
    assert spectrum_b.active is False


def test_apply_add_names_result_with_basenames_when_factor_is_one(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path="/fake/dir/a.spe")
    spectrum_b = _make_active_spectrum(main_window, path="/fake/other/b.spe")

    main_window._apply_add(spectrum_a, spectrum_b, 1.0)

    # Anchored at Spectrum A's directory (see the auto-log-location test
    # below) -- the label itself is still just the two basenames.
    assert main_window.spectra[-1].path == os.path.join("/fake/dir", "a.spe + b.spe")


def test_apply_add_names_result_with_factor_when_not_one(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path="/fake/dir/a.spe")
    spectrum_b = _make_active_spectrum(main_window, path="/fake/other/b.spe")

    main_window._apply_add(spectrum_a, spectrum_b, 2.5)

    assert main_window.spectra[-1].path == os.path.join("/fake/dir", "a.spe + 2.5xb.spe")


def test_apply_add_shows_a_warning_instead_of_crashing_on_overflow(qapp, monkeypatch):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_count_before = len(main_window.spectra)
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda *a, **k: warnings.append(a) or None,
    )

    main_window._apply_add(spectrum_a, spectrum_b, 1e20)  # must not raise

    assert len(warnings) == 1
    # No garbage spectrum was added to the list on failure.
    assert len(main_window.spectra) == spectrum_count_before


def test_apply_subtract_produces_subtracted_data(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20, 30], dtype=np.int64)
    spectrum_b.data = np.array([1, 2, 3], dtype=np.int64)

    main_window._apply_subtract(spectrum_a, spectrum_b, 2.0)

    assert list(main_window.spectra[-1].data) == [8, 16, 24]


def test_apply_subtract_makes_the_result_active_and_deactivates_the_rest(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = True
    spectrum_b.active = False

    main_window._apply_subtract(spectrum_a, spectrum_b, 1.0)

    result = main_window.spectra[-1]
    assert result.active is True
    assert spectrum_a.active is False
    assert spectrum_b.active is False


def test_apply_subtract_names_result_with_basenames_when_factor_is_one(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path="/fake/dir/a.spe")
    spectrum_b = _make_active_spectrum(main_window, path="/fake/other/b.spe")

    main_window._apply_subtract(spectrum_a, spectrum_b, 1.0)

    assert main_window.spectra[-1].path == os.path.join("/fake/dir", "a.spe - b.spe")


def test_apply_subtract_names_result_with_factor_when_not_one(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path="/fake/dir/a.spe")
    spectrum_b = _make_active_spectrum(main_window, path="/fake/other/b.spe")

    main_window._apply_subtract(spectrum_a, spectrum_b, 2.5)

    assert main_window.spectra[-1].path == os.path.join("/fake/dir", "a.spe - 2.5xb.spe")


def test_apply_subtract_shows_a_warning_instead_of_crashing_on_overflow(qapp, monkeypatch):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_count_before = len(main_window.spectra)
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda *a, **k: warnings.append(a) or None,
    )

    main_window._apply_subtract(spectrum_a, spectrum_b, 1e20)  # must not raise

    assert len(warnings) == 1
    assert len(main_window.spectra) == spectrum_count_before


def test_apply_add_result_integration_auto_logs_next_to_spectrum_a_directory(qapp, tmp_path):
    """Regression guard: integrating on a spectrum produced by Add must
    auto-log next to Spectrum A's real file, not the process's cwd.
    _apply_add used to name the combined spectrum with bare basenames
    only (e.g. "a.spe + b.spe"), which has no directory of its own --
    fit_export.auto_log_path's empty os.path.dirname sent every
    auto-log write to the process's cwd instead. The dirname assertion
    is checked before run_integration() executes, so a failing (RED)
    run never writes a stray log file outside tmp_path."""
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path=str(tmp_path / "a.spe"))
    spectrum_b = _make_active_spectrum(main_window)  # a different (temp) directory

    main_window._apply_add(spectrum_a, spectrum_b, 1.0)

    added = main_window.spectra[-1]
    from fit_export import auto_log_path
    log_path = auto_log_path(added.path)
    assert os.path.dirname(log_path) == str(tmp_path)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    assert len(added.fits) == 1
    assert os.path.exists(log_path)


def test_export_fits_default_directory_resolves_for_an_added_spectrum(qapp, monkeypatch, tmp_path):
    """fit_mode.py's _export_fits (the "Export Fit Report" dialog) reads
    os.path.dirname(active.path) for its default save directory -- the
    exact same pattern as fit_export.auto_log_path. Proves it inherits
    the fix above with no changes of its own needed: once _apply_add
    anchors the combined spectrum's .path at Spectrum A's directory,
    the export dialog defaults there too instead of the process's cwd."""
    # This test only cares about _export_fits' default directory, not the
    # auto-log file run_integration() writes as an unrelated side effect
    # -- same no-op technique test_matrix_panel_and_main_window_integrate_identically
    # uses to keep that side effect out of the repo.
    monkeypatch.setattr(fit_mode.fit_export, "append_auto_log", lambda *a, **k: None)

    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path=str(tmp_path / "a.spe"))
    spectrum_b = _make_active_spectrum(main_window)  # a different (temp) directory

    main_window._apply_add(spectrum_a, spectrum_b, 1.0)
    added = main_window.spectra[-1]

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()
    assert len(added.fits) == 1

    captured = []
    monkeypatch.setattr(
        fit_mode.QFileDialog, "getSaveFileName",
        lambda *a, **k: captured.append(a) or ("", ""),
    )
    main_window.fit_controller._export_fits(added, [0])

    default_path = captured[0][2]
    assert os.path.dirname(default_path) == str(tmp_path)


def test_add_combined_spectrum_disambiguates_a_colliding_path(qapp):
    # Regression test: _add_combined_spectrum is the one call site that
    # inserts into self.spectra without a uniqueness guard on .path (unlike
    # _load_files' `if any(s.path == path ...): continue`). A collision used
    # to leave two spectra sharing one .path, which made _remove_spectrum
    # (matches by `s.path != path`, no break) delete both at once and
    # _on_active_toggled mark both active simultaneously. Repeating the same
    # Add with the same inputs/factor is the ordinary way a user hits this.
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window, path="/fake/dir/a.spe")
    spectrum_b = _make_active_spectrum(main_window, path="/fake/other/b.spe")

    main_window._apply_add(spectrum_a, spectrum_b, 1.0)
    main_window._apply_add(spectrum_a, spectrum_b, 1.0)

    first_result, second_result = main_window.spectra[-2], main_window.spectra[-1]
    assert first_result.path != second_result.path


def test_add_action_disabled_with_fewer_than_two_spectra(qapp):
    main_window = MainWindow()
    assert main_window.add_action.isEnabled() is False
    _make_active_spectrum(main_window)
    assert main_window.add_action.isEnabled() is False


def test_add_action_enabled_with_two_spectra(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    assert main_window.add_action.isEnabled() is True


def test_add_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.add_action.shortcut() == QKeySequence("Ctrl+A")


def test_add_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.add_action in main_window.operations_menu.actions()


def test_subtract_action_disabled_with_fewer_than_two_spectra(qapp):
    main_window = MainWindow()
    assert main_window.subtract_action.isEnabled() is False
    _make_active_spectrum(main_window)
    assert main_window.subtract_action.isEnabled() is False


def test_subtract_action_enabled_with_two_spectra(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    assert main_window.subtract_action.isEnabled() is True


def test_subtract_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.subtract_action.shortcut() == QKeySequence("Ctrl+Shift+A")


def test_subtract_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.subtract_action in main_window.operations_menu.actions()


def test_open_add_dialog_applies_the_chosen_spectra_and_factor(qapp, monkeypatch):
    from combine_dialog import CombineDialog
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([1, 2], dtype=np.int64)

    def fake_exec(self):
        self.result_spectrum_a = spectrum_a
        self.result_spectrum_b = spectrum_b
        self.result_factor = 3.0
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CombineDialog, "exec", fake_exec)
    main_window._open_add_dialog()

    assert list(main_window.spectra[-1].data) == [13, 26]


def test_open_add_dialog_does_nothing_when_cancelled(qapp, monkeypatch):
    from combine_dialog import CombineDialog
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)

    def fake_exec(self):
        self.result_spectrum_a = None
        self.result_spectrum_b = None
        self.result_factor = None
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(CombineDialog, "exec", fake_exec)
    main_window._open_add_dialog()

    assert len(main_window.spectra) == 2


def test_open_add_dialog_does_nothing_with_fewer_than_two_spectra(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._open_add_dialog()  # must not raise
    assert len(main_window.spectra) == 1


def test_open_subtract_dialog_applies_the_chosen_spectra_and_factor(qapp, monkeypatch):
    from combine_dialog import CombineDialog
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([1, 2], dtype=np.int64)

    def fake_exec(self):
        self.result_spectrum_a = spectrum_a
        self.result_spectrum_b = spectrum_b
        self.result_factor = 3.0
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CombineDialog, "exec", fake_exec)
    main_window._open_subtract_dialog()

    assert list(main_window.spectra[-1].data) == [7, 14]


def test_open_subtract_dialog_does_nothing_when_cancelled(qapp, monkeypatch):
    from combine_dialog import CombineDialog
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)

    def fake_exec(self):
        self.result_spectrum_a = None
        self.result_spectrum_b = None
        self.result_factor = None
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(CombineDialog, "exec", fake_exec)
    main_window._open_subtract_dialog()

    assert len(main_window.spectra) == 2


def test_open_subtract_dialog_does_nothing_with_fewer_than_two_spectra(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._open_subtract_dialog()  # must not raise
    assert len(main_window.spectra) == 1


def test_save_spectrum_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.save_spectrum_action.isEnabled() is False


def test_save_spectrum_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.save_spectrum_action.isEnabled() is True


def test_save_spectrum_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.save_spectrum_action.shortcut() == QKeySequence("Ctrl+S")


def test_save_spectrum_action_in_file_menu_before_recent_files(qapp):
    main_window = MainWindow()
    actions = main_window.file_menu.actions()
    assert main_window.save_spectrum_action in actions
    save_index = actions.index(main_window.save_spectrum_action)
    recent_index = actions.index(main_window.recent_menu.menuAction())
    assert save_index < recent_index


def test_write_spectrum_extension_takes_priority_over_chosen_filter(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    txt_path = tmp_path / "out.txt"
    main_window._write_spectrum(spectrum, str(txt_path), "SPE files (*.spe)")

    from histogram_io import load_histogram
    assert list(load_histogram(str(txt_path))[:len(spectrum.data)]) == list(spectrum.data)


def test_write_spectrum_matching_extension_is_not_doubled(qapp, tmp_path):
    # Guards against a "foo.spe" + SPE filter turning into "foo.spe.spe":
    # a path that already carries the extension the chosen filter implies
    # must be written completely unchanged.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out.spe"
    main_window._write_spectrum(spectrum, str(path), "SPE files (*.spe)")

    assert os.path.exists(str(path))
    assert not os.path.exists(str(path) + ".spe")


def test_write_spectrum_falls_back_to_the_chosen_filter_with_no_recognized_extension(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out"
    main_window._write_spectrum(spectrum, str(path), "SPK files (*.spk)")

    from spk_io import load_spk
    assert not path.exists()
    assert list(load_spk(str(path) + ".spk")) == list(spectrum.data)


def test_write_spectrum_text_filter_fallback_appends_txt_extension(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out"
    main_window._write_spectrum(spectrum, str(path), "Text files (*.txt)")

    from histogram_io import load_histogram
    assert not path.exists()
    assert list(load_histogram(str(path) + ".txt")[:len(spectrum.data)]) == list(spectrum.data)


def test_write_spectrum_defaults_to_spe_with_no_extension_or_recognized_filter(qapp, tmp_path):
    # "All files (*)" (or any other unrecognized filter string) falls back
    # to .spe rather than .txt: SPE is both the first-listed filter and
    # -- since _open_save_spectrum_dialog never passes a selectedFilter --
    # the Save dialog's actual pre-selected default, so it's the more
    # consistent "no clearly chosen format" default than an arbitrary
    # second special case for Text.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out"
    main_window._write_spectrum(spectrum, str(path), "All files (*)")

    from spe_io import load_spe
    assert not path.exists()
    assert list(load_spe(str(path) + ".spe")) == list(spectrum.data)


def test_write_spectrum_spe_extension_dispatches_to_save_spe(qapp, tmp_path):
    # A recognized-but-mismatched extension (path says .spe, filter says
    # SPK) is left completely alone: the extension already on the path
    # wins over the chosen filter, both for the writer used and for the
    # filename itself (no forced rename to match the filter).
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out.spe"
    main_window._write_spectrum(spectrum, str(path), "SPK files (*.spk)")

    from spe_io import load_spe
    assert list(load_spe(str(path))) == list(spectrum.data)


def test_write_spectrum_spe_filter_fallback_dispatches_to_save_spe(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out"
    main_window._write_spectrum(spectrum, str(path), "SPE files (*.spe)")

    from spe_io import load_spe
    assert not path.exists()
    assert list(load_spe(str(path) + ".spe")) == list(spectrum.data)


def test_write_spectrum_with_bare_path_reloads_correctly_through_extension_dispatch(qapp, tmp_path):
    # Regression test for the underlying bug: QFileDialog.getSaveFileName()
    # has no setDefaultSuffix equivalent, so on Qt's own cross-platform
    # Save dialog (what Linux gets without native GTK auto-suffixing) a
    # bare filename previously stayed bare on disk. _try_load_spectrum
    # dispatches purely by extension, so a bare binary SPE file would
    # silently fall through to the plain-text histogram loader on the next
    # open. Round-trip through the real write/load pair to prove the fix
    # closes that gap end-to-end, not just at the file-naming level.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    bare_path = str(tmp_path / "myspectrum")
    main_window._write_spectrum(spectrum, bare_path, "SPE files (*.spe)")

    assert not os.path.exists(bare_path)
    assert os.path.exists(bare_path + ".spe")

    reloaded, error, _calibration = main_window._try_load_spectrum(bare_path + ".spe")
    assert error is None
    assert list(reloaded.data) == list(spectrum.data)


def test_write_spectrum_shows_a_warning_instead_of_crashing_on_failure(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda *a, **k: warnings.append(a) or None,
    )

    def failing_writer(path, data):
        raise ValueError("simulated failure")

    monkeypatch.setattr("main_window.save_histogram", failing_writer)
    path = tmp_path / "out.txt"

    main_window._write_spectrum(spectrum, str(path), "Text files (*.txt)")  # must not raise

    assert len(warnings) == 1


def test_write_spectrum_gives_actionable_message_when_spk_encoding_overflows(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([0, 10 ** 14])
    warnings = []
    monkeypatch.setattr(
        "main_window.QMessageBox.warning",
        lambda *a, **k: warnings.append(a) or None,
    )

    path = tmp_path / "out.spk"
    main_window._write_spectrum(spectrum, str(path), "SPK files (*.spk)")  # must not raise

    assert len(warnings) == 1
    message = warnings[0][2]
    assert "smaller factor" in message.lower()


def test_open_save_spectrum_dialog_writes_the_chosen_file(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    target = tmp_path / "chosen.spk"

    monkeypatch.setattr(
        "main_window.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(target), "SPK files (*.spk)"),
    )
    main_window._open_save_spectrum_dialog()

    from spk_io import load_spk
    assert list(load_spk(str(target))) == list(spectrum.data)


def test_open_save_spectrum_dialog_does_nothing_when_cancelled(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    monkeypatch.setattr("main_window.QFileDialog.getSaveFileName", lambda *a, **k: ("", ""))
    main_window._open_save_spectrum_dialog()  # must not raise


def test_open_save_spectrum_dialog_does_nothing_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    main_window._open_save_spectrum_dialog()  # must not raise


def test_remove_spectrum_removes_only_the_matching_path(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = False
    spectrum_b.active = True

    main_window._remove_spectrum(spectrum_a.path)

    assert main_window.spectra == [spectrum_b]
    assert spectrum_b.active is True


def test_remove_spectrum_promotes_another_to_active_when_the_active_one_is_removed(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = True
    spectrum_b.active = False

    main_window._remove_spectrum(spectrum_a.path)

    assert main_window.spectra == [spectrum_b]
    assert spectrum_b.active is True


def test_remove_spectrum_leaves_the_program_empty_when_it_was_the_only_one(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._remove_spectrum(spectrum.path)

    assert main_window.spectra == []


def test_close_active_spectrum_removes_the_active_one(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = False
    spectrum_b.active = True

    main_window._close_active_spectrum()

    assert main_window.spectra == [spectrum_a]
    assert spectrum_a.active is True


def test_close_active_spectrum_does_nothing_with_no_spectra_loaded(qapp):
    main_window = MainWindow()
    main_window._close_active_spectrum()  # must not raise
    assert main_window.spectra == []


def test_close_spectrum_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.close_spectrum_action.isEnabled() is False


def test_close_spectrum_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.close_spectrum_action.isEnabled() is True


def test_close_spectrum_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.close_spectrum_action.shortcut() == QKeySequence("Ctrl+W")


def test_close_spectrum_action_in_file_menu_between_save_and_recent_files(qapp):
    main_window = MainWindow()
    actions = main_window.file_menu.actions()
    assert main_window.close_spectrum_action in actions
    save_index = actions.index(main_window.save_spectrum_action)
    close_index = actions.index(main_window.close_spectrum_action)
    recent_index = actions.index(main_window.recent_menu.menuAction())
    assert save_index < close_index < recent_index


def test_toggling_spectrum_visibility_preserves_the_current_view(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    main_window.axes.set_xlim(10, 50)

    main_window._on_show_toggled(spectrum_b.path, False)

    assert main_window.axes.get_xlim() == (10.0, 50.0)


def test_toggling_the_last_visible_spectrum_off_does_not_crash(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._on_show_toggled(spectrum.path, False)  # must not raise

    assert spectrum.visible is False


def test_remove_spectrum_preserves_the_current_view(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    main_window.axes.set_xlim(10, 50)

    main_window._remove_spectrum(spectrum_a.path)

    assert main_window.axes.get_xlim() == (10.0, 50.0)


def test_close_active_spectrum_preserves_the_current_view(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.active = False
    spectrum_b.active = True
    main_window.axes.set_xlim(10, 50)

    main_window._close_active_spectrum()

    assert main_window.axes.get_xlim() == (10.0, 50.0)


def test_toggling_log_scale_preserves_the_current_view(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window.axes.set_xlim(10, 50)

    main_window._on_log_scale_toggled(True)

    assert main_window.axes.get_xlim() == (10.0, 50.0)


def test_opening_n42_file_with_calibration_auto_activates_it(qapp):
    main_window = MainWindow()
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "316-2_160V_0785uA.n42")

    main_window._load_files([fixture])

    assert main_window._calibration_active is True
    assert main_window._calibration.kind == "quadratic"
    assert main_window._calibration.a == pytest.approx(-11.3498272291349)
    assert main_window.calibration_toggle_action.isChecked() is True
    assert main_window.calibration_active_menu_action.isChecked() is True


def test_opening_n42_file_does_not_override_an_already_active_calibration(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "316-2_160V_0785uA.n42")
    main_window._apply_calibration_change(Calibration(kind="linear", a=99.0, b=1.0), True)

    main_window._load_files([fixture])

    assert main_window._calibration.kind == "linear"
    assert main_window._calibration.a == 99.0


def test_spectrum_list_rebuild_does_not_leak_button_groups(qapp):
    # QButtonGroup(self) gives the main window C++ ownership, so rebinding
    # self.active_button_group alone left every previous group alive as a
    # child of the window -- unbounded growth across a session, one per
    # add/remove (v3.1.0 audit, Minor). Counts real QObject children
    # rather than trusting Python refcounts, since that is exactly where
    # the old reasoning went wrong.
    import gc

    from PySide6.QtWidgets import QButtonGroup

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    for _ in range(20):
        main_window._update_spectrum_list()
    gc.collect()

    groups = [c for c in main_window.children() if isinstance(c, QButtonGroup)]
    assert len(groups) == 1, f"expected exactly one live QButtonGroup, found {len(groups)}"
    assert groups[0] is main_window.active_button_group


def test_spectrum_list_still_functional_after_rebuilds(qapp):
    # Guards the leak fix from the other direction: setParent(None) must
    # detach only the OLD group, leaving the current one wired up.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._update_spectrum_list()

    assert len(main_window.active_button_group.buttons()) == len(main_window.spectra)
    assert spectrum.active is True


def _write_n42_with_calibration(tmp_path, name="cal.n42"):
    xml_text = """<?xml version="1.0"?>
<RadInstrumentData xmlns="http://physics.nist.gov/N42/2011/N42">
  <EnergyCalibration id="EnergyCalibration-1">
    <CoefficientValues>10.0 0.5</CoefficientValues>
  </EnergyCalibration>
  <RadMeasurement id="RadMeasurement-1">
    <Spectrum id="Spectrum-1" energyCalibrationReference="EnergyCalibration-1">
      <ChannelData compressionCode="None">0 1 2 3 4 5</ChannelData>
    </Spectrum>
  </RadMeasurement>
</RadInstrumentData>
"""
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_text)
    return path


def test_loading_a_calibrated_n42_redraws_once_with_the_spectrum_present(qapp, tmp_path):
    # The embedded calibration used to be applied inside
    # _try_load_spectrum, i.e. before the new spectrum was appended --
    # so _apply_calibration_change redrew a plot that did not yet contain
    # the file that supplied the calibration, and _load_files then redrew
    # again (v3.1.0 audit, Minor). Asserts both halves: exactly one
    # redraw, and self.spectra already complete when it happens.
    main_window = MainWindow()
    _make_active_spectrum(main_window)  # a pre-existing spectrum

    calls = []
    real_plot_data = main_window._plot_data

    def counting_plot_data(*args, **kwargs):
        calls.append(len(main_window.spectra))
        return real_plot_data(*args, **kwargs)

    main_window._plot_data = counting_plot_data
    main_window._load_files([_write_n42_with_calibration(tmp_path)])

    assert len(calls) == 1, f"expected a single redraw, got {len(calls)}"
    assert calls[0] == 2, "redraw must happen with the newly loaded spectrum already in the list"
    assert main_window._calibration is not None
    assert main_window._calibration_active is True


def test_calibrated_n42_still_applies_its_calibration_when_loaded_first(qapp, tmp_path):
    # The empty-list case never showed the double redraw (the old code's
    # _apply_calibration_change no-ops its replot when self.spectra is
    # empty), so this guards that moving the call didn't break the
    # auto-apply itself.
    main_window = MainWindow()
    main_window._load_files([_write_n42_with_calibration(tmp_path)])

    assert len(main_window.spectra) == 1
    assert main_window._calibration is not None
    assert main_window._calibration_active is True


# --- incremental spectrum-list updates (v3.1.0 audit, M16) -------------


def _list_state(main_window):
    """Everything about the spectrum list that a full rebuild determines."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox, QLabel, QRadioButton

    rows = []
    for i in range(main_window.spectrum_list.count()):
        item = main_window.spectrum_list.item(i)
        widget = main_window.spectrum_list.itemWidget(item)
        rows.append({
            "path": item.data(Qt.ItemDataRole.UserRole),
            "checkboxes": [
                c.isChecked() for c in widget.findChildren(QCheckBox)
                if not isinstance(c, QRadioButton)
            ],
            "radios": [r.isChecked() for r in widget.findChildren(QRadioButton)],
            "labels": [lbl.text() for lbl in widget.findChildren(QLabel) if lbl.text()],
        })
    group = main_window.active_button_group
    return {
        "rows": rows,
        "buttons": len(group.buttons()),
        "checked": sum(1 for b in group.buttons() if b.isChecked()),
        "exclusive": group.exclusive(),
    }


def _add_spectrum(main_window, name, active=False):
    spectrum = LoadedSpectrum(name, np.zeros(32, dtype=np.int64), "#1f77b4")
    spectrum.active = active
    main_window.spectra.append(spectrum)
    return spectrum


def test_incremental_add_matches_a_full_rebuild(qapp):
    # The whole basis for appending instead of rebuilding: the resulting
    # list must be indistinguishable from the rebuilt one.
    main_window = MainWindow()
    for i in range(5):
        for s in main_window.spectra:
            s.active = False
        spectrum = _add_spectrum(main_window, f"s{i}.txt", active=True)
        main_window._append_spectrum_row(spectrum)
        main_window._sync_active_radios()

    incremental = _list_state(main_window)
    main_window._update_spectrum_list()
    assert incremental == _list_state(main_window)


def test_incremental_removal_matches_a_full_rebuild(qapp):
    main_window = MainWindow()
    for i in range(5):
        _add_spectrum(main_window, f"r{i}.txt", active=(i == 3))
    main_window._update_spectrum_list()

    # Removing the ACTIVE spectrum promotes spectra[0], so the checked
    # radio has to move -- the case incremental removal most easily gets
    # wrong.
    main_window._remove_spectrum("r3.txt")

    incremental = _list_state(main_window)
    main_window._update_spectrum_list()
    assert incremental == _list_state(main_window)
    assert main_window.spectra[0].active is True
    assert incremental["checked"] == 1


def test_adding_a_spectrum_does_not_rebuild_existing_rows(qapp):
    # The actual point of M16: existing rows must survive an add
    # untouched. Compares widget identity, so reverting to a full
    # rebuild fails this even though the visible state would look right.
    main_window = MainWindow()
    for i in range(3):
        _add_spectrum(main_window, f"keep{i}.txt", active=(i == 0))
    main_window._update_spectrum_list()

    before = [
        main_window.spectrum_list.itemWidget(main_window.spectrum_list.item(i))
        for i in range(main_window.spectrum_list.count())
    ]

    new = _add_spectrum(main_window, "added.txt")
    main_window._append_spectrum_row(new)
    main_window._sync_active_radios()

    after = [
        main_window.spectrum_list.itemWidget(main_window.spectrum_list.item(i))
        for i in range(main_window.spectrum_list.count())
    ]
    assert len(after) == len(before) + 1
    for original, current in zip(before, after):
        assert original is current, "an existing row widget was reconstructed by an append"


def test_removing_every_spectrum_leaves_an_empty_consistent_list(qapp):
    main_window = MainWindow()
    for i in range(3):
        _add_spectrum(main_window, f"e{i}.txt", active=(i == 0))
    main_window._update_spectrum_list()

    for i in range(3):
        main_window._remove_spectrum(f"e{i}.txt")

    state = _list_state(main_window)
    assert state["rows"] == []
    assert state["buttons"] == 0
    main_window._update_spectrum_list()
    assert state == _list_state(main_window)


def test_removing_an_unknown_path_falls_back_to_a_full_rebuild(qapp):
    # _remove_spectrum_row returns False when the list and self.spectra
    # have diverged; the caller must rebuild rather than leave a stale
    # list behind.
    main_window = MainWindow()
    for i in range(3):
        _add_spectrum(main_window, f"f{i}.txt", active=(i == 0))
    main_window._update_spectrum_list()

    assert main_window._remove_spectrum_row("not-in-the-list.txt") is False
    assert main_window.spectrum_list.count() == 3


def test_a_zoom_burst_coalesces_into_a_single_render(qapp):
    """A mouse wheel emits events far faster than a full canvas render
    completes. With a synchronous canvas.draw() each tick rendered
    separately, so the view lagged the wheel by one full render per tick
    -- tolerable on a GPU desktop, not on a software renderer (WSLg
    passes through no GPU at all), which is where this was reported.

    Asserts the RENDER COUNT rather than elapsed time, so it means the
    same thing on any machine.
    """
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    renders = []
    real_draw = main_window.canvas.draw
    main_window.canvas.draw = lambda *a, **k: (renders.append(1), real_draw(*a, **k))[1]

    for _ in range(10):
        main_window._zoom_x(0.8)
    qapp.processEvents()

    assert len(renders) == 1, (
        f"expected a 10-tick zoom burst to coalesce into one render, got {len(renders)} "
        "-- has canvas.draw() replaced draw_idle() in _zoom_x?"
    )


def test_zoom_still_actually_changes_the_view(qapp):
    # Guard from the other side: coalescing must not mean the zoom is
    # skipped -- the limits still have to move on every tick.
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._show_full_spectrum()

    before = main_window.axes.get_xlim()
    main_window._zoom_x(0.5)
    after = main_window.axes.get_xlim()

    assert abs(after[1] - after[0]) < abs(before[1] - before[0])


def test_right_drag_pans_x_keeping_span_and_refitting_y(qapp):
    """Right-drag walks along the spectrum at the current zoom: the X
    span is preserved and Y rescales to whatever is now visible, so a
    small peak is not left flattened by a tall one that has scrolled off.
    """
    from matplotlib.backend_bases import MouseEvent

    main_window = MainWindow()
    main_window.resize(900, 600)
    main_window.show()
    qapp.processEvents()

    length = 4000
    data = np.full(length, 10, dtype=np.int64)
    data[500] = 5000      # tall peak, visible at the start
    data[3000] = 200      # small peak, only visible after panning
    spectrum = LoadedSpectrum("pan.txt", data, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()
    main_window.axes.set_xlim(300, 800)
    main_window._autoscale_y((300, 800))

    before_x = main_window.axes.get_xlim()
    before_top = main_window.axes.get_ylim()[1]

    def px(xdata):
        return float(main_window.axes.transData.transform((xdata, 0))[0])

    ymid = float(main_window.axes.bbox.y0 + main_window.axes.bbox.height / 2)
    main_window._on_pan_press(
        MouseEvent("button_press_event", main_window.canvas, px(700), ymid, button=3)
    )
    main_window._on_mouse_move(
        MouseEvent("motion_notify_event", main_window.canvas, px(400), ymid)
    )
    main_window._on_pan_release(
        MouseEvent("button_release_event", main_window.canvas, px(400), ymid, button=3)
    )

    after_x = main_window.axes.get_xlim()
    after_top = main_window.axes.get_ylim()[1]

    assert after_x[0] > before_x[0], "dragging left should walk the view right"
    assert abs((after_x[1] - after_x[0]) - (before_x[1] - before_x[0])) < 1e-6
    assert after_top < before_top, "Y must rescale once the tall peak scrolls off"


def test_pan_stops_at_the_edge_of_the_data(qapp):
    from matplotlib.backend_bases import MouseEvent

    main_window = MainWindow()
    main_window.resize(900, 600)
    main_window.show()
    qapp.processEvents()
    _make_active_spectrum(main_window)          # 200 channels
    length = len(main_window.spectra[0].data)
    main_window._plot_data()
    main_window.axes.set_xlim(20, 70)
    span = 50

    def px(xdata):
        return float(main_window.axes.transData.transform((xdata, 0))[0])

    ymid = float(main_window.axes.bbox.y0 + main_window.axes.bbox.height / 2)
    # Drag far past the left edge in several strokes.
    for _ in range(10):
        main_window._on_pan_press(
            MouseEvent("button_press_event", main_window.canvas, px(30), ymid, button=3)
        )
        main_window._on_mouse_move(
            MouseEvent("motion_notify_event", main_window.canvas, px(30) + 300, ymid)
        )
        main_window._on_pan_release(
            MouseEvent("button_release_event", main_window.canvas, px(30) + 300, ymid, button=3)
        )

    lo, hi = main_window.axes.get_xlim()
    assert lo >= -1e-6, f"panned past channel 0 (lo={lo})"
    assert abs((hi - lo) - span) < 1e-6, "span must survive clamping at the edge"


def _drag(main_window, from_x, to_x, button=1):
    from matplotlib.backend_bases import MouseEvent

    ymid = float(main_window.axes.bbox.y0 + main_window.axes.bbox.height / 2)
    px = lambda v: float(main_window.axes.transData.transform((v, 0))[0])
    main_window._on_pan_press(
        MouseEvent("button_press_event", main_window.canvas, px(from_x), ymid, button=button))
    main_window._on_mouse_move(
        MouseEvent("motion_notify_event", main_window.canvas, px(to_x), ymid))


def _ready_window(qapp):
    main_window = MainWindow()
    main_window.resize(900, 600)
    main_window.show()
    qapp.processEvents()
    _make_active_spectrum(main_window)
    main_window._plot_data()
    main_window.axes.set_xlim(20, 70)
    return main_window


def test_left_drag_pans_the_main_window(qapp):
    """v4.0.1: a bare left drag pans, the same gesture the right button
    already had. It was free to take -- fit_mode.on_click returns early
    without a held key, so a plain left press did nothing at all."""
    main_window = _ready_window(qapp)
    before = main_window.axes.get_xlim()

    _drag(main_window, 40, 30)

    after = main_window.axes.get_xlim()
    assert after != before, "a bare left drag must pan"
    # X span preserved exactly -- panning walks along, it does not zoom.
    assert (after[1] - after[0]) == pytest.approx(before[1] - before[0])
    # Dragging left moves the view right: the content follows the cursor.
    assert after[0] > before[0]


def test_a_held_marking_key_suppresses_left_drag_panning(qapp):
    """Marking always wins. Marking is precise work and must not depend on
    how steady the hand is, so a drag with B held places background marks
    and never nudges the view."""
    main_window = _ready_window(qapp)
    before = main_window.axes.get_xlim()

    main_window.fit_controller._held_key = "b"
    _drag(main_window, 40, 30)

    assert main_window.axes.get_xlim() == before


def test_left_drag_stands_down_while_the_toolbar_owns_the_button(qapp):
    """The matplotlib toolbar's own Pan and Zoom tools drive the left
    button themselves; both acting on one drag would fight."""
    main_window = _ready_window(qapp)
    before = main_window.axes.get_xlim()

    main_window.nav_toolbar.mode = "pan/zoom"
    _drag(main_window, 40, 30)

    assert main_window.axes.get_xlim() == before


def test_right_drag_still_pans(qapp):
    """The v3.1.3 gesture is unchanged -- removing it would break the
    habit of anyone already using it, and it works with a marking key
    held, where the left button deliberately does not."""
    main_window = _ready_window(qapp)
    before = main_window.axes.get_xlim()
    _drag(main_window, 40, 30, button=3)
    assert main_window.axes.get_xlim() != before

    main_window.axes.set_xlim(*before)
    main_window.fit_controller._held_key = "b"
    _drag(main_window, 40, 30, button=3)
    assert main_window.axes.get_xlim() != before


# ---------------------------------------------------------------------------
# v4.1.0 audit S3/S4: Multiply, Rebin and Normalize must carry a derived
# spectrum's propagated variance through with the counts.
# ---------------------------------------------------------------------------


def _derived_spectrum(main_window, length=200, path="cut.spk"):
    """A spectrum standing in for a matrix cut: real counts plus a
    propagated variance that deliberately exceeds them, which is what
    makes it distinguishable from a Poisson assumption."""
    data = np.full(length, 40, dtype=np.int64)
    variance = np.full(length, 160.0)  # 4x the counts, as a cut's would be
    spectrum = LoadedSpectrum(path, data, "#1f77b4", variance=variance)
    for existing in main_window.spectra:
        existing.active = False
    spectrum.active = True
    main_window.spectra.append(spectrum)
    return spectrum


def test_multiply_scales_the_propagated_variance_by_the_factor_squared(qapp):
    main_window = MainWindow()
    spectrum = _derived_spectrum(main_window)
    main_window._apply_multiply(spectrum, 3.0)
    assert np.allclose(spectrum.data, 120)
    assert np.allclose(spectrum.variance, 160.0 * 9.0), (
        "variance must scale as factor**2, or every later fit reports "
        "uncertainties smaller than the data supports"
    )


def test_multiply_leaves_a_poisson_spectrum_without_a_variance(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    assert spectrum.variance is None
    main_window._apply_multiply(spectrum, 2.0)
    assert spectrum.variance is None


def test_rebin_rebins_the_propagated_variance_to_match(qapp):
    main_window = MainWindow()
    spectrum = _derived_spectrum(main_window)
    main_window._apply_rebin(spectrum, 2)
    assert spectrum.variance is not None
    assert spectrum.variance.size == spectrum.data.size, (
        "a length mismatch here makes the spectrum permanently unfittable"
    )
    assert np.allclose(spectrum.variance, 320.0)  # two 160.0 channels added


def test_rebinned_derived_spectrum_can_still_be_fitted(qapp):
    """The user-visible symptom the length mismatch caused: fit_peaks
    rejects a variance whose length does not match the spectrum, so a
    rebinned cut could not be fitted at all."""
    from peak_fit import channel_indices, fit_peaks

    main_window = MainWindow()
    data = np.full(400, 40, dtype=np.int64)
    peak = (3000.0 * np.exp(
        -((np.arange(400) - 200.0) ** 2) / (2 * 8.0 ** 2))).astype(np.int64)
    data = data + peak
    spectrum = LoadedSpectrum(
        "cut.spk", data, "#1f77b4", variance=np.maximum(data, 0) * 2.0)
    for existing in main_window.spectra:
        existing.active = False
    spectrum.active = True
    main_window.spectra.append(spectrum)

    main_window._apply_rebin(spectrum, 2)

    x = channel_indices(len(spectrum.data))
    result = fit_peaks(
        x, spectrum.data.astype(float),
        left_bg_region=(10.0, 40.0), right_bg_region=(160.0, 190.0),
        fit_region=(80.0, 120.0), peak_positions=[100.0],
        variance=spectrum.variance,
    )
    assert result.peaks[0].area > 0.0


def test_rebin_leaves_a_poisson_spectrum_without_a_variance(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._apply_rebin(spectrum, 2)
    assert spectrum.variance is None


def test_normalize_scales_the_propagated_variance_too(qapp):
    """Normalize scales through the same multiply() call, so it carries
    the same obligation."""
    main_window = MainWindow()
    small = _derived_spectrum(main_window, path="small.spk")
    big = LoadedSpectrum("big.spk", np.full(200, 120, dtype=np.int64), "#ff7f0e")
    main_window.spectra.append(big)
    # Normalize needs a marked channel to read each spectrum's reference
    # value from; channel 1 reads 40 and 120, so `small` is scaled by 3.
    main_window.fit_controller.state.pending_fit_click = 1
    main_window._normalize_spectra()
    # `small` was scaled up by 3 to match `big`, so its variance must have
    # gone up by 9.
    assert np.allclose(small.data, 120)
    assert np.allclose(small.variance, 160.0 * 9.0)
