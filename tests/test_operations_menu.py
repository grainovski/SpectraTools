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
