"""Clearing an energy calibration, and what that is NOT.

The app already had Ctrl+T, which switches the axis back to Channel while
keeping the coefficients so it can be switched on again. Clear Calibration
is the other thing: discard the coefficients entirely, back to the state
before any calibration was set.

Two menu entries that sound alike is a real risk, so these tests pin the
difference rather than only the new behaviour -- if the two ever converge,
one of them has lost its reason to exist.
"""

import numpy as np

from calibration import Calibration
from main_window import MainWindow
from spectrum import LoadedSpectrum


def _window(qapp):
    window = MainWindow()
    spectrum = LoadedSpectrum("eu.txt", np.full(512, 100.0), "#FF0000")
    spectrum.active = True
    window.spectra.append(spectrum)
    window._apply_calibration_change(Calibration("linear", 50.0, 0.65), True)
    return window


def test_clearing_discards_the_coefficients(qapp, monkeypatch):
    monkeypatch.setattr("main_window.QMessageBox.question",
                        staticmethod(lambda *a, **k: _yes()))
    window = _window(qapp)
    assert window._calibration is not None

    window.clear_calibration()

    assert window._calibration is None, "the coefficients survived"
    assert window._calibration_active is False
    assert window.axes.get_xlabel() == "Channel"
    window.close()


def test_clearing_greys_out_the_toggle(qapp, monkeypatch):
    """The visible difference from Ctrl+T: afterwards there is nothing left
    to switch back on."""
    monkeypatch.setattr("main_window.QMessageBox.question",
                        staticmethod(lambda *a, **k: _yes()))
    window = _window(qapp)
    assert window.calibration_active_menu_action.isEnabled()

    window.clear_calibration()

    assert not window.calibration_active_menu_action.isEnabled()
    window.close()


def test_the_toggle_still_keeps_the_coefficients(qapp):
    """Control, and the whole reason both actions exist. Ctrl+T must NOT
    become a second Clear: it reverts the axis and keeps the calibration."""
    window = _window(qapp)

    window._apply_calibration_change(window._calibration, False)

    assert window._calibration is not None, "Ctrl+T discarded the calibration"
    assert window._calibration_active is False
    assert window.axes.get_xlabel() == "Channel"
    assert window.calibration_active_menu_action.isEnabled(), (
        "the toggle must stay usable -- there is still something to toggle")
    window.close()


def test_clearing_asks_first(qapp, monkeypatch):
    """It throws away coefficients that may have taken real work to
    produce, and unlike Ctrl+T there is no undo."""
    asked = {}

    def _question(*args, **kwargs):
        asked["yes"] = True
        return _no()

    monkeypatch.setattr("main_window.QMessageBox.question",
                        staticmethod(_question))
    window = _window(qapp)

    window.clear_calibration()

    assert asked.get("yes"), "cleared without asking"
    assert window._calibration is not None, "declining still cleared it"
    window.close()


def test_clearing_with_no_calibration_does_nothing(qapp, monkeypatch):
    """Nothing to discard, so nothing to confirm either."""
    asked = {}
    monkeypatch.setattr("main_window.QMessageBox.question",
                        staticmethod(lambda *a, **k: asked.setdefault("yes", True) or _yes()))
    window = MainWindow()
    window.clear_calibration()
    assert not asked, "asked about discarding a calibration that is not there"
    window.close()


def test_the_two_actions_explain_how_they_differ(qapp):
    """Two similar names in one menu need their tooltips to separate them,
    or the pair is worse than either alone."""
    window = _window(qapp)
    toggle = window.calibration_active_menu_action.toolTip().lower()
    clear = window.clear_calibration_action.toolTip().lower()

    assert "kept" in toggle or "keep" in toggle, toggle
    assert "discard" in clear, clear
    assert toggle != clear
    window.close()


def _yes():
    from PySide6.QtWidgets import QMessageBox

    return QMessageBox.StandardButton.Yes


def _no():
    from PySide6.QtWidgets import QMessageBox

    return QMessageBox.StandardButton.No
