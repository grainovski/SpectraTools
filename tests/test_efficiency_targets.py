"""Choosing what an efficiency is applied to, and keeping it around.

Three things that go together: the fitted efficiency outlives the window it
was produced in, the window can aim at any loaded spectrum rather than only
the active one, and it can correct them all at once.
"""

import os

import numpy as np
import pytest

from calibration import Calibration
from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo
from main_window import MainWindow
from spectrum import LoadedSpectrum

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


@pytest.fixture(scope="module")
def result():
    data = np.loadtxt(os.path.join(FIXTURES, "demo1.txt"), ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    return EfficiencyResult(
        fit=fit, mc=run_monte_carlo(fit, N, dN, I, dI, iterations=200),
        model="krf", calibration=Calibration("linear", 50.0, 0.65))


def _window(qapp, names=("a.txt", "b.txt", "c.txt")):
    window = MainWindow()
    window._calibration = Calibration("linear", 50.0, 0.65)
    window._calibration_active = True
    for i, name in enumerate(names):
        s = LoadedSpectrum(name, np.full(256, 1000.0), "#FF0000")
        s.active = (i == 0)
        window.spectra.append(s)
    return window


def _dialog(window, result, monkeypatch):
    from efficiency_dialog import EfficiencyDialog

    monkeypatch.setattr("main_window.QMessageBox.information",
                        staticmethod(lambda *a, **k: None))
    d = EfficiencyDialog(window, result, np.full(len(result.fit.E), 0.01))
    return d


# --- the efficiency outlives its window ---------------------------------


def test_the_main_window_has_no_efficiency_until_one_is_fitted(qapp):
    window = MainWindow()
    assert window._efficiency is None
    assert not window.show_efficiency_action.isEnabled()
    window.close()


def test_storing_one_enables_reopening_it(qapp, result):
    window = _window(qapp)
    window.set_efficiency(result)
    assert window._efficiency is result
    assert window.show_efficiency_action.isEnabled()
    window.close()


def test_show_efficiency_reopens_a_window_on_the_stored_result(qapp, result):
    """The point of holding it: close the results window, come back later,
    and apply it to something without refitting."""
    window = _window(qapp)
    window.set_efficiency(result)
    window.show_efficiency()
    assert window._efficiency_dialog is not None
    assert window._efficiency_dialog.result is result
    window._efficiency_dialog.close()
    window.close()


# --- aiming at a spectrum other than the active one ----------------------


def test_the_picker_lists_every_loaded_spectrum(qapp, result, monkeypatch):
    window = _window(qapp)
    d = _dialog(window, result, monkeypatch)
    listed = [d.target_combo.itemText(i) for i in range(d.target_combo.count())]
    assert listed == ["a.txt", "b.txt", "c.txt"], listed
    d.close(); window.close()


def test_the_picker_starts_on_the_active_spectrum(qapp, result, monkeypatch):
    window = _window(qapp)
    window.spectra[0].active = False
    window.spectra[2].active = True
    d = _dialog(window, result, monkeypatch)
    assert d.target_combo.currentText() == "c.txt"
    d.close(); window.close()


def test_applying_corrects_the_PICKED_spectrum_not_the_active_one(qapp,
                                                                  result,
                                                                  monkeypatch):
    """The whole reason the picker exists. Correcting the active spectrum
    regardless would satisfy every other test here."""
    window = _window(qapp)
    assert window.spectra[0].active and window.spectra[0].path == "a.txt"

    d = _dialog(window, result, monkeypatch)
    d.target_combo.setCurrentIndex(1)          # b.txt
    d._on_apply()

    assert "b.txt" in window.spectra[-1].path, window.spectra[-1].path
    assert "a.txt" not in window.spectra[-1].path
    d.close(); window.close()


def test_the_picker_follows_spectra_added_while_it_is_open(qapp, result,
                                                           monkeypatch):
    """Applying adds a spectrum, so the list the user is looking at goes
    stale the moment they use it."""
    window = _window(qapp)
    d = _dialog(window, result, monkeypatch)
    before = d.target_combo.count()

    d._on_apply()
    d.refresh_targets()

    assert d.target_combo.count() > before
    d.close(); window.close()


# --- all at once ---------------------------------------------------------


def test_apply_to_all_corrects_every_spectrum(qapp, result, monkeypatch):
    window = _window(qapp)
    d = _dialog(window, result, monkeypatch)

    d._on_apply_all()

    corrected = [s.path for s in window.spectra if "eff-corrected" in s.path]
    assert len(corrected) == 3, corrected
    for name in ("a.txt", "b.txt", "c.txt"):
        assert any(name in p for p in corrected), name
    d.close(); window.close()


def test_apply_to_all_does_not_correct_a_correction(qapp, result, monkeypatch):
    """Dividing an already-corrected spectrum by the efficiency a second
    time is physically meaningless, and pressing the button twice is an easy
    thing to do. The corrections are skipped and the count reported."""
    window = _window(qapp)
    d = _dialog(window, result, monkeypatch)

    d._on_apply_all()
    after_first = len(window.spectra)
    d.refresh_targets()
    d._on_apply_all()

    assert len(window.spectra) == after_first, (
        "a second Apply to all corrected the corrections")
    d.close(); window.close()


def test_a_corrected_spectrum_is_marked_as_one(qapp, result, monkeypatch):
    """The skip above keys on a flag rather than on the name, because a
    name is something the user can change."""
    window = _window(qapp)
    d = _dialog(window, result, monkeypatch)
    d._on_apply()
    assert getattr(window.spectra[-1], "efficiency_corrected", False) is True
    assert getattr(window.spectra[0], "efficiency_corrected", False) is False
    d.close(); window.close()
