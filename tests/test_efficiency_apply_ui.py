"""Applying an efficiency to a spectrum, from the interface."""

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


def _window(qapp, calibrated=True, channels=512):
    window = MainWindow()
    window._calibration = Calibration("linear", 50.0, 0.65)
    window._calibration_active = calibrated
    spectrum = LoadedSpectrum("source.txt",
                              np.full(channels, 1000.0), "#FF0000")
    spectrum.active = True
    window.spectra.append(spectrum)
    return window, spectrum


def test_applying_adds_a_new_spectrum_and_keeps_the_original(qapp, result):
    window, original = _window(qapp)
    before = len(window.spectra)
    untouched = original.data.copy()

    window.apply_efficiency(original, result)

    assert len(window.spectra) == before + 1
    assert original.data == pytest.approx(untouched), \
        "the original spectrum was modified"
    assert "eff" in window.spectra[-1].path.lower()
    window.close()


def test_the_new_spectrum_is_named_for_the_model_used(qapp, result):
    """Two corrections of one spectrum under the two models must be
    distinguishable afterwards, or the plot is ambiguous."""
    window, original = _window(qapp)
    window.apply_efficiency(original, result)
    assert "KRF" in window.spectra[-1].path

    result.model = "rw"
    window.apply_efficiency(original, result)
    assert "RW" in window.spectra[-1].path
    result.model = "krf"
    window.close()


def test_counts_only_rise(qapp, result):
    """The applied curve is normalised to peak at 1, so dividing by it can
    never lower a count. A correction that lowered them would mean the
    normalisation had been lost."""
    window, original = _window(qapp)
    window.apply_efficiency(original, result)
    corrected = window.spectra[-1].data
    assert np.all(corrected[corrected > 0] >= 1000.0 - 1e-9)
    window.close()


def test_applying_needs_an_active_calibration(qapp, result):
    """Without one, bins have no energies and the correction is undefined."""
    window, _original = _window(qapp, calibrated=False)
    assert not window.can_apply_efficiency()
    window.close()


def test_an_active_calibration_allows_it(qapp, result):
    """Control for the test above."""
    window, _original = _window(qapp, calibrated=True)
    assert window.can_apply_efficiency()
    window.close()


def test_the_corrected_spectrum_is_finite_everywhere(qapp, result, monkeypatch):
    """The curve is extrapolated without restriction, so the guarantee that
    nothing non-finite reaches a LoadedSpectrum has to hold here too, not
    only inside efficiency_apply."""
    window, original = _window(qapp)
    # A calibration reaching far past the fitted range, and starting below
    # zero energy, where both models fail in different ways.
    window._calibration = Calibration("linear", -30.0, 4.0)
    # That calibration pushes bins outside the fitted range, so
    # apply_efficiency reports zeroed bins and pops a QMessageBox.information
    # -- harmless with a user at the keyboard, but a modal dialog nobody can
    # dismiss hangs a headless test forever. Every other test in this suite
    # that crosses a QMessageBox call patches it the same way; see
    # test_operations_menu.py.
    monkeypatch.setattr("main_window.QMessageBox.information",
                        lambda *a, **k: None)
    window.apply_efficiency(original, result)
    assert np.all(np.isfinite(window.spectra[-1].data))
    window.close()


def test_the_variance_is_corrected_with_the_counts(qapp, result):
    """Leaving it alone would give a spectrum whose stored variance no
    longer matches its counts, which silently corrupts any later fit."""
    window, original = _window(qapp)
    original.variance = np.full(len(original.data), 1000.0)
    window.apply_efficiency(original, result)
    new = window.spectra[-1]
    assert getattr(new, "variance", None) is not None
    assert not np.allclose(new.variance, 1000.0), \
        "the variance was carried over unchanged"
    window.close()
